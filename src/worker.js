import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parentPort, workerData } from 'node:worker_threads';
import init, { Miner, version } from '../vendor/miner/hash_miner.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.resolve(__dirname, '../vendor/miner/hash_miner_bg.wasm');

let miner = null;
let running = false;
let counter = 0n;
let totalHashes = 0;
let smoothHashrate = 0;
let startedAt = 0;
let lastProgressAt = 0;
let batchSize = BigInt(workerData.batchSize ?? 0); // 0 = auto-calibrate

/*
 * Auto-calibrate batchSize so each searchLoop tick takes ~TARGET_MS milliseconds.
 * This keeps the event-loop responsive while maximising throughput.
 * We probe with a small batch, measure, then scale linearly.
 */
const TARGET_BATCH_MS  = 50;   // aim for 50 ms per tick
const CALIBRATE_HASHES = 50_000n;

function calibrateBatchSize() {
  if (batchSize !== 0n) return; // already set by caller
  const t0  = performance.now();
  miner.search(0n, CALIBRATE_HASHES);          // throw-away warmup/probe
  const dt  = performance.now() - t0;
  const hps = dt > 0 ? Number(CALIBRATE_HASHES) / (dt / 1000) : 500_000;
  const optimal = Math.max(10_000, Math.round(hps * (TARGET_BATCH_MS / 1000)));
  batchSize = BigInt(optimal);
  post({ type: 'calibrated', id: workerData.id, batchSize: optimal, hashrate: Math.round(hps) });
}

function post(msg) {
  parentPort.postMessage(msg);
}

function hexToBytes(hex, expectedLength) {
  const h = hex.startsWith('0x') ? hex.slice(2) : hex;
  if (h.length !== expectedLength * 2) {
    throw new Error(`expected ${expectedLength} bytes, got ${h.length / 2}`);
  }
  const out = new Uint8Array(expectedLength);
  for (let i = 0; i < expectedLength; i++) out[i] = Number.parseInt(h.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function bytesToHex(bytes) {
  let out = '0x';
  for (const b of bytes) out += b.toString(16).padStart(2, '0');
  return out;
}

function makeMiner({ challenge, difficulty, noncePrefix }) {
  miner?.free?.();
  miner = new Miner(
    hexToBytes(challenge, 32),
    hexToBytes(difficulty, 32),
    hexToBytes(noncePrefix, 24),
  );
  counter = 0n;
  totalHashes = 0;
  smoothHashrate = 0;
  startedAt = performance.now();
  lastProgressAt = startedAt;
}

function sendProgress(now = performance.now()) {
  post({
    type: 'progress',
    hashes: totalHashes,
    hashrate: smoothHashrate,
    elapsedMs: Math.round(now - startedAt),
  });
  lastProgressAt = now;
}

function searchLoop() {
  if (!running || !miner) return;
  const iterations = batchSize;
  const before = performance.now();
  const hit = miner.search(counter, iterations);
  const after = performance.now();

  const iterNum = Number(iterations);
  totalHashes += iterNum;
  counter += iterations;

  const dt = after - before;
  const instant = dt > 0 ? iterNum / (dt / 1000) : 0;
  smoothHashrate = smoothHashrate === 0 ? instant : smoothHashrate + 0.25 * (instant - smoothHashrate);

  if (after - lastProgressAt > 500) sendProgress(after);

  if (hit) {
    running = false;
    post({
      type: 'found',
      nonce: bytesToHex(hit.nonce),
      result: bytesToHex(hit.result),
      hashes: totalHashes,
      elapsedMs: Math.round(after - startedAt),
    });
    return;
  }

  setImmediate(searchLoop);
}

async function start() {
  await init({ module_or_path: fs.readFileSync(wasmPath) });
  makeMiner(workerData);
  calibrateBatchSize();   // sets batchSize if auto (0)
  post({ type: 'ready', id: workerData.id, version: version() });
  running = true;
  searchLoop();
}

parentPort.on('message', (msg) => {
  try {
    if (msg.type === 'stop') {
      const wasRunning = running;
      running = false;
      if (wasRunning) {
        post({ type: 'stopped', hashes: totalHashes, elapsedMs: Math.round(performance.now() - startedAt) });
      }
    } else if (msg.type === 'retarget') {
      const oldPrefix = workerData.noncePrefix;
      makeMiner({ challenge: msg.challenge, difficulty: msg.difficulty, noncePrefix: oldPrefix });
      running = true;
      searchLoop();
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
});

start().catch((err) => {
  post({ type: 'error', message: err instanceof Error ? err.stack || err.message : String(err) });
});
