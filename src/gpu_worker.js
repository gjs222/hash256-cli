/**
 * gpu_worker.js
 *
 * GPU mining worker thread.
 * Spawns src/gpu_miner.exe (compiled from src/gpu_miner.c, no OpenCL SDK needed).
 * Communicates via newline-delimited JSON on the exe's stdout.
 *
 * Same message protocol as worker.js:
 *   Inbound:  { type:'stop' }  |  { type:'retarget', challenge, difficulty }
 *   Outbound: ready / progress / found / stopped / error
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parentPort, workerData } from 'node:worker_threads';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/* Prefer the CUDA binary (gpu_miner_cuda) over the OpenCL binary (gpu_miner).
 * Build CUDA:  bash build-gpu-cuda.sh
 * Build OpenCL: gcc -O2 src/gpu_miner.c -ldl -o src/gpu_miner */
const GPU_EXE_CUDA = path.join(__dirname, 'gpu_miner_cuda');
const GPU_EXE_OCL  = path.join(__dirname, 'gpu_miner');
const GPU_EXE = fs.existsSync(GPU_EXE_CUDA) ? GPU_EXE_CUDA : GPU_EXE_OCL;

/* ── Config ─────────────────────────────────────────────────────────────── */
const {
  id          = 0,
  challenge,
  difficulty,
  noncePrefix,
  batchSize   = 0,
  gpuPlatform = -1,
  gpuDevice   = 0,
} = workerData;

/* ── State ───────────────────────────────────────────────────────────────── */
let running     = false;
let totalHashes = 0;
let startedAt   = 0;
let child       = null;

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function post(msg) { parentPort.postMessage(msg); }

function formatNonce(noncePrefixHex, counterStr) {
  const prefix = noncePrefixHex.startsWith('0x') ? noncePrefixHex.slice(2) : noncePrefixHex;
  const ctr    = BigInt(counterStr).toString(16).padStart(16, '0');
  return '0x' + prefix + ctr;
}

function killChild() {
  if (child) { try { child.kill(); } catch (_) {} child = null; }
}

/* ── Spawn gpu_miner exe ─────────────────────────────────────────────────── */
function spawnMiner(challenge_, difficulty_, noncePrefixHex_, startCounter_) {
  if (!fs.existsSync(GPU_EXE)) {
    const hint = process.platform === 'win32'
      ? '.\\build-gpu.ps1  (requires gcc from MSYS2 – https://www.msys2.org/)'
      : 'gcc -O2 src/gpu_miner.c -ldl -o src/gpu_miner';
    throw new Error(
      `GPU miner binary not found: ${GPU_EXE}\n` +
      `Build it first (no OpenCL SDK headers needed):\n  ${hint}`
    );
  }

  const args = [
    '--challenge',    challenge_,
    '--difficulty',   difficulty_,
    '--nonce-prefix', noncePrefixHex_,
    '--start',        String(startCounter_),
    '--progress-ms',  '2000',
  ];
  if (batchSize > 0)    args.push('--batch-size',     String(batchSize));
  if (gpuPlatform >= 0) args.push('--platform-index', String(gpuPlatform));
  args.push('--device-index', String(gpuDevice));

  const proc = spawn(GPU_EXE, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let buf = '';

  proc.stdout.on('data', (chunk) => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try { handleExeMsg(JSON.parse(line)); } catch (_) {}
    }
  });

  proc.stderr.on('data', (chunk) => {
    const txt = chunk.toString('utf8').trim();
    if (txt) post({ type: 'error', message: `gpu_miner: ${txt}` });
  });

  proc.on('error', (err) => {
    post({ type: 'error', message: `Failed to spawn gpu_miner: ${err.message}` });
  });

  proc.on('exit', (code, signal) => {
    if (running && code !== 0 && !signal)
      post({ type: 'error', message: `gpu_miner exited with code ${code}` });
    if (!running)
      post({ type: 'stopped', hashes: totalHashes, elapsedMs: Math.round(performance.now() - startedAt) });
  });

  return proc;
}

/* ── Handle JSON from exe ────────────────────────────────────────────────── */
function handleExeMsg(msg) {
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'device') {
    const bsz = msg.batch_size ? Number(msg.batch_size) : 0;
    post({
      type:    'ready',
      id,
      version: `GPU: ${msg.vendor} / ${msg.device} (${msg.cu} CU, batch=${bsz.toLocaleString()})`,
    });
    running = true;

  } else if (msg.type === 'progress') {
    totalHashes = Number(msg.hashes);
    const elapsed  = performance.now() - startedAt;
    const hashrate = elapsed > 0 ? (totalHashes / (elapsed / 1000)) : 0;
    post({ type: 'progress', hashes: totalHashes, hashrate: Math.round(hashrate), elapsedMs: Math.round(elapsed) });

  } else if (msg.type === 'found') {
    totalHashes = Number(msg.hashes);
    const nonce = formatNonce(noncePrefix, msg.counter);
    running = false;
    killChild();
    post({ type: 'found', nonce, hashes: totalHashes, elapsedMs: Math.round(performance.now() - startedAt) });

  } else if (msg.type === 'error') {
    post({ type: 'error', message: msg.message || 'unknown GPU error' });
  }
}

/* ── Start / message handler ─────────────────────────────────────────────── */
async function start() {
  startedAt = performance.now();
  try {
    child = spawnMiner(challenge, difficulty, noncePrefix, 0);
  } catch (err) {
    post({ type: 'error', message: err.message });
  }
}

parentPort.on('message', (msg) => {
  if (msg.type === 'stop') {
    running = false;
    killChild();
    post({ type: 'stopped', hashes: totalHashes, elapsedMs: Math.round(performance.now() - startedAt) });

  } else if (msg.type === 'retarget') {
    killChild();
    totalHashes = 0;
    startedAt   = performance.now();
    running     = false;
    try {
      child = spawnMiner(msg.challenge, msg.difficulty, noncePrefix, 0);
    } catch (err) {
      post({ type: 'error', message: err.message });
    }
  }
});

start();
