#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { randomBytes } from 'node:crypto';
import { Contract, JsonRpcProvider, Wallet, formatUnits, isAddress, parseUnits } from 'ethers';

const CONTRACT_ADDRESS = '0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc';
loadDotEnv();
const DEFAULT_RPC_URL = process.env.HASH256_RPC_URL || 'https://ethereum.publicnode.com';
const DEFAULT_BATCH_SIZE = 0; // 0 = auto-calibrate per worker
const ABI = [
  'function getChallenge(address miner) view returns (bytes32)',
  'function miningState() view returns (uint256 era,uint256 reward,uint256 difficulty,uint256 minted,uint256 remaining,uint256 epoch,uint256 epochBlocksLeft)',
  'function genesisState() view returns (uint256 minted,uint256 remaining,uint256 ethRaised,bool complete)',
  'function balanceOf(address account) view returns (uint256)',
  'function mine(uint256 nonce)',
];

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(__dirname, 'worker.js');
const gpuWorkerPath = path.join(__dirname, 'gpu_worker.js');
const DEFAULT_GPU_BATCH_SIZE = 4_000_000;

function loadDotEnv() {
  const envPath = path.resolve(process.cwd(), '.env');
  if (!fs.existsSync(envPath)) return;
  const text = fs.readFileSync(envPath, 'utf8');
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (process.env[key] == null) process.env[key] = value;
  }
}

function parseArgs(argv) {
  const args = {
    rpc: DEFAULT_RPC_URL,
    privateKey: process.env.HASH256_PRIVATE_KEY,
    address: undefined,
    threads: Math.max(1, os.availableParallelism?.() || os.cpus().length || 1),
    batchSize: DEFAULT_BATCH_SIZE,
    gpu: false,
    gpuPlatform: parseInt(process.env.HASH256_GPU_PLATFORM || '0', 10),
    gpuDevice: parseInt(process.env.HASH256_GPU_DEVICE || '0', 10),
    gpuBatchSize: parseInt(process.env.HASH256_GPU_BATCH_SIZE || String(DEFAULT_GPU_BATCH_SIZE), 10),
    status: false,
    benchmark: false,
    noSubmit: false,
    once: false,
    gasMultiplier: 1.5,
    minGas: 200_000n,
    maxGas: 400_000n,
    priorityFeeGwei: process.env.HASH256_PRIORITY_FEE_GWEI || '8',
    maxFeeGwei: process.env.HASH256_MAX_FEE_GWEI || '20',
    feeMultiplier: Number.parseFloat(process.env.HASH256_BASE_FEE_MULTIPLIER || '2'),
    retargetMs: 12_000,
    progressMs: 5_000,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} requires a value`);
      return argv[++i];
    };
    if (a === '--rpc') args.rpc = next();
    else if (a === '--private-key') args.privateKey = next();
    else if (a === '--address') args.address = next();
    else if (a === '--threads') args.threads = Number.parseInt(next(), 10);
    else if (a === '--batch-size') args.batchSize = Number.parseInt(next(), 10);
    else if (a === '--gpu') args.gpu = true;
    else if (a === '--gpu-platform') args.gpuPlatform = Number.parseInt(next(), 10);
    else if (a === '--gpu-device') args.gpuDevice = Number.parseInt(next(), 10);
    else if (a === '--gpu-batch-size') args.gpuBatchSize = Number.parseInt(next(), 10);
    else if (a === '--status') args.status = true;
    else if (a === '--benchmark') { args.benchmark = true; args.noSubmit = true; }
    else if (a === '--no-submit') args.noSubmit = true;
    else if (a === '--once') args.once = true;
    else if (a === '--gas-multiplier') args.gasMultiplier = Number.parseFloat(next());
    else if (a === '--min-gas') args.minGas = BigInt(next());
    else if (a === '--max-gas') args.maxGas = BigInt(next());
    else if (a === '--priority-fee-gwei') args.priorityFeeGwei = next();
    else if (a === '--max-fee-gwei') args.maxFeeGwei = next();
    else if (a === '--base-fee-multiplier') args.feeMultiplier = Number.parseFloat(next());
    else if (a === '--retarget-ms') args.retargetMs = Number.parseInt(next(), 10);
    else if (a === '--progress-ms') args.progressMs = Number.parseInt(next(), 10);
    else if (a === '-h' || a === '--help') { printHelp(); process.exit(0); }
    else throw new Error(`unknown argument: ${a}`);
  }

  if (!Number.isInteger(args.threads) || args.threads < 1) throw new Error('--threads must be >= 1');
  if (!Number.isInteger(args.batchSize) || args.batchSize < 0) throw new Error('--batch-size must be >= 0 (0 = auto)');
  if (!Number.isFinite(args.gasMultiplier) || args.gasMultiplier <= 0) throw new Error('--gas-multiplier must be > 0');
  if (!Number.isFinite(args.feeMultiplier) || args.feeMultiplier <= 0) throw new Error('--base-fee-multiplier must be > 0');
  parseGwei(args.priorityFeeGwei, '--priority-fee-gwei');
  if (args.maxFeeGwei != null) parseGwei(args.maxFeeGwei, '--max-fee-gwei');
  return args;
}

function printHelp() {
  console.log(`hash256-mine

CPU / GPU CLI miner for HASH / hash256.fun.

Usage:
  hash256-mine [options]

Options:
  --rpc <url>              Ethereum mainnet RPC (default: ${DEFAULT_RPC_URL})
  --private-key <hex>      Mining wallet private key (or HASH256_PRIVATE_KEY)
  --address <0x...>        Address for --benchmark/--status without private key
  --threads <n>            CPU worker threads (default: all CPU threads)
  --batch-size <n>         WASM search batch per worker tick (default: auto-calibrate)

GPU options (requires: npm install node-opencl):
  --gpu                    Enable OpenCL GPU mining (replaces CPU workers)
  --gpu-platform <n>       OpenCL platform index (default: 0)
  --gpu-device <n>         OpenCL device index within platform (default: 0)
  --gpu-batch-size <n>     Hashes per GPU kernel dispatch (default: ${DEFAULT_GPU_BATCH_SIZE})

  --status                 Print chain mining status and exit; no key needed
  --benchmark              Mine locally for --address/wallet but do not submit
  --no-submit              Find a nonce but do not send mine(nonce)
  --once                   Stop after one successful submitted tx
  --gas-multiplier <n>     Gas limit estimate multiplier (default: 1.5)
  --min-gas <n>            Minimum gas limit (default: 200000)
  --max-gas <n>            Maximum gas limit (default: 400000)
  --priority-fee-gwei <n>  EIP-1559 priority fee / tip (default: 8)
  --max-fee-gwei <n>       EIP-1559 max fee cap (default: 20)
  -h, --help               Show help

Examples:
  HASH256_PRIVATE_KEY=0x... hash256-mine
  hash256-mine --threads 8
  hash256-mine --gpu                                   # OpenCL GPU mining
  hash256-mine --gpu --gpu-device 1                    # second GPU
  hash256-mine --benchmark --address 0xYourAddress
  hash256-mine --status
`);
}

function hexToBigInt(hex) {
  return BigInt(hex.startsWith('0x') ? hex : `0x${hex}`);
}

function parseGwei(value, label) {
  try {
    return parseUnits(String(value), 'gwei');
  } catch {
    throw new Error(`${label} must be a valid gwei decimal number, got ${value}`);
  }
}

async function buildFeeOverrides(provider, args) {
  const maxPriorityFeePerGas = parseGwei(args.priorityFeeGwei, '--priority-fee-gwei');
  let maxFeePerGas;
  if (args.maxFeeGwei != null) {
    maxFeePerGas = parseGwei(args.maxFeeGwei, '--max-fee-gwei');
  } else {
    const block = await provider.getBlock('latest');
    const baseFee = block?.baseFeePerGas;
    if (baseFee == null) {
      const feeData = await provider.getFeeData();
      maxFeePerGas = feeData.maxFeePerGas ?? (feeData.gasPrice ? feeData.gasPrice * 2n : parseGwei('20', '--max-fee-gwei'));
    } else {
      const multiplierScaled = BigInt(Math.ceil(args.feeMultiplier * 1000));
      maxFeePerGas = (baseFee * multiplierScaled) / 1000n + maxPriorityFeePerGas;
    }
  }
  if (maxFeePerGas < maxPriorityFeePerGas) {
    throw new Error(`max fee (${formatUnits(maxFeePerGas, 'gwei')} gwei) must be >= priority fee (${formatUnits(maxPriorityFeePerGas, 'gwei')} gwei)`);
  }
  return { maxFeePerGas, maxPriorityFeePerGas };
}

function targetToExpectedHashes(difficultyHex) {
  const target = hexToBigInt(difficultyHex);
  if (target <= 0n) return Infinity;
  return Number((1n << 256n) / target);
}

function formatHashrate(hps) {
  if (!Number.isFinite(hps) || hps <= 0) return '0 H/s';
  const units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s'];
  let value = hps;
  let idx = 0;
  while (value >= 1000 && idx < units.length - 1) { value /= 1000; idx++; }
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[idx]}`;
}

function formatCount(n) {
  if (!Number.isFinite(n) || n <= 0) return '?';
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

function asBytes32Hex(value) {
  let h = typeof value === 'bigint' ? value.toString(16) : BigInt(value).toString(16);
  if (h.length > 64) throw new Error(`value too large for bytes32: ${value}`);
  return `0x${h.padStart(64, '0')}`;
}

function scaleGas(estimated, multiplier, minGas, maxGas) {
  const scaled = BigInt(Math.ceil(Number(estimated) * multiplier));
  if (scaled < minGas) return minGas;
  if (scaled > maxGas) return maxGas;
  return scaled;
}

async function getState(contract) {
  const [g, s] = await Promise.all([contract.genesisState(), contract.miningState()]);
  return {
    genesisComplete: g.complete,
    genesisMinted: g.minted,
    genesisRemaining: g.remaining,
    genesisEthRaised: g.ethRaised,
    era: s.era,
    reward: s.reward,
    difficulty: s.difficulty,
    difficultyHex: asBytes32Hex(s.difficulty),
    minted: s.minted,
    remaining: s.remaining,
    epoch: s.epoch,
    epochBlocksLeft: s.epochBlocksLeft,
  };
}

function printState(state, address, balance) {
  console.log('HASH mining status');
  console.log(`  contract:          ${CONTRACT_ADDRESS}`);
  console.log(`  genesis complete:  ${state.genesisComplete}`);
  console.log(`  era:               ${state.era.toString()}`);
  console.log(`  reward:            ${formatUnits(state.reward, 18)} HASH`);
  console.log(`  difficulty target: ${state.difficultyHex}`);
  console.log(`  epoch:             ${state.epoch.toString()} (${state.epochBlocksLeft.toString()} blocks left)`);
  console.log(`  mined:             ${formatUnits(state.minted, 18)} HASH`);
  console.log(`  remaining:         ${formatUnits(state.remaining, 18)} HASH`);
  if (address) console.log(`  miner:             ${address}`);
  if (balance != null) console.log(`  HASH balance:      ${formatUnits(balance, 18)} HASH`);
}

class MinerFarm {
  constructor({ threads, batchSize, progressMs, onFound, onError, gpu = false, gpuPlatform = 0, gpuDevice = 0, gpuBatchSize = DEFAULT_GPU_BATCH_SIZE }) {
    this.threads = threads;
    this.batchSize = batchSize;
    this.progressMs = progressMs;
    this.onFound = onFound;
    this.onError = onError;
    this.gpu = gpu;
    this.gpuPlatform = gpuPlatform;
    this.gpuDevice = gpuDevice;
    this.gpuBatchSize = gpuBatchSize;
    this.workers = [];
    this.found = false;
    this.current = null;
    this.progressTimer = null;
    this.startedAt = 0;
  }

  start({ challenge, difficultyHex }) {
    this.stop();
    this.found = false;
    this.current = { challenge, difficultyHex };
    this.startedAt = Date.now();

    if (this.gpu) {
      /* Single GPU worker — the kernel dispatches millions of threads internally */
      const worker = new Worker(gpuWorkerPath, {
        workerData: {
          id: 0,
          challenge,
          difficulty: difficultyHex,
          noncePrefix: `0x${randomBytes(24).toString('hex')}`,
          batchSize: this.gpuBatchSize,
          gpuPlatform: this.gpuPlatform,
          gpuDevice: this.gpuDevice,
        },
      });
      const state = { id: 0, worker, hashes: 0, hashrate: 0, elapsedMs: 0, status: 'starting' };
      worker.on('message', (msg) => this.#onMessage(state, msg));
      worker.on('error', (err) => this.onError?.(err));
      worker.on('exit', (code) => { state.status = code === 0 ? 'stopped' : `exit ${code}`; });
      this.workers.push(state);
    } else {
      for (let i = 0; i < this.threads; i++) {
        const worker = new Worker(workerPath, {
          workerData: {
            id: i,
            challenge,
            difficulty: difficultyHex,
            noncePrefix: `0x${randomBytes(24).toString('hex')}`,
            batchSize: this.batchSize,
          },
        });
        const state = { id: i, worker, hashes: 0, hashrate: 0, elapsedMs: 0, status: 'starting' };
        worker.on('message', (msg) => this.#onMessage(state, msg));
        worker.on('error', (err) => this.onError?.(err));
        worker.on('exit', (code) => {
          state.status = code === 0 ? 'stopped' : `exit ${code}`;
        });
        this.workers.push(state);
      }
    }
    this.progressTimer = setInterval(() => this.printProgress(), this.progressMs);
  }

  retarget({ challenge, difficultyHex }) {
    if (this.found) return;
    this.current = { challenge, difficultyHex };
    this.found = false;
    for (const s of this.workers) {
      s.hashes = 0;
      s.worker.postMessage({ type: 'retarget', challenge, difficulty: difficultyHex });
    }
  }

  stop() {
    if (this.progressTimer) clearInterval(this.progressTimer);
    this.progressTimer = null;
    for (const s of this.workers) {
      try { s.worker.postMessage({ type: 'stop' }); } catch { }
      try { s.worker.terminate(); } catch { }
    }
    this.workers = [];
  }

  totals() {
    const hashes = this.workers.reduce((sum, s) => sum + s.hashes, 0);
    const hashrate = this.workers.reduce((sum, s) => sum + s.hashrate, 0);
    return { hashes, hashrate };
  }

  printProgress() {
    const { hashes, hashrate } = this.totals();
    const expected = this.current ? targetToExpectedHashes(this.current.difficultyHex) : Infinity;
    const eta = hashrate > 0 ? expected / hashrate : Infinity;
    const progress = Number.isFinite(expected) && expected > 0
      ? ` | ${hashes.toLocaleString()} / ~${formatCount(expected)} (${Math.min(100, hashes / expected * 100).toFixed(1)}%)`
      : ` | hashes=${hashes.toLocaleString()}`;
    process.stdout.write(`\r[${new Date().toLocaleTimeString()}] ${formatHashrate(hashrate)}${progress} | eta≈${formatEta(eta)}     `);
  }

  #onMessage(state, msg) {
    if (msg.type === 'ready') {
      state.status = 'ready';
      if (state.id === 0) console.log(`miner wasm loaded: ${msg.version}`);
    } else if (msg.type === 'calibrated') {
      state.status = 'ready';
      console.log(`worker ${msg.id}: auto batch=${msg.batchSize.toLocaleString()} (~${formatHashrate(msg.hashrate)} single-core)`);
    } else if (msg.type === 'progress') {
      state.status = 'running';
      state.hashes = msg.hashes;
      state.hashrate = msg.hashrate;
      state.elapsedMs = msg.elapsedMs;
    } else if (msg.type === 'found') {
      if (this.found) return;
      this.found = true;
      state.status = 'found';
      this.printProgress();
      console.log(`\nfound nonce by worker ${state.id}: ${msg.nonce}`);
      for (const s of this.workers) {
        if (s !== state) s.worker.postMessage({ type: 'stop' });
      }
      this.onFound?.(msg);
    } else if (msg.type === 'stopped') {
      state.status = 'stopped';
      state.hashes = msg.hashes ?? state.hashes;
      state.elapsedMs = msg.elapsedMs ?? state.elapsedMs;
    } else if (msg.type === 'error') {
      state.status = 'error';
      this.onError?.(new Error(`worker ${state.id}: ${msg.message}`));
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const provider = new JsonRpcProvider(args.rpc, 1, { staticNetwork: true });
  const wallet = args.privateKey ? new Wallet(args.privateKey, provider) : null;
  const address = wallet?.address || args.address;
  const contract = new Contract(CONTRACT_ADDRESS, ABI, wallet || provider);

  if (address && !isAddress(address)) throw new Error(`invalid address: ${address}`);

  const state = await getState(contract);
  let balance = null;
  if (address) balance = await contract.balanceOf(address);

  if (args.status) {
    printState(state, address, balance);
    return;
  }

  if (!state.genesisComplete) throw new Error('genesis is not complete; mining is not open yet');
  if (!address) throw new Error('private key required. Set HASH256_PRIVATE_KEY=0x... or use --benchmark --address 0x...');
  if (!wallet && !args.noSubmit) throw new Error('submitting requires a private key; use --benchmark/--no-submit with --address for local-only mining');

  printState(state, address, balance);
  console.log(`rpc:      ${args.rpc}`);
  if (args.gpu) {
    console.log(`mode:     GPU (OpenCL) platform=${args.gpuPlatform} device=${args.gpuDevice} gpuBatch=${args.gpuBatchSize.toLocaleString()}`);
  } else {
    const batchDesc = args.batchSize === 0 ? 'auto' : args.batchSize.toLocaleString();
    console.log(`threads:  ${args.threads} (auto-detected: ${os.cpus().length} logical CPUs)`);
    console.log(`batch:    ${batchDesc} hashes/tick per worker (auto-calibrates on start)`);
  }
  console.log(`mode:     ${args.noSubmit ? 'local search only; will NOT submit tx' : 'submit mine(nonce) tx when found'}`);

  let currentEpoch = state.epoch;
  let currentDifficultyHex = state.difficultyHex;
  let currentChallenge = await contract.getChallenge(address);
  let restarting = false;
  let retargetTimer;
  let farm;

  const restartMining = async (reason) => {
    if (restarting) return;
    restarting = true;
    try {
      const s = await getState(contract);
      currentEpoch = s.epoch;
      currentDifficultyHex = s.difficultyHex;
      currentChallenge = await contract.getChallenge(address);
      console.log(`\n${reason}; challenge=${currentChallenge}, epoch=${currentEpoch.toString()}, target=${currentDifficultyHex}`);
      farm.start({ challenge: currentChallenge, difficultyHex: currentDifficultyHex });
    } finally {
      restarting = false;
    }
  };

  farm = new MinerFarm({
    threads: args.threads,
    batchSize: args.batchSize,
    progressMs: args.progressMs,
    gpu: args.gpu,
    gpuPlatform: args.gpuPlatform,
    gpuDevice: args.gpuDevice,
    gpuBatchSize: args.gpuBatchSize,
    onError: (err) => console.error(`\nminer error: ${err.stack || err.message}`),
    onFound: async (hit) => {
      try {
        if (args.noSubmit) {
          console.log(`result: ${hit.result}`);
          if (args.benchmark || args.once) {
            clearInterval(retargetTimer);
            farm.stop();
            return;
          }
          await restartMining('continuing after local hit');
          return;
        }

        const nonce = hexToBigInt(hit.nonce);
        let gasLimit = 300_000n;
        try {
          const estimated = await contract.mine.estimateGas(nonce);
          gasLimit = scaleGas(estimated, args.gasMultiplier, args.minGas, args.maxGas);
        } catch (err) {
          console.warn(`gas estimation failed, fallback gas=${gasLimit}: ${err.shortMessage || err.message}`);
        }
        const feeOverrides = await buildFeeOverrides(provider, args);
        console.log(`submitting mine(${hit.nonce}) gasLimit=${gasLimit.toString()} maxFee=${formatUnits(feeOverrides.maxFeePerGas, 'gwei')}gwei priority=${formatUnits(feeOverrides.maxPriorityFeePerGas, 'gwei')}gwei...`);
        const tx = await contract.mine(nonce, { gasLimit, ...feeOverrides });
        console.log(`tx sent: https://etherscan.io/tx/${tx.hash}`);
        const receipt = await tx.wait();
        console.log(`tx mined in block ${receipt.blockNumber}, status=${receipt.status}`);

        if (args.once) {
          clearInterval(retargetTimer);
          farm.stop();
          return;
        }
        await restartMining('tx confirmed; restarting');
      } catch (err) {
        console.error(`submit/restart failed: ${err.shortMessage || err.message}`);
        await restartMining('restarting after failed submission');
      }
    },
  });

  await restartMining('starting');

  retargetTimer = setInterval(async () => {
    try {
      const s = await getState(contract);
      const challenge = await contract.getChallenge(address);
      const difficultyHex = s.difficultyHex;
      if (s.epoch !== currentEpoch || difficultyHex !== currentDifficultyHex || challenge !== currentChallenge) {
        currentEpoch = s.epoch;
        currentDifficultyHex = difficultyHex;
        currentChallenge = challenge;
        console.log(`\nretarget: epoch=${currentEpoch.toString()}, blocksLeft=${s.epochBlocksLeft.toString()}, target=${difficultyHex}`);
        farm.retarget({ challenge, difficultyHex });
      }
    } catch (err) {
      console.warn(`\nretarget check failed: ${err.shortMessage || err.message}`);
    }
  }, args.retargetMs);

  process.on('SIGINT', () => {
    console.log('\nstopping...');
    clearInterval(retargetTimer);
    farm.stop();
    process.exit(0);
  });
}

main().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
