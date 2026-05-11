# hash256-cli

CPU / GPU CLI miner for [hash256.fun](https://hash256.fun/mine).  
Supports **three mining backends**: CPU (WASM), OpenCL GPU, and **CUDA GPU** (NVIDIA, recommended for best performance).  
Submits successful nonces directly to the HASH Ethereum mainnet contract.

Contract: [`0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc`](https://etherscan.io/address/0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc)

---

## Understanding the Game — Why "Mining is not open"?

### Three-Phase Launch

HASH follows a strict three-phase lifecycle enforced entirely on-chain:

```
Phase 1 · GENESIS  →  Phase 2 · SEED  →  Phase 3 · MINING (open)
```

**Phase 1 · Genesis Mint** *(currently active / may be incomplete)*  
- 1,050,000 HASH sold at a fixed price of **0.01 ETH per 1,000 HASH** (~$0.03/HASH)  
- Max 5,000 HASH per transaction · no per-wallet cap · first come first served  
- All raised ETH is locked inside the contract until the pool is seeded  
- Closes automatically when 1,050,000 HASH is fully sold  
- ➜ https://hash256.fun/ (Genesis page)

**Phase 2 · Seed Pool**  
- After Genesis sells out, anyone can call `seedPool()` on the contract  
- The contract atomically mints another 1,050,000 HASH and seeds a **Uniswap V4 ETH/HASH pool** with 10.5 ETH + 1,050,000 HASH  
- The LP position is locked forever — no one can remove it  
- Pool opens at the same $0.03 launch price

**Phase 3 · Mining Opens**  
- Only after `seedPool()` is called does `genesisComplete` flip to `true`  
- The contract then accepts `mine(nonce)` calls  
- This CLI becomes usable at this point

---

### How the PoW Puzzle Works

```
challenge  = keccak256(chainId ‖ contract ‖ miner_address ‖ epoch)
valid nonce = keccak256(challenge ‖ nonce) < currentDifficulty
```

| Property | Detail |
|---|---|
| Address-bound | Your challenge is unique to your wallet — mempool sniping is impossible |
| Epoch rotation | Every ~100 blocks (~20 min) the challenge rotates — pre-computed solutions expire |
| Replay-proof | Each (miner, nonce, epoch) tuple can only mint once |
| Rate limit | Hard cap of 10 mints per block on-chain |

---

### Tokenomics & Halvings

Total supply: **21,000,000 HASH** (Bitcoin-style hard cap)

| Allocation | Amount |
|---|---|
| Genesis Mint | 1,050,000 HASH (5%) |
| LP Side (locked) | 1,050,000 HASH (5%) |
| Mining Rewards | 18,900,000 HASH (90%) |
| Team / VC / Airdrop | **0%** |

| Era | Reward / mint | Trigger |
|---|---|---|
| Era 1 | 100 HASH | starts at genesis complete |
| Era 2 | 50 HASH | after 100,000 mints |
| Era 3 | 25 HASH | after 200,000 mints |
| Era 4 | 12.5 HASH | after 300,000 mints |
| Era 5+ | 6.25 HASH | and so on… |

Difficulty retargets every **2,016 mints** (same formula as Bitcoin), targeting 1 mint/minute globally.  
Full distribution takes approximately **~290 days**.

---
## From 0 to Running

### Step 1 — Prerequisites

| Requirement | CPU mining | OpenCL GPU | CUDA GPU |
|---|---|---|---|
| [Node.js](https://nodejs.org/) ≥ 20 | ✅ | ✅ | ✅ |
| GCC | — | ✅ | — |
| OpenCL runtime (GPU driver) | — | ✅ | — |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (nvcc ≥ 11) | — | — | ✅ |
| NVIDIA driver ≥ 520 | — | — | ✅ |

---

### Step 2 — Clone & Install

```bash
git clone https://github.com/gjs222/hash256-cli.git
cd hash256-cli
npm install
```

`npm install` automatically downloads the WASM miner files into `vendor/miner/`.

---

### Step 3 — Configure

Copy the example env file and fill in your private key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required — use a fresh mining wallet with limited ETH for gas
HASH256_PRIVATE_KEY=0xyour_private_key_here

# Optional — defaults to Ethereum mainnet public RPC
HASH256_RPC_URL=https://ethereum.publicnode.com

# Optional — EIP-1559 fee tuning
HASH256_PRIORITY_FEE_GWEI=8
HASH256_MAX_FEE_GWEI=20
```

> ⚠️ Never commit `.env` to git. It is already in `.gitignore`.

#### Win the block-cap race with a private RPC

The HASH contract allows a hard maximum of **10 `mine()` calls per block**. When
many miners find a nonce in the same epoch, only the first 10 txs to land win —
the rest revert and still cost gas.

Validators (block builders) include **private-mempool txs before public ones**.
Raising your gwei tip does not buy you placement against builders — they see
private orderflow first regardless of fee. Routing your `mine()` through a
private RPC sends it directly to builders, ahead of the public mempool.

| Provider | RPC URL | Notes |
|---|---|---|
| **MEV-Blocker** | `https://rpc.mevblocker.io/fast` | Free; routes to multiple builders |
| **Flashbots Protect** | `https://rpc.flashbots.net/fast` | Free; Flashbots builder network |

Set your private RPC in `.env`:

```bash
# Private orderflow — reaches block builders before public mempool
HASH256_RPC_URL=https://rpc.mevblocker.io/fast
# or
HASH256_RPC_URL=https://rpc.flashbots.net/fast
```

Or pass it at runtime:

```bash
node src/cli.js --rpc https://rpc.mevblocker.io/fast
```

---


### Step 4 — Build GPU binary *(skip if CPU mining)*

**CUDA (NVIDIA — highest hashrate):**

```bash
# Auto-detects your GPU's SM architecture via nvidia-smi
bash build-gpu-cuda.sh

# Or specify manually:
bash build-gpu-cuda.sh sm_89   # RTX 40xx (Ada)
bash build-gpu-cuda.sh sm_86   # RTX 30xx (Ampere)
bash build-gpu-cuda.sh sm_75   # RTX 20xx (Turing)
bash build-gpu-cuda.sh sm_90   # H100 (Hopper)
```

Produces `src/gpu_miner_cuda` — automatically preferred over OpenCL when present.

**OpenCL (AMD / Intel / NVIDIA without CUDA Toolkit):**

```bash
gcc -O2 src/gpu_miner.c -ldl -o src/gpu_miner
```

---

### Step 5 — Run

**CUDA or OpenCL GPU** (auto-selects CUDA if `src/gpu_miner_cuda` exists):

```bash
npm run mine:gpu
```

**CPU** (no build step needed):

```bash
npm run mine
```

**Check chain status** (no private key required):

```bash
node src/cli.js --status
```

---

## Backend Comparison

| Backend | npm script | Typical Hashrate |
|---|---|---|
| CPU (WASM) | `npm run mine` | ~2–20 MH/s |
| OpenCL GPU | `npm run mine:gpu` | ~100–500 MH/s |
| CUDA GPU | `npm run mine:gpu` | ~500 MH/s – 2+ GH/s |

---

## All Options

```bash
node src/cli.js --status                       # chain status; no key needed
node src/cli.js --benchmark --address 0x...    # local benchmark, no tx submitted
node src/cli.js --threads 8                    # CPU: override thread count
node src/cli.js --gpu                          # enable GPU mining
node src/cli.js --gpu --gpu-device 1           # GPU: second device (0-indexed)
node src/cli.js --gpu --gpu-batch-size 8000000 # GPU: override batch size
node src/cli.js --rpc https://...              # override RPC endpoint
node src/cli.js --priority-fee-gwei 3          # faster tx inclusion
node src/cli.js --max-fee-gwei 10              # cap on max fee
node src/cli.js --once                         # stop after one successful tx
node src/cli.js --no-submit                    # find nonce but do not broadcast
```

---

## Notes

- Mining challenge is bound to your wallet address — mempool sniping is impossible.
- Challenge rotates every ~100 blocks (~20 min). Retargeting is handled automatically.
- Public RPCs can rate-limit. For sustained mining, a private RPC is recommended.
- Submitting a found nonce costs mainnet gas; stale submissions may also cost gas.
- If nonces often revert, increase `--priority-fee-gwei`. Reverts are usually due to stale epoch/difficulty, not gas exhaustion.
- On Linux with a display GPU, CUDA batch size is tuned to ~20 ms per launch to avoid X server stalls.

## Credits

Based on [haha256](https://github.com/vcing/haha256) by [@vcing](https://github.com/vcing).  
OpenCL GPU mining extension added in this fork.  
CUDA GPU backend (`src/gpu_miner_cuda.cu`) with keccak256 optimizations added subsequently.
