#!/usr/bin/env bash
# build-gpu-cuda.sh — Build the CUDA keccak256 miner (Linux)
# Usage:  bash build-gpu-cuda.sh [sm_75|sm_86|sm_89|sm_90]
#
# SM targets:
#   sm_75  RTX 20xx / Quadro RTX (Turing)
#   sm_86  RTX 30xx (Ampere)
#   sm_89  RTX 40xx (Ada Lovelace)
#   sm_90  H100 (Hopper)
set -euo pipefail
cd "$(dirname "$0")"

SRC="src/gpu_miner_cuda.cu"
OUT="src/gpu_miner_cuda"

# ── Check nvcc ────────────────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found. Install the CUDA Toolkit:"
  echo "  https://developer.nvidia.com/cuda-downloads"
  echo "  or:  sudo apt install nvidia-cuda-toolkit"
  exit 1
fi

NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+' || echo "?")
echo "nvcc version: $NVCC_VER"

# ── Auto-detect SM architecture from the first GPU ────────────────────────────
if [ -z "${1:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '.' | tr -d ' ')
    if [ -n "$CUDA_CAP" ]; then
      ARCH="sm_${CUDA_CAP}"
      echo "Auto-detected GPU compute capability: ${ARCH}"
    else
      ARCH="sm_86"
      echo "Could not detect GPU, defaulting to ${ARCH} (RTX 30xx)"
    fi
  else
    ARCH="sm_86"
    echo "nvidia-smi not found, defaulting to ${ARCH}"
  fi
else
  ARCH="$1"
fi

# ── Compile ───────────────────────────────────────────────────────────────────
echo "Compiling $SRC -> $OUT  (arch=$ARCH) ..."
nvcc -O3 -arch="${ARCH}" \
     --use_fast_math \
     -Xcompiler -O3 \
     -o "$OUT" "$SRC"

echo ""
echo "Done. Built: $OUT"
echo ""
echo "Usage:"
echo "  npm run mine:gpu"
echo "  node src/cli.js --gpu"
echo "  node src/cli.js --gpu --list-gpu-devices"
