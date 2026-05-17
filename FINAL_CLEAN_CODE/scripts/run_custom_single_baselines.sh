#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/custom_single_baseline_%j.out
#SBATCH -e logs/custom_single_baseline_%j.err
#SBATCH --time=00:20:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvcc --version | tail -n 1

echo "=== Build default H=128 ==="
make all

echo "=== Custom CUDA baseline: H=128 ==="
./build/train inp.txt --epochs 1 --max-steps 100

echo "=== Build H=256 ==="
make all BUILD_DIR=build_h256 \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64"

echo "=== Custom CUDA baseline: H=256 ==="
./build_h256/train inp.txt --epochs 1 --max-steps 50
