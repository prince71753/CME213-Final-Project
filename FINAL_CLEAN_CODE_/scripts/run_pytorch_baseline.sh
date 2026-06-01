#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/pytorch_baseline_%j.out
#SBATCH -e logs/pytorch_baseline_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "=== PyTorch baseline: default H=128 ==="
python3 benchmarks/pytorch_baseline.py \
    --data inp.txt \
    --device cuda \
    --epochs 1 \
    --max-steps 100 \
    --warmup-steps 5 \
    --seq-len 64 \
    --hidden-dim 128 \
    --num-heads 4 \
    --ff-dim 512 \
    --batch-size 32

echo "=== PyTorch baseline: H=256 ==="
python3 benchmarks/pytorch_baseline.py \
    --data inp.txt \
    --device cuda \
    --epochs 1 \
    --max-steps 50 \
    --warmup-steps 5 \
    --seq-len 64 \
    --hidden-dim 256 \
    --num-heads 4 \
    --ff-dim 1024 \
    --batch-size 32
