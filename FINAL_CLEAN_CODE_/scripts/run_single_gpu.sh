#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/train_single_%j.out
#SBATCH -e logs/train_single_%j.err
#SBATCH --time=00:30:00

mkdir -p logs
module load course/cme213/nvhpc/24.1

echo "=== Single-GPU Training ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

./build/train inp.txt
