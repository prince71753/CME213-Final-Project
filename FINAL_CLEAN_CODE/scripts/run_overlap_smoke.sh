#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/overlap_smoke_%j.out
#SBATCH -e logs/overlap_smoke_%j.err
#SBATCH --time=00:20:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

STEPS="${STEPS:-20}"

echo "=== Build ==="
make all mpi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"

echo "=== 4 GPUs: blocking direct allreduce ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps "${STEPS}"

echo "=== 4 GPUs: bucketed nonblocking host-staged allreduce ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps "${STEPS}" --overlap

echo "=== DONE ==="
