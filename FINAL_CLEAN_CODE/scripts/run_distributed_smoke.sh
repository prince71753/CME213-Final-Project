#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/distributed_smoke_%j.out
#SBATCH -e logs/distributed_smoke_%j.err
#SBATCH --time=00:20:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

echo "=== Build ==="
make all mpi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
which mpicxx
mpirun --version | head -n 1

echo "=== Single GPU smoke ==="
./build/train inp.txt --epochs 1 --max-steps 5

echo "=== MPI smoke: 2 ranks, host-staged gradients ==="
mpirun -np 2 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 5

echo "=== MPI smoke: 4 ranks, host-staged gradients ==="
mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 5

echo "=== MPI smoke: 4 ranks, CUDA-aware direct gradients ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 5

echo "=== DONE ==="
