#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/blocking_scaling_%j.out
#SBATCH -e logs/blocking_scaling_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

STEPS="${STEPS:-100}"
EPOCHS="${EPOCHS:-1}"

echo "=== Build ==="
make all mpi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS} epochs=${EPOCHS}"

echo "=== 1 GPU: no MPI ==="
./build/train inp.txt --epochs "${EPOCHS}" --max-steps "${STEPS}"

echo "=== 2 GPUs: MPI blocking, host-staged gradients ==="
mpirun -np 2 ./build_mpi/train_mpi inp.txt --epochs "${EPOCHS}" --max-steps "${STEPS}"

echo "=== 4 GPUs: MPI blocking, host-staged gradients ==="
mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs "${EPOCHS}" --max-steps "${STEPS}"

echo "=== 2 GPUs: MPI blocking, CUDA-aware direct gradients ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 2 ./build_mpi/train_mpi inp.txt --epochs "${EPOCHS}" --max-steps "${STEPS}"

echo "=== 4 GPUs: MPI blocking, CUDA-aware direct gradients ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs "${EPOCHS}" --max-steps "${STEPS}"

echo "=== DONE ==="
