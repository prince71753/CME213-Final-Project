#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/h512_sync_stability_%j.out
#SBATCH -e logs/h512_sync_stability_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

STEPS="${STEPS:-50}"
LR="${LR:-1e-4}"
BUCKET_KB="${BUCKET_KB:-256}"
HIDDEN=512
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
BUILD_DIR="build_h${HIDDEN}"
MPI_BUILD_DIR="build_mpi_h${HIDDEN}"
FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "lr=${LR}"
echo "bucket_kb=${BUCKET_KB}"

echo "=== Build hidden=${HIDDEN} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="${FLAGS}"

echo "=== hidden=512: backend=direct sync_mode=blocking repeat=1 ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
    inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
    --sync-mode blocking

echo "=== hidden=512: backend=direct sync_mode=blocking repeat=2 ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
    inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
    --sync-mode blocking

echo "=== hidden=512: backend=host_staged sync_mode=blocking ==="
CME213_CUDA_AWARE_MPI=0 mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
    inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
    --sync-mode blocking

echo "=== hidden=512: backend=pinned sync_mode=overlap bucket_kb=${BUCKET_KB} ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
    inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
    --sync-mode overlap --bucket-kb "${BUCKET_KB}"

echo "=== DONE ==="
