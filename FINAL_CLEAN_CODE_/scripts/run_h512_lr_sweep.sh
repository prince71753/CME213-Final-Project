#!/bin/bash
# Sweep H512 learning rates to separate sync-path speed from training stability.
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/h512_lr_sweep_%j.out
#SBATCH -e logs/h512_lr_sweep_%j.err
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
REPEATS="${REPEATS:-5}"
LR_LIST="${LR_LIST:-1e-4 5e-5 2e-5}"
HIDDEN=512
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
BUILD_DIR="build_h${HIDDEN}_lr"
MPI_BUILD_DIR="build_mpi_h${HIDDEN}_lr"
FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "repeats=${REPEATS}"
echo "lr_list=${LR_LIST}"

echo "=== Build hidden=${HIDDEN} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="${FLAGS}"

for lr in ${LR_LIST}; do
    for repeat in $(seq 1 "${REPEATS}"); do
        echo "=== h512_lr lr=${lr} backend=direct sync_mode=blocking repeat=${repeat} ==="
        CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
            --sync-mode blocking
    done
done

echo "=== DONE ==="
