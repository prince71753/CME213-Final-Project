#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/profile_overlap_h256_%j.out
#SBATCH -e logs/profile_overlap_h256_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs profiles
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

STEPS="${STEPS:-20}"
BUCKET_KB="${BUCKET_KB:-256}"
LR="${LR:-3e-4}"
HIDDEN=256
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
BUILD_DIR="build_h${HIDDEN}"
MPI_BUILD_DIR="build_mpi_h${HIDDEN}"
FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "bucket_kb=${BUCKET_KB}"
echo "lr=${LR}"
echo "tmpdir=${TMPDIR}"

echo "=== Build hidden=${HIDDEN} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    USE_NVTX=1 EXTRA_NVCC_FLAGS="${FLAGS}"

echo "=== Nsight Systems: hidden=${HIDDEN} blocking direct ==="
cme213_clean_env CME213_CUDA_AWARE_MPI=1 mpirun -np 4 nsys profile \
    --trace=cuda,nvtx,mpi,osrt,cublas \
    --sample=none \
    --stats=true \
    --force-overwrite=true \
    -o "profiles/h256_blocking_${SLURM_JOB_ID}_rank%q{OMPI_COMM_WORLD_RANK}" \
    "./${MPI_BUILD_DIR}/train_mpi" inp.txt --epochs 1 \
    --max-steps "${STEPS}" --lr "${LR}"

echo "=== Nsight Systems: hidden=${HIDDEN} pinned overlap ==="
cme213_clean_env CME213_CUDA_AWARE_MPI=1 mpirun -np 4 nsys profile \
    --trace=cuda,nvtx,mpi,osrt,cublas \
    --sample=none \
    --stats=true \
    --force-overwrite=true \
    -o "profiles/h256_overlap_${SLURM_JOB_ID}_rank%q{OMPI_COMM_WORLD_RANK}" \
    "./${MPI_BUILD_DIR}/train_mpi" inp.txt --epochs 1 \
    --max-steps "${STEPS}" --lr "${LR}" --overlap \
    --bucket-kb "${BUCKET_KB}"

echo "=== DONE ==="
