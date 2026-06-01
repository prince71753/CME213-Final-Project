#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/profile_h128_mpi_backend_pair_%j.out
#SBATCH -e logs/profile_h128_mpi_backend_pair_%j.err
#SBATCH --time=00:25:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs profiles results "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${SLURM_JOB_ID:-local}"
HIDDEN=128
FF=512
HEAD_DIM=32
STEPS="${STEPS:-100}"
LR="${LR:-3e-4}"
BUCKET_KB="${BUCKET_KB:-256}"
BUILD_DIR="${BUILD_DIR:-build_profile_h128_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_profile_h128_mpi_${JOB}}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "lr=${LR}"
echo "bucket_kb=${BUCKET_KB}"
echo "tmpdir=${TMPDIR}"

echo "=== Build H128 MPI profile binary ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    USE_NVTX=1 \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

run_profile() {
    local backend="$1"
    local tag="h128_mpi_${backend}_${JOB}"
    local profile="profiles/${tag}"

    echo "=== Nsight Systems ${tag} ==="
    cme213_clean_env CME213_CUDA_AWARE_MPI=1 CME213_GEMM_BACKEND="${backend}" \
        mpirun -np 4 nsys profile \
        --trace=cuda,nvtx,mpi,osrt,cublas \
        --sample=none --stats=true --force-overwrite=true \
        -o "${profile}_rank%q{OMPI_COMM_WORLD_RANK}" \
        "./${MPI_BUILD_DIR}/train_mpi" inp.txt --epochs 1 \
        --max-steps "${STEPS}" --lr "${LR}" \
        --sync-mode overlap --bucket-kb "${BUCKET_KB}"

    echo "=== Extract NSys stats ${tag} rank0 ==="
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
        --format csv --output - "${profile}_rank0.sqlite" > "results/${tag}_rank0_nsys_stats.csv"
    python3 scripts/extract_nsys_profile_summary.py \
        --sqlite "${profile}_rank0.sqlite" \
        --label "${tag}_rank0" \
        --out "results/${tag}_rank0_nsys_summary.csv"
}

run_profile custom
run_profile cublas_tc

echo "=== DONE profile H128 MPI backend pair job=${JOB} ==="
