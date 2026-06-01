#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/profile_comm_thread_pair_%j.out
#SBATCH -e logs/profile_comm_thread_pair_%j.err
#SBATCH --time=00:30:00

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
HIDDEN="${HIDDEN:-256}"
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
STEPS="${STEPS:-30}"
BUCKET_KB="${BUCKET_KB:-1024}"
LR="${LR:-}"
BUILD_DIR="${BUILD_DIR:-build_profile_comm_h${HIDDEN}_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_profile_comm_mpi_h${HIDDEN}_${JOB}}"

if [ -z "${LR}" ]; then
    if [ "${HIDDEN}" -ge 256 ]; then
        LR=1e-4
    else
        LR=3e-4
    fi
fi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cme213_log_gpu_state "start"
echo "hidden=${HIDDEN}"
echo "steps=${STEPS}"
echo "bucket_kb=${BUCKET_KB}"
echo "lr=${LR}"
echo "tmpdir=${TMPDIR}"

echo "=== Build hidden=${HIDDEN} profile binary ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    USE_NVTX=1 \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

run_profile() {
    local backend_label="$1"
    local mode="$2"
    local bucket="$3"
    shift 3
    local extra_env=("$@")
    local tag="h${HIDDEN}_comm_${backend_label}_${JOB}"
    local profile="profiles/${tag}"
    local common_args=(inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}")
    local sync_args=()
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    env_args+=("${extra_env[@]}")
    if [ "${mode}" = "blocking" ]; then
        sync_args=(--sync-mode blocking)
    else
        sync_args=(--sync-mode overlap --bucket-kb "${bucket}")
    fi

    echo "=== Nsight Systems ${tag} mode=${mode} bucket=${bucket} ==="
    cme213_clean_env "${env_args[@]}" mpirun -np 4 nsys profile \
        --trace=cuda,nvtx,mpi,osrt,cublas \
        --sample=none --stats=true --force-overwrite=true \
        -o "${profile}_rank%q{OMPI_COMM_WORLD_RANK}" \
        "./${MPI_BUILD_DIR}/train_mpi" "${common_args[@]}" "${sync_args[@]}"

    echo "=== Extract NSys summary ${tag} rank0 ==="
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
        --format csv --output - "${profile}_rank0.sqlite" > "results/${tag}_rank0_nsys_stats.csv"
    python3 scripts/extract_nsys_profile_summary.py \
        --sqlite "${profile}_rank0.sqlite" \
        --label "${tag}_rank0" \
        --out "results/${tag}_rank0_nsys_summary.csv"
}

run_profile direct blocking 0
run_profile pinned overlap "${BUCKET_KB}"
run_profile openmp_thread overlap "${BUCKET_KB}" CME213_COMM_THREAD=1

cme213_log_gpu_state "end"
echo "=== DONE profile comm-thread pair job=${JOB} ==="
