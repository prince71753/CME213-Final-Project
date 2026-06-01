#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/nsys_hidden_breakdown_%j.out
#SBATCH -e logs/nsys_hidden_breakdown_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

mkdir -p logs profiles results plots "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${JOB:-${SLURM_JOB_ID:-local}}"
HIDDEN_LIST="${HIDDEN_LIST:-128 256 512}"
HIDDEN_LIST="${HIDDEN_LIST//,/ }"
RANKS="${RANKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-20}"
LR="${LR:-1e-4}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
BUCKET_H128="${BUCKET_H128:-256}"
BUCKET_H256="${BUCKET_H256:-2048}"
BUCKET_H512="${BUCKET_H512:-2048}"
RUN_LOG="logs/nsys_hidden_breakdown_${JOB}_raw.txt"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

bucket_for_hidden() {
    local hidden="$1"
    case "${hidden}" in
        128) echo "${BUCKET_H128}" ;;
        256) echo "${BUCKET_H256}" ;;
        512) echo "${BUCKET_H512}" ;;
        *) echo "${BUCKET_H256}" ;;
    esac
}

build_hidden() {
    local hidden="$1"
    local ff=$((4 * hidden))
    local head_dim=$((hidden / 4))
    local build_dir="build_nsys_breakdown_h${hidden}_${JOB}"
    local mpi_build_dir="build_mpi_nsys_breakdown_h${hidden}_${JOB}"

    log "=== Build hidden=${hidden} batch=${BATCH_SIZE} ==="
    make mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
        USE_NVTX=1 \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim} -DBATCH_SIZE=${BATCH_SIZE}" \
        2>&1 | tee -a "${RUN_LOG}"
}

run_profile() {
    local hidden="$1"
    local bucket="$2"
    local mpi_build_dir="build_mpi_nsys_breakdown_h${hidden}_${JOB}"
    local tag="h${hidden}_openmp_breakdown_${JOB}"
    local profile="profiles/${tag}"
    local env_args=(CME213_CUDA_AWARE_MPI=1 CME213_COMM_THREAD=1)

    if [ "${GEMM_BACKEND}" != "auto" ] && [ "${GEMM_BACKEND}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${GEMM_BACKEND}")
    fi

    log "=== Nsight Systems hidden=${hidden} ranks=${RANKS} batch=${BATCH_SIZE} bucket_kb=${bucket} steps=${STEPS} lr=${LR} gemm_backend=${GEMM_BACKEND} ==="
    cme213_clean_env "${env_args[@]}" mpirun -np "${RANKS}" nsys profile \
        --trace=cuda,nvtx,mpi,osrt,cublas \
        --sample=none --stats=true --force-overwrite=true \
        -o "${profile}_rank%q{OMPI_COMM_WORLD_RANK}" \
        "./${mpi_build_dir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
        --sync-mode overlap --bucket-kb "${bucket}" 2>&1 | tee -a "${RUN_LOG}"

    log "=== Extract rank0 Nsight summary hidden=${hidden} ==="
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
        --format csv --output - "${profile}_rank0.sqlite" \
        > "results/${tag}_rank0_nsys_stats.csv"
    python3 scripts/extract_nsys_profile_summary.py \
        --sqlite "${profile}_rank0.sqlite" \
        --label "${tag}_rank0" \
        --out "results/${tag}_rank0_nsys_summary.csv" \
        2>&1 | tee -a "${RUN_LOG}"
}

log "=== Environment ==="
hostname | tee -a "${RUN_LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${RUN_LOG}"
cme213_log_gpu_state "start" | tee -a "${RUN_LOG}"
log "hidden_list=${HIDDEN_LIST}"
log "ranks=${RANKS}"
log "batch_size=${BATCH_SIZE}"
log "steps=${STEPS}"
log "lr=${LR}"
log "gemm_backend=${GEMM_BACKEND}"
log "bucket_h128=${BUCKET_H128}"
log "bucket_h256=${BUCKET_H256}"
log "bucket_h512=${BUCKET_H512}"

FAIL=0
for hidden in ${HIDDEN_LIST}; do
    bucket="$(bucket_for_hidden "${hidden}")"
    build_hidden "${hidden}" || FAIL=1
    run_profile "${hidden}" "${bucket}" || FAIL=1
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize hidden-size Nsight breakdown ==="
python3 scripts/summarize_nsys_hidden_breakdown.py \
    --tag "${JOB}" \
    --hidden-list "${HIDDEN_LIST}" \
    --ranks "${RANKS}" \
    --batch "${BATCH_SIZE}" \
    --steps "${STEPS}" \
    --bucket-h128 "${BUCKET_H128}" \
    --bucket-h256 "${BUCKET_H256}" \
    --bucket-h512 "${BUCKET_H512}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE nsys hidden breakdown job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
