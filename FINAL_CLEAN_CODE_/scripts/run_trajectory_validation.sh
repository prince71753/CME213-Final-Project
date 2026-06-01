#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/trajectory_validation_%j.out
#SBATCH -e logs/trajectory_validation_%j.err
#SBATCH --time=00:12:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

mkdir -p logs results "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${JOB:-${SLURM_JOB_ID:-local}}"
HIDDEN="${HIDDEN:-128}"
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
BATCH_SIZE="${BATCH_SIZE:-32}"
RANKS="${RANKS:-4}"
STEPS="${STEPS:-12}"
LR="${LR:-3e-4}"
BUCKET_KB="${BUCKET_KB:-256}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-90}"
RUN_LOG="logs/trajectory_validation_${JOB}_raw.txt"
BUILD_DIR="build_trajectory_h${HIDDEN}_${JOB}"
MPI_BUILD_DIR="build_mpi_trajectory_h${HIDDEN}_${JOB}"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local name="$1"
    local mode="$2"
    local comm_thread="$3"
    local bucket="$4"
    local tmp="logs/trajectory_validation_${JOB}_${name}.tmp"
    local status=0
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    if [ "${comm_thread}" = "yes" ]; then
        env_args+=(CME213_COMM_THREAD=1)
    fi

    log "=== trajectory_case name=${name} hidden=${HIDDEN} ranks=${RANKS} batch=${BATCH_SIZE} mode=${mode} comm_thread=${comm_thread} bucket_kb=${bucket} steps=${STEPS} lr=${LR} ==="
    if [ "${mode}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking --trace-trajectory > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb "${bucket}" --trace-trajectory \
            > "${tmp}" 2>&1 || status=$?
    fi

    cat "${tmp}" | tee -a "${RUN_LOG}"
    rm -f "${tmp}"
    return "${status}"
}

log "=== Environment ==="
hostname | tee -a "${RUN_LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${RUN_LOG}"
cme213_log_gpu_state "start" | tee -a "${RUN_LOG}"
log "hidden=${HIDDEN}"
log "ranks=${RANKS}"
log "batch_size=${BATCH_SIZE}"
log "steps=${STEPS}"
log "lr=${LR}"
log "bucket_kb=${BUCKET_KB}"

FAIL=0
log "=== Build hidden=${HIDDEN} batch=${BATCH_SIZE} ==="
make mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM} -DBATCH_SIZE=${BATCH_SIZE}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

run_case "blocking_direct" "blocking" "no" 0 || FAIL=1
run_case "pinned_overlap" "overlap" "no" "${BUCKET_KB}" || FAIL=1
run_case "openmp_thread" "overlap" "yes" "${BUCKET_KB}" || FAIL=1

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_trajectory_validation.py --log "${RUN_LOG}" --tag "${JOB}" \
    --steps "${STEPS}" 2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
