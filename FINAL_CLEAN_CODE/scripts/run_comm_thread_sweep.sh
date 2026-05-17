#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/comm_thread_sweep_%j.out
#SBATCH -e logs/comm_thread_sweep_%j.err
#SBATCH --time=00:20:00

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
HIDDEN="${HIDDEN:-256}"
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
REPEATS="${REPEATS:-5}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
STEPS="${STEPS:-50}"
BUCKETS_KB="${BUCKETS_KB:-1024}"
BUCKETS_KB="${BUCKETS_KB//,/ }"
LR="${LR:-}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
RUN_BLOCKING="${RUN_BLOCKING:-1}"
BUILD_DIR="${BUILD_DIR:-build_comm_thread_h${HIDDEN}_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_comm_thread_mpi_h${HIDDEN}_${JOB}}"
RUN_LOG="logs/comm_thread_sweep_${JOB}_raw.txt"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-60}"

if [ -z "${LR}" ]; then
    if [ "${HIDDEN}" -ge 256 ]; then
        LR=1e-4
    else
        LR=3e-4
    fi
fi

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local backend_label="$1"
    local mode="$2"
    local bucket="$3"
    local repeat="$4"
    shift 4
    local extra_env=("$@")
    local tmp="logs/comm_thread_sweep_${JOB}_${backend_label}_${mode}_${bucket}_r${repeat}.tmp"
    local status=0
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    if [ "${GEMM_BACKEND}" != "auto" ] && [ "${GEMM_BACKEND}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${GEMM_BACKEND}")
    fi
    env_args+=("${extra_env[@]}")

    log "=== hidden=${HIDDEN} backend=${backend_label} sync_mode=${mode} bucket_kb=${bucket} repeat=${repeat} steps=${STEPS} lr=${LR} ==="
    if [ "${mode}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb "${bucket}" > "${tmp}" 2>&1 || status=$?
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
log "repeats=${REPEATS}"
log "cooldown_sec=${COOLDOWN_SEC}"
log "steps=${STEPS}"
log "buckets_kb=${BUCKETS_KB}"
log "lr=${LR}"
log "gemm_backend=${GEMM_BACKEND}"
log "run_blocking=${RUN_BLOCKING}"
log "case_timeout_sec=${CASE_TIMEOUT_SEC}"

log "=== Build hidden=${HIDDEN} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}" \
    2>&1 | tee -a "${RUN_LOG}"

FAIL=0
for repeat in $(seq 1 "${REPEATS}"); do
    if [ "${RUN_BLOCKING}" != "0" ]; then
        if ! run_case direct blocking 0 "${repeat}"; then
            log "COMM_THREAD_SWEEP_FAIL backend=direct mode=blocking repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
    fi
    for bucket in ${BUCKETS_KB}; do
        if ! run_case pinned overlap "${bucket}" "${repeat}"; then
            log "COMM_THREAD_SWEEP_FAIL backend=pinned mode=overlap bucket=${bucket} repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
        if ! run_case openmp_thread overlap "${bucket}" "${repeat}" CME213_COMM_THREAD=1; then
            log "COMM_THREAD_SWEEP_FAIL backend=openmp_thread mode=overlap bucket=${bucket} repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_training_bucket_sweep.py --log "${RUN_LOG}" --tag "comm_thread_${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
