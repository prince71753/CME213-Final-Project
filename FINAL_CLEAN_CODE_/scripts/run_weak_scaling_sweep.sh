#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/weak_scaling_%j.out
#SBATCH -e logs/weak_scaling_%j.err
#SBATCH --time=00:15:00

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
LOCAL_BATCH="${LOCAL_BATCH:-32}"
STEPS="${STEPS:-50}"
REPEATS="${REPEATS:-3}"
MIN_REPEAT="${MIN_REPEAT:-2}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
BUCKET_KB="${BUCKET_KB:-1024}"
RANK_LIST="${RANK_LIST:-1 2 4}"
MODES="${MODES:-openmp_thread}"
LR="${LR:-1e-4}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-75}"
BUILD_DIR="${BUILD_DIR:-build_weak_h${HIDDEN}_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_mpi_weak_h${HIDDEN}_${JOB}}"
RUN_LOG="logs/weak_scaling_${JOB}_raw.txt"

RANK_LIST="${RANK_LIST//,/ }"
MODES="${MODES//,/ }"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local ranks="$1"
    local mode="$2"
    local repeat="$3"
    local total_batch=$((LOCAL_BATCH * ranks))
    local backend="${mode}"
    local tmp="logs/weak_scaling_${JOB}_r${ranks}_${mode}_rep${repeat}.tmp"
    log "=== weak hidden=${HIDDEN} ranks=${ranks} local_batch=${LOCAL_BATCH} total_batch=${total_batch} backend=${backend} sync_variant=${mode} bucket_kb=${BUCKET_KB} repeat=${repeat} steps=${STEPS} lr=${LR} ==="

    local env_args=(CME213_CUDA_AWARE_MPI=1)
    if [ "${GEMM_BACKEND}" != "auto" ] && [ "${GEMM_BACKEND}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${GEMM_BACKEND}")
    fi

    local status=0
    if [ "${mode}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${ranks}" "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    elif [ "${mode}" = "pinned" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${ranks}" "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb "${BUCKET_KB}" > "${tmp}" 2>&1 || status=$?
    elif [ "${mode}" = "openmp_thread" ]; then
        env_args+=(CME213_COMM_THREAD=1)
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${ranks}" "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb "${BUCKET_KB}" > "${tmp}" 2>&1 || status=$?
    else
        log "WEAK_SCALING_FAIL unknown_mode=${mode}"
        return 1
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
log "local_batch=${LOCAL_BATCH}"
log "steps=${STEPS}"
log "repeats=${REPEATS}"
log "summary_min_repeat=${MIN_REPEAT}"
log "rank_list=${RANK_LIST}"
log "modes=${MODES}"
log "bucket_kb=${BUCKET_KB}"
log "lr=${LR}"
log "gemm_backend=${GEMM_BACKEND}"
log "case_timeout_sec=${CASE_TIMEOUT_SEC}"

log "=== Build hidden=${HIDDEN} local_batch=${LOCAL_BATCH} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM} -DBATCH_SIZE=${LOCAL_BATCH}" \
    2>&1 | tee -a "${RUN_LOG}"

FAIL=0
for repeat in $(seq 1 "${REPEATS}"); do
    for ranks in ${RANK_LIST}; do
        for mode in ${MODES}; do
            if ! run_case "${ranks}" "${mode}" "${repeat}"; then
                log "WEAK_SCALING_FAIL hidden=${HIDDEN} ranks=${ranks} mode=${mode} repeat=${repeat}"
                FAIL=1
            fi
            sleep "${COOLDOWN_SEC}"
        done
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_weak_scaling.py --log "${RUN_LOG}" --tag "${JOB}" \
    --min-repeat "${MIN_REPEAT}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
