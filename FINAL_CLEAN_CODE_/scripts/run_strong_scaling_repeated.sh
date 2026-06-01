#!/bin/bash
# Repeated h256 fixed-total-batch strong-scaling sweep.
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/strong_scaling_repeated_%j.out
#SBATCH -e logs/strong_scaling_repeated_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

mkdir -p logs results plots "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${JOB:-${SLURM_JOB_ID:-local}}"
HIDDEN="${HIDDEN:-256}"
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
TOTAL_BATCH="${TOTAL_BATCH:-32}"
STEPS="${STEPS:-50}"
REPEATS="${REPEATS:-4}"
MIN_REPEAT="${MIN_REPEAT:-2}"
COOLDOWN_SEC="${COOLDOWN_SEC:-3}"
RANK_LIST="${RANK_LIST:-1 2 4}"
BUCKET_KB="${BUCKET_KB:-2048}"
LR="${LR:-1e-4}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-90}"
RUN_LOG="logs/strong_scaling_repeated_${JOB}_raw.txt"

RANK_LIST="${RANK_LIST//,/ }"
: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

build_ranks() {
    local ranks="$1"
    local local_batch=$((TOTAL_BATCH / ranks))
    local build_dir="build_strong_h${HIDDEN}_r${ranks}_${JOB}"
    local mpi_build_dir="build_mpi_strong_h${HIDDEN}_r${ranks}_${JOB}"
    log "=== Build hidden=${HIDDEN} ranks=${ranks} total_batch=${TOTAL_BATCH} local_batch=${local_batch} ==="
    make mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM} -DBATCH_SIZE=${local_batch}" \
        2>&1 | tee -a "${RUN_LOG}"
}

run_case() {
    local ranks="$1"
    local backend="$2"
    local repeat="$3"
    local local_batch=$((TOTAL_BATCH / ranks))
    local mpi_build_dir="build_mpi_strong_h${HIDDEN}_r${ranks}_${JOB}"
    local tmp="logs/strong_scaling_repeated_${JOB}_r${ranks}_${backend}_rep${repeat}.tmp"
    local status=0
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    if [ "${GEMM_BACKEND}" != "auto" ] && [ "${GEMM_BACKEND}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${GEMM_BACKEND}")
    fi
    if [ "${backend}" = "openmp_thread" ]; then
        env_args+=(CME213_COMM_THREAD=1)
    fi

    log "=== strong_repeated hidden=${HIDDEN} ranks=${ranks} total_batch=${TOTAL_BATCH} local_batch=${local_batch} backend=${backend} bucket_kb=${BUCKET_KB} repeat=${repeat} steps=${STEPS} lr=${LR} gemm_backend=${GEMM_BACKEND} ==="

    if [ "${backend}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${ranks}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    elif [ "${backend}" = "openmp_thread" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${ranks}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb "${BUCKET_KB}" > "${tmp}" 2>&1 || status=$?
    else
        log "STRONG_SCALING_FAIL unknown_backend=${backend}"
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
log "total_batch=${TOTAL_BATCH}"
log "steps=${STEPS}"
log "repeats=${REPEATS}"
log "summary_min_repeat=${MIN_REPEAT}"
log "rank_list=${RANK_LIST}"
log "bucket_kb=${BUCKET_KB}"
log "lr=${LR}"
log "gemm_backend=${GEMM_BACKEND}"
log "case_timeout_sec=${CASE_TIMEOUT_SEC}"

if [ $((TOTAL_BATCH % 4)) -ne 0 ]; then
    log "STRONG_SCALING_FAIL total_batch=${TOTAL_BATCH} must divide ranks 1,2,4"
    exit 1
fi

FAIL=0
for ranks in ${RANK_LIST}; do
    build_ranks "${ranks}" || FAIL=1
done

for repeat in $(seq 1 "${REPEATS}"); do
    for ranks in ${RANK_LIST}; do
        if ! run_case "${ranks}" "blocking" "${repeat}"; then
            log "STRONG_SCALING_FAIL ranks=${ranks} backend=blocking repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
        if [ "${ranks}" -gt 1 ]; then
            if ! run_case "${ranks}" "openmp_thread" "${repeat}"; then
                log "STRONG_SCALING_FAIL ranks=${ranks} backend=openmp_thread repeat=${repeat}"
                FAIL=1
            fi
            sleep "${COOLDOWN_SEC}"
        fi
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_strong_scaling_repeated.py --log "${RUN_LOG}" \
    --tag "${JOB}" --min-repeat "${MIN_REPEAT}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE strong scaling repeated job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
