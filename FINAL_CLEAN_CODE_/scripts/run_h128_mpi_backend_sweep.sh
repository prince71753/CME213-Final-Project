#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/h128_mpi_backend_sweep_%j.out
#SBATCH -e logs/h128_mpi_backend_sweep_%j.err
#SBATCH --time=00:40:00

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
HIDDEN=128
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
REPEATS="${REPEATS:-10}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
STEPS="${STEPS:-50}"
LR="${LR:-3e-4}"
BACKENDS="${BACKENDS:-auto custom cublas_tc}"
OVERLAP_BUCKET_KB="${OVERLAP_BUCKET_KB:-256}"
BUILD_DIR="${BUILD_DIR:-build_h128_mpi_backend_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_h128_mpi_backend_mpi_${JOB}}"
RUN_LOG="logs/h128_mpi_backend_sweep_${JOB}_raw.txt"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local backend="$1"
    local mode="$2"
    local bucket="$3"
    local repeat="$4"
    local tmp="logs/h128_mpi_backend_sweep_${JOB}_${backend}_${mode}_${bucket}_r${repeat}.tmp"
    local status=0
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    if [ "${backend}" != "auto" ] && [ "${backend}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${backend}")
    fi

    log "=== hidden=${HIDDEN} backend=${backend} sync_mode=${mode} bucket_kb=${bucket} repeat=${repeat} steps=${STEPS} lr=${LR} ==="
    if [ "${mode}" = "blocking" ]; then
        cme213_clean_env "${env_args[@]}" \
            mpirun -np 4 "./${MPI_BUILD_DIR}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env "${env_args[@]}" \
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
log "lr=${LR}"
log "backends=${BACKENDS}"
log "overlap_bucket_kb=${OVERLAP_BUCKET_KB}"

log "=== Build hidden=${HIDDEN} ==="
make all mpi BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}" \
    2>&1 | tee -a "${RUN_LOG}"

FAIL=0
for backend in ${BACKENDS}; do
    for repeat in $(seq 1 "${REPEATS}"); do
        if ! run_case "${backend}" blocking 0 "${repeat}"; then
            log "H128_MPI_BACKEND_FAIL backend=${backend} mode=blocking repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
        if ! run_case "${backend}" overlap "${OVERLAP_BUCKET_KB}" "${repeat}"; then
            log "H128_MPI_BACKEND_FAIL backend=${backend} mode=overlap bucket=${OVERLAP_BUCKET_KB} repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_training_bucket_sweep.py --log "${RUN_LOG}" --tag "h128_mpi_backend_${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
