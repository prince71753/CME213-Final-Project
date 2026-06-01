#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/overlap_speedup_by_hidden_%j.out
#SBATCH -e logs/overlap_speedup_by_hidden_%j.err
#SBATCH --time=00:30:00

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
HIDDEN_LIST="${HIDDEN_LIST:-128 256 512}"
HIDDEN_LIST="${HIDDEN_LIST//,/ }"
RANKS="${RANKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-50}"
REPEATS="${REPEATS:-5}"
MIN_REPEAT="${MIN_REPEAT:-2}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
LR="${LR:-1e-4}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-120}"
BUCKET_H128="${BUCKET_H128:-256}"
BUCKET_H256="${BUCKET_H256:-1024}"
BUCKET_H512="${BUCKET_H512:-2048}"
RUN_LOG="logs/overlap_speedup_by_hidden_${JOB}_raw.txt"

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
    local build_dir="build_overlap_h${hidden}_${JOB}"
    local mpi_build_dir="build_mpi_overlap_h${hidden}_${JOB}"

    log "=== Build hidden=${hidden} batch=${BATCH_SIZE} ==="
    make mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim} -DBATCH_SIZE=${BATCH_SIZE}" \
        2>&1 | tee -a "${RUN_LOG}"
}

run_case() {
    local hidden="$1"
    local backend_label="$2"
    local sync_mode="$3"
    local bucket="$4"
    local repeat="$5"
    local mpi_build_dir="build_mpi_overlap_h${hidden}_${JOB}"
    local tmp="logs/overlap_speedup_by_hidden_${JOB}_h${hidden}_${backend_label}_r${repeat}.tmp"
    local status=0
    local env_args=(CME213_CUDA_AWARE_MPI=1)

    if [ "${GEMM_BACKEND}" != "auto" ] && [ "${GEMM_BACKEND}" != "default" ]; then
        env_args+=(CME213_GEMM_BACKEND="${GEMM_BACKEND}")
    fi
    if [ "${backend_label}" = "openmp_thread" ]; then
        env_args+=(CME213_COMM_THREAD=1)
    fi

    log "=== overlap_speedup hidden=${hidden} ranks=${RANKS} batch=${BATCH_SIZE} backend=${backend_label} sync_mode=${sync_mode} bucket_kb=${bucket} repeat=${repeat} steps=${STEPS} lr=${LR} gemm_backend=${GEMM_BACKEND} ==="

    if [ "${sync_mode}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${mpi_build_dir}/train_mpi" \
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
log "hidden_list=${HIDDEN_LIST}"
log "ranks=${RANKS}"
log "batch_size=${BATCH_SIZE}"
log "steps=${STEPS}"
log "repeats=${REPEATS}"
log "summary_min_repeat=${MIN_REPEAT}"
log "lr=${LR}"
log "gemm_backend=${GEMM_BACKEND}"
log "bucket_h128=${BUCKET_H128}"
log "bucket_h256=${BUCKET_H256}"
log "bucket_h512=${BUCKET_H512}"

FAIL=0
for hidden in ${HIDDEN_LIST}; do
    build_hidden "${hidden}" || FAIL=1
    bucket="$(bucket_for_hidden "${hidden}")"
    for repeat in $(seq 1 "${REPEATS}"); do
        if ! run_case "${hidden}" "blocking" "blocking" 0 "${repeat}"; then
            log "OVERLAP_SPEEDUP_FAIL hidden=${hidden} backend=blocking repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
        if ! run_case "${hidden}" "openmp_thread" "overlap" "${bucket}" "${repeat}"; then
            log "OVERLAP_SPEEDUP_FAIL hidden=${hidden} backend=openmp_thread repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_overlap_speedup.py --log "${RUN_LOG}" --tag "${JOB}" \
    --min-repeat "${MIN_REPEAT}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
