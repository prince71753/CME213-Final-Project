#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/nccl_allreduce_baseline_%j.out
#SBATCH -e logs/nccl_allreduce_baseline_%j.err
#SBATCH --time=00:10:00

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
MIN_BYTES="${MIN_BYTES:-4KB}"
MAX_BYTES="${MAX_BYTES:-32MB}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"
RANK_LIST="${RANK_LIST:-2 4}"
BACKENDS="${BACKENDS:-device nccl}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-120}"
BUILD_DIR="${BUILD_DIR:-build_nccl_alpha_beta_${JOB}}"
MPI_BUILD_DIR="${MPI_BUILD_DIR:-build_mpi_nccl_alpha_beta_${JOB}}"
RUN_LOG="logs/nccl_allreduce_baseline_${JOB}_raw.txt"

RANK_LIST="${RANK_LIST//,/ }"
BACKENDS="${BACKENDS//,/ }"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local ranks="$1"
    local backend="$2"
    local tmp="logs/nccl_allreduce_baseline_${JOB}_r${ranks}_${backend}.tmp"
    log "=== nccl_allreduce_baseline ranks=${ranks} backend=${backend} min_bytes=${MIN_BYTES} max_bytes=${MAX_BYTES} iters=${ITERS} warmup=${WARMUP} ==="
    local status=0
    cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" \
        NCCL_DEBUG=WARN \
        mpirun -np "${ranks}" "./${MPI_BUILD_DIR}/benchmark_allreduce_alpha_beta" \
        --backend "${backend}" --min-bytes "${MIN_BYTES}" --max-bytes "${MAX_BYTES}" \
        --iters "${ITERS}" --warmup "${WARMUP}" > "${tmp}" 2>&1 || status=$?
    cat "${tmp}" | tee -a "${RUN_LOG}"
    rm -f "${tmp}"
    return "${status}"
}

log "=== Environment ==="
hostname | tee -a "${RUN_LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${RUN_LOG}"
cme213_log_gpu_state "start" | tee -a "${RUN_LOG}"
log "min_bytes=${MIN_BYTES}"
log "max_bytes=${MAX_BYTES}"
log "iters=${ITERS}"
log "warmup=${WARMUP}"
log "rank_list=${RANK_LIST}"
log "backends=${BACKENDS}"

log "=== Build alpha_beta_bench with NCCL support ==="
make alpha_beta_bench BUILD_DIR="${BUILD_DIR}" MPI_BUILD_DIR="${MPI_BUILD_DIR}" \
    2>&1 | tee -a "${RUN_LOG}"

FAIL=0
for ranks in ${RANK_LIST}; do
    for backend in ${BACKENDS}; do
        if ! run_case "${ranks}" "${backend}"; then
            log "NCCL_ALLREDUCE_BASELINE_FAIL ranks=${ranks} backend=${backend}"
            FAIL=1
        fi
        sleep 2
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_nccl_allreduce_baseline.py --log "${RUN_LOG}" --tag "${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
