#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/edge_case_validation_%j.out
#SBATCH -e logs/edge_case_validation_%j.err
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
STEPS="${STEPS:-15}"
LR="${LR:-3e-4}"
RANKS="${RANKS:-3}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-60}"
RUN_LOG="logs/edge_case_validation_${JOB}_raw.txt"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

build_batch() {
    local batch="$1"
    local mpi_build_dir="build_mpi_edge_h${HIDDEN}_b${batch}_${JOB}"
    log "=== build hidden=${HIDDEN} batch=${batch} ==="
    make mpi MPI_BUILD_DIR="${mpi_build_dir}" \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM} -DBATCH_SIZE=${batch}" \
        2>&1 | tee -a "${RUN_LOG}"
}

run_case() {
    local name="$1"
    local batch="$2"
    local mode="$3"
    local comm_thread="$4"
    local mpi_build_dir="build_mpi_edge_h${HIDDEN}_b${batch}_${JOB}"
    local tmp="logs/edge_case_validation_${JOB}_${name}.tmp"
    local status=0
    log "=== edge_case name=${name} hidden=${HIDDEN} ranks=${RANKS} batch=${batch} mode=${mode} comm_thread=${comm_thread} steps=${STEPS} lr=${LR} ==="

    local env_args=(CME213_CUDA_AWARE_MPI=1)
    if [ "${comm_thread}" = "yes" ]; then
        env_args+=(CME213_COMM_THREAD=1)
    fi

    if [ "${mode}" = "blocking" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" "${env_args[@]}" \
            mpirun -np "${RANKS}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode overlap --bucket-kb 256 > "${tmp}" 2>&1 || status=$?
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
log "steps=${STEPS}"
log "ranks=${RANKS}"
log "lr=${LR}"

FAIL=0
build_batch 32 || FAIL=1
build_batch 31 || FAIL=1

run_case "rank3_batch32_blocking" 32 blocking no || FAIL=1
run_case "rank3_batch32_pinned_overlap" 32 overlap no || FAIL=1
run_case "rank3_batch32_openmp_thread" 32 overlap yes || FAIL=1
run_case "rank3_batch31_blocking" 31 blocking no || FAIL=1
run_case "rank3_batch31_openmp_thread" 31 overlap yes || FAIL=1

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_edge_case_validation.py --log "${RUN_LOG}" --tag "${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
