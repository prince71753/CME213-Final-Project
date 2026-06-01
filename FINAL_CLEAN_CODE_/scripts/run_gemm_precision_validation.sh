#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/gemm_precision_validation_%j.out
#SBATCH -e logs/gemm_precision_validation_%j.err
#SBATCH --time=00:12:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

mkdir -p logs results
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${JOB:-${SLURM_JOB_ID:-local}}"
RUN_LOG="logs/gemm_precision_validation_${JOB}_raw.txt"
: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

run_case() {
    local name="$1"
    local requested_backend="$2"
    local strict_fp32="$3"
    local description="$4"
    local tmp="logs/gemm_precision_validation_${JOB}_${name}.tmp"
    local status=0

    log "=== gemm_precision case=${name} requested_backend=${requested_backend} strict_fp32=${strict_fp32} description=${description} ==="

    if [ "${requested_backend}" = "auto" ]; then
        cme213_clean_env env CME213_STRICT_FP32="${strict_fp32}" \
            ./build/test_gemm > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env env CME213_GEMM_BACKEND="${requested_backend}" \
            CME213_STRICT_FP32="${strict_fp32}" \
            ./build/test_gemm > "${tmp}" 2>&1 || status=$?
    fi

    cat "${tmp}" | tee -a "${RUN_LOG}"
    rm -f "${tmp}"
    log "case_status name=${name} status=${status}"
    return "${status}"
}

log "=== Environment ==="
hostname | tee -a "${RUN_LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${RUN_LOG}"
nvcc --version | tail -n 1 | tee -a "${RUN_LOG}"

log "=== Build ==="
make build/test_gemm 2>&1 | tee -a "${RUN_LOG}"

FAIL=0
run_case "default_auto_cublas_tc" "auto" "0" \
    "default_throughput_path_expected_to_use_auto_cublas_tc" || FAIL=1
run_case "cublas_strict_fp32" "cublas" "1" \
    "library_fp32_reference_path_for_sanity" || FAIL=1
run_case "custom_strict_fp32" "custom" "1" \
    "custom_cuda_kernel_checked_against_cpu_at_1e-5_tolerance" || FAIL=1

log "=== Summarize ==="
python3 scripts/summarize_gemm_precision_validation.py --log "${RUN_LOG}" --tag "${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
