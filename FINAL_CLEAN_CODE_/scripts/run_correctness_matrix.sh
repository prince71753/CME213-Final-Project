#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/correctness_matrix_%j.out
#SBATCH -e logs/correctness_matrix_%j.err
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

JOB="${SLURM_JOB_ID:-local}"
SUMMARY="results/correctness_matrix_${JOB}.md"
FAIL=0
EXTRA_FEATURE_ENV=()
if [ "${CME213_COMM_THREAD:-0}" != "0" ]; then
    EXTRA_FEATURE_ENV+=(CME213_COMM_THREAD="${CME213_COMM_THREAD}")
fi

{
    echo "# Correctness matrix (${JOB})"
    echo
    echo "Host: $(hostname)"
    echo "Date: $(date -Is)"
    echo
} > "${SUMMARY}"

record() {
    local label="$1"
    local logfile="$2"
    local status="$3"
    if [ "${status}" -eq 0 ]; then
        echo "- ${label}: PASS" | tee -a "${SUMMARY}"
        tail -n 3 "${logfile}" | sed 's/^/  /' | tee -a "${SUMMARY}"
    else
        echo "- ${label}: FAIL (exit ${status})" | tee -a "${SUMMARY}"
        tail -n 12 "${logfile}" | sed 's/^/  /' | tee -a "${SUMMARY}"
        FAIL=1
    fi
    echo | tee -a "${SUMMARY}"
}

run_case() {
    local label="$1"
    shift
    local safe_label
    safe_label="$(echo "${label}" | tr ' /=' '____')"
    local logfile="logs/correctness_matrix_${JOB}_${safe_label}.txt"
    echo "=== ${label} ==="
    local status=0
    "$@" > "${logfile}" 2>&1 || status=$?
    record "${label}" "${logfile}" "${status}"
}

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cme213_log_gpu_state "start"

echo "=== Build default trainer, tests, and MPI ==="
build_log="logs/correctness_matrix_${JOB}_build.txt"
make -j8 all tests mpi > "${build_log}" 2>&1 || BUILD_STATUS=$?
BUILD_STATUS="${BUILD_STATUS:-0}"
record "make all tests mpi" "${build_log}" "${BUILD_STATUS}"

if [ "${BUILD_STATUS}" -eq 0 ]; then
    run_case "test_gemm" cme213_clean_env ./build/test_gemm
    run_case "test_attention" cme213_clean_env ./build/test_attention
    run_case "test_layernorm" cme213_clean_env ./build/test_layernorm
    run_case "test_model_reference custom" cme213_clean_env CME213_GEMM_BACKEND=custom ./build/test_model_reference
    run_case "test_model_reference cublas" cme213_clean_env CME213_GEMM_BACKEND=cublas ./build/test_model_reference
    run_case "test_model_reference cublas_tc" cme213_clean_env CME213_GEMM_BACKEND=cublas_tc ./build/test_model_reference
    run_case "test_model_reference cublas_tc_lt" cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_LT_FUSION=1 ./build/test_model_reference
    run_case "single_gpu_smoke default_auto" cme213_clean_env ./build/train inp.txt --epochs 1 --max-steps 5 --lr 3e-4
    run_case "single_gpu_smoke cublas_tc" cme213_clean_env CME213_GEMM_BACKEND=cublas_tc ./build/train inp.txt --epochs 1 --max-steps 5 --lr 3e-4
    run_case "train_validate_config" cme213_clean_env ./build/train --validate-config
    run_case "mpi_4_blocking_direct_smoke" cme213_clean_env "${EXTRA_FEATURE_ENV[@]}" CME213_CUDA_AWARE_MPI=1 CME213_GEMM_BACKEND=cublas_tc mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 5 --lr 3e-4 --sync-mode blocking
    run_case "mpi_4_overlap_pinned_smoke" cme213_clean_env "${EXTRA_FEATURE_ENV[@]}" CME213_CUDA_AWARE_MPI=1 CME213_GEMM_BACKEND=cublas_tc mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 5 --lr 3e-4 --sync-mode overlap --bucket-kb 256
fi

cme213_log_gpu_state "end"

{
    echo "## Result"
    if [ "${FAIL}" -eq 0 ]; then
        echo "PASS"
    else
        echo "FAIL"
    fi
} >> "${SUMMARY}"

echo "Summary written: ${SUMMARY}"
exit "${FAIL}"
