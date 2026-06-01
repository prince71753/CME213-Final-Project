#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -o logs/single_gpu_repeated_bench_%j.out
#SBATCH -e logs/single_gpu_repeated_bench_%j.err
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
BACKENDS="${BACKENDS:-custom cublas_tc}"
BACKENDS="${BACKENDS//,/ }"
REPEATS="${REPEATS:-5}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
STEPS="${STEPS:-}"
LR="${LR:-}"
BUILD_DIR="${BUILD_DIR:-build_bench_h${HIDDEN}_${JOB}}"
RUN_LOG="logs/single_gpu_repeated_bench_${JOB}_raw.txt"

if [ -z "${STEPS}" ]; then
    if [ "${HIDDEN}" -eq 512 ]; then
        STEPS=50
    else
        STEPS=100
    fi
fi
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

run_train() {
    local backend_label="$1"
    local repeat="$2"
    local tmp="logs/single_gpu_repeated_bench_${JOB}_${backend_label}_r${repeat}.tmp"
    log "=== hidden=${HIDDEN} backend=${backend_label} repeat=${repeat} steps=${STEPS} lr=${LR} ==="

    local status=0
    case "${backend_label}" in
        cublas_tc_lt)
            cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_LT_FUSION=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        cublas_tc_graph)
            cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_USE_CUDA_GRAPH=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        cublas_tc_fp16_storage)
            cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_FP16_STORAGE_ONLY=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        cublas_tc_ffn_fp16)
            cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_FFN_FP16=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        auto_fp16_storage|default_fp16_storage)
            cme213_clean_env CME213_FP16_STORAGE_ONLY=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        auto_ffn_fp16|default_ffn_fp16)
            cme213_clean_env CME213_FFN_FP16=1 \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        auto|default)
            cme213_clean_env \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
        *)
            cme213_clean_env CME213_GEMM_BACKEND="${backend_label}" \
                "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                > "${tmp}" 2>&1 || status=$?
            ;;
    esac
    cat "${tmp}" | tee -a "${RUN_LOG}"
    rm -f "${tmp}"
    return "${status}"
}

log "=== Environment ==="
hostname | tee -a "${RUN_LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${RUN_LOG}"
cme213_log_gpu_state "start" | tee -a "${RUN_LOG}"
log "hidden=${HIDDEN}"
log "backends=${BACKENDS}"
log "repeats=${REPEATS}"
log "cooldown_sec=${COOLDOWN_SEC}"
log "steps=${STEPS}"
log "lr=${LR}"
log "build_dir=${BUILD_DIR}"

log "=== Build hidden=${HIDDEN} ==="
make all BUILD_DIR="${BUILD_DIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}" \
    2>&1 | tee -a "${RUN_LOG}"

FAIL=0
for repeat in $(seq 1 "${REPEATS}"); do
    for backend in ${BACKENDS}; do
        if ! run_train "${backend}" "${repeat}"; then
            log "BENCH_FAIL hidden=${HIDDEN} backend=${backend} repeat=${repeat}"
            FAIL=1
        fi
        sleep "${COOLDOWN_SEC}"
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_single_gpu_bench.py --log "${RUN_LOG}" --tag "${JOB}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
