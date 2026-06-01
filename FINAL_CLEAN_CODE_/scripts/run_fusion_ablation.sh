#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -o logs/fusion_ablation_%j.out
#SBATCH -e logs/fusion_ablation_%j.err
#SBATCH --time=00:20:00

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
REPEATS="${REPEATS:-4}"
MIN_REPEAT="${MIN_REPEAT:-2}"
COOLDOWN_SEC="${COOLDOWN_SEC:-2}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-90}"
RUN_LOG="logs/fusion_ablation_${JOB}_raw.txt"

: > "${RUN_LOG}"

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

steps_for_hidden() {
    local hidden="$1"
    if [ "${hidden}" -ge 512 ]; then
        echo 50
    else
        echo 100
    fi
}

lr_for_hidden() {
    local hidden="$1"
    if [ "${hidden}" -ge 256 ]; then
        echo 1e-4
    else
        echo 3e-4
    fi
}

build_hidden() {
    local hidden="$1"
    local ff=$((4 * hidden))
    local head_dim=$((hidden / 4))
    local build_dir="build_fusion_h${hidden}_${JOB}"
    log "=== Build hidden=${hidden} ==="
    make all BUILD_DIR="${build_dir}" \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}" \
        2>&1 | tee -a "${RUN_LOG}"
}

run_case() {
    local hidden="$1"
    local backend="$2"
    local repeat="$3"
    local steps="$4"
    local lr="$5"
    local build_dir="build_fusion_h${hidden}_${JOB}"
    local tmp="logs/fusion_ablation_${JOB}_h${hidden}_${backend}_r${repeat}.tmp"
    local status=0

    log "=== hidden=${hidden} backend=${backend} repeat=${repeat} steps=${steps} lr=${lr} ==="
    if [ "${backend}" = "cublas_tc_lt" ]; then
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" \
            CME213_GEMM_BACKEND=cublas_tc CME213_LT_FUSION=1 \
            "./${build_dir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}" \
            > "${tmp}" 2>&1 || status=$?
    else
        cme213_clean_env_timeout "${CASE_TIMEOUT_SEC}" \
            CME213_GEMM_BACKEND=cublas_tc \
            "./${build_dir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}" \
            > "${tmp}" 2>&1 || status=$?
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
log "repeats=${REPEATS}"
log "summary_min_repeat=${MIN_REPEAT}"
log "cooldown_sec=${COOLDOWN_SEC}"

FAIL=0
for hidden in ${HIDDEN_LIST}; do
    build_hidden "${hidden}" || FAIL=1
    steps="$(steps_for_hidden "${hidden}")"
    lr="$(lr_for_hidden "${hidden}")"
    for repeat in $(seq 1 "${REPEATS}"); do
        run_case "${hidden}" "cublas_tc" "${repeat}" "${steps}" "${lr}" || FAIL=1
        sleep "${COOLDOWN_SEC}"
        run_case "${hidden}" "cublas_tc_lt" "${repeat}" "${steps}" "${lr}" || FAIL=1
        sleep "${COOLDOWN_SEC}"
    done
done

cme213_log_gpu_state "end" | tee -a "${RUN_LOG}"

log "=== Summarize ==="
python3 scripts/summarize_single_gpu_bench.py --log "${RUN_LOG}" \
    --tag "fusion_ablation_${JOB}" --min-repeat "${MIN_REPEAT}" \
    2>&1 | tee -a "${RUN_LOG}" || FAIL=1
python3 scripts/summarize_fusion_ablation.py \
    --summary "results/single_gpu_repeated_bench_summary_fusion_ablation_${JOB}.csv" \
    --tag "${JOB}" 2>&1 | tee -a "${RUN_LOG}" || FAIL=1

log "=== DONE job=${JOB} status=${FAIL} ==="
exit "${FAIL}"
