#!/bin/bash

# Shared helpers for benchmark/profile scripts.  Keep this file side-effect
# light: scripts source it after `set -euo pipefail`.

CME213_ENV_VARS=(
    CME213_GEMM_BACKEND
    CME213_LT_FUSION
    CME213_USE_CUDA_GRAPH
    CME213_USE_NCCL
    CME213_ALLOW_UNVALIDATED_NCCL
    CME213_COMM_THREAD
    CME213_DEFER_GRAD_AVG_TO_ADAM
    CME213_ASYNC_DEVICE_MPI
    CME213_CUDA_AWARE_MPI
    CME213_MIXED_PRECISION
    CME213_FP16_STORAGE_ONLY
    CME213_FFN_FP16
)

cme213_clean_env() {
    local cmd=(env)
    local var
    for var in "${CME213_ENV_VARS[@]}"; do
        cmd+=(-u "${var}")
    done
    "${cmd[@]}" "$@"
}

cme213_clean_env_timeout() {
    local timeout_sec="$1"
    shift
    local cmd=(timeout --kill-after=10s "${timeout_sec}" env)
    local var
    for var in "${CME213_ENV_VARS[@]}"; do
        cmd+=(-u "${var}")
    done
    "${cmd[@]}" "$@"
}

cme213_log_gpu_state() {
    local label="${1:-gpu_state}"
    echo "=== GPU state: ${label} ==="
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used \
        --format=csv || true
}
