#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/final_bench_%j.out
#SBATCH -e logs/final_bench_%j.err
#SBATCH --time=00:25:00

set -uo pipefail
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

rm -rf build_h128 build_h256 build_h512
make all BUILD_DIR=build_h128 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_h256 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
make all BUILD_DIR=build_h512 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

run_one() {
    local label="$1"; local bdir="$2"; local backend="$3"
    local steps="$4"; local lr="$5"
    local out="logs/final_bench_${JOB}_${label}.txt"
    echo "=== ${label} (backend=${backend}, ${steps} steps, lr=${lr}) ==="
    if [ "$backend" = "auto" ] || [ "$backend" = "default" ]; then
        cme213_clean_env "./${bdir}/train" inp.txt \
            --epochs 1 --max-steps "$steps" --lr "$lr" > "$out" 2>&1
    else
        cme213_clean_env CME213_GEMM_BACKEND="$backend" "./${bdir}/train" inp.txt \
            --epochs 1 --max-steps "$steps" --lr "$lr" > "$out" 2>&1
    fi
    if grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$out" >/dev/null; then
        echo "DIVERGED ${label}"; tail -n 5 "$out"; return 1
    fi
    tail -n 3 "$out"
}

# Canonical baseline settings use lr=3e-4 for H128 and lr=1e-4 for H256/H512.
run_one h128_auto      build_h128 auto      100 3e-4
run_one h128_custom    build_h128 custom    100 3e-4
run_one h128_cublas_tc build_h128 cublas_tc 100 3e-4
run_one h256_auto      build_h256 auto      100 1e-4
run_one h256_custom    build_h256 custom    100 1e-4
run_one h256_cublas_tc build_h256 cublas_tc 100 1e-4
run_one h512_auto      build_h512 auto      50  1e-4
run_one h512_custom    build_h512 custom    50  1e-4
run_one h512_cublas_tc build_h512 cublas_tc 50  1e-4

echo "=== DONE ${JOB} ==="
