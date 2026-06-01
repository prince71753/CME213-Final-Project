#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/post_ln_validate_%j.out
#SBATCH -e logs/post_ln_validate_%j.err
#SBATCH --time=00:30:00

set -uo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1

JOB="${SLURM_JOB_ID:-local}"

echo "=== Clean build ==="
rm -rf build build_h128 build_h256 build_h512

echo "=== Build tests ==="
make tests -j4

echo "=== test_gemm ==="
./build/test_gemm | tail -n 40
echo "=== test_attention ==="
./build/test_attention | tail -n 40
echo "=== test_layernorm ==="
./build/test_layernorm | tail -n 40
echo "=== test_model_reference ==="
./build/test_model_reference | tail -n 40

echo "=== Throughput benchmarks ==="
make all BUILD_DIR=build_h128 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_h256 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
make all BUILD_DIR=build_h512 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

run_one() {
    local label="$1"; local bdir="$2"; local backend="$3"; local graph="$4"
    local steps="$5"; local lr="$6"
    local env_var=""
    [ "$graph" = "graph" ] && env_var="CME213_USE_CUDA_GRAPH=1"
    local out="logs/post_ln_validate_${JOB}_${label}.txt"
    echo "=== $label ==="
    env CME213_GEMM_BACKEND="$backend" $env_var "./${bdir}/train" inp.txt \
        --epochs 1 --max-steps "$steps" --lr "$lr" > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$out" >/dev/null; then
        echo "FAILED $label rc=$rc"; tail -n 20 "$out"; return 1
    fi
    tail -n 5 "$out"
}

# Use lr=1e-4 to keep losses finite over 200 steps for fair throughput.
run_one h128_custom_nograph    build_h128 custom    nograph 200 1e-4
run_one h128_custom_graph      build_h128 custom    graph   200 1e-4
run_one h128_cublas_tc_nograph build_h128 cublas_tc nograph 200 1e-4
run_one h256_cublas_tc_nograph build_h256 cublas_tc nograph 200 1e-4
run_one h256_cublas_tc_graph   build_h256 cublas_tc graph   200 1e-4
run_one h512_cublas_tc_nograph build_h512 cublas_tc nograph 100 5e-5
run_one h512_cublas_tc_graph   build_h512 cublas_tc graph   100 5e-5

echo "=== DONE ${JOB} ==="
