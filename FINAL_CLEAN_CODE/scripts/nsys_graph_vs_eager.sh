#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/nsys_graph_vs_eager_%j.out
#SBATCH -e logs/nsys_graph_vs_eager_%j.err
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
source scripts/common_env.sh

JOB="${SLURM_JOB_ID:-local}"
rm -rf build_h256 build_h128
make all BUILD_DIR=build_h128 USE_NVTX=1 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_h256 USE_NVTX=1 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'

run_nsys() {
    local label="$1"; local bdir="$2"; local backend="$3"; local graph="$4"
    local steps="$5"; local lr="$6"
    local env_var=""
    [ "$graph" = "graph" ] && env_var="CME213_USE_CUDA_GRAPH=1"
    local rep="results/nsys_graph_${JOB}_${label}"
    echo "=== ${label} ==="
    cme213_clean_env CME213_GEMM_BACKEND="$backend" $env_var \
        nsys profile -t cuda,nvtx -o "$rep" --force-overwrite=true \
        "./${bdir}/train" inp.txt --epochs 1 --max-steps "$steps" --lr "$lr" \
        > "logs/nsys_graph_${JOB}_${label}.txt" 2>&1 || true
    # Even without cudaProfilerApi calls, nsys will record everything.
    nsys stats --report cuda_gpu_kern_sum -f csv "${rep}.nsys-rep" \
        > "results/nsys_graph_${JOB}_${label}_kern.csv" 2>/dev/null || true
    nsys stats --report cuda_api_sum -f csv "${rep}.nsys-rep" \
        > "results/nsys_graph_${JOB}_${label}_api.csv" 2>/dev/null || true
    echo "--- top kernels ---"
    head -n 25 "results/nsys_graph_${JOB}_${label}_kern.csv" || true
    echo "--- top CUDA API ---"
    head -n 25 "results/nsys_graph_${JOB}_${label}_api.csv" || true
}

# Run at 200 steps; lowered lr to avoid divergence for H256 cublas_tc.
run_nsys "h128_custom_nograph"    build_h128 custom    nograph 200 1e-4
run_nsys "h128_custom_graph"      build_h128 custom    graph   200 1e-4
run_nsys "h256_cublas_tc_nograph" build_h256 cublas_tc nograph 200 1e-4
run_nsys "h256_cublas_tc_graph"   build_h256 cublas_tc graph   200 1e-4

echo "=== DONE ${JOB} ==="
