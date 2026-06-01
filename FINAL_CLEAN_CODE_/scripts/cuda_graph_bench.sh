#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/cuda_graph_bench_%j.out
#SBATCH -e logs/cuda_graph_bench_%j.err
#SBATCH --time=00:35:00

set -uo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1

JOB="${SLURM_JOB_ID:-local}"

# Reuse build dirs if present.
rm -rf build build_mpi build_h128 build_h256 build_h512
make all BUILD_DIR=build_h128 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_h256 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
make all BUILD_DIR=build_h512 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

csv="results/cuda_graph_bench_${JOB}.csv"
echo "hidden,backend,graph,steps,ms,tok_per_s,loss" > "$csv"

run_one() {
    local label="$1"; local hidden="$2"; local backend="$3"; local graph="$4"
    local bdir="$5"; local steps="$6"; local lr="$7"
    local out="logs/cuda_graph_bench_${JOB}_${label}.txt"
    local env_var=""
    [ "$graph" = "graph" ] && env_var="CME213_USE_CUDA_GRAPH=1"
    echo "=== ${label} ==="
    env CME213_GEMM_BACKEND="$backend" $env_var "./${bdir}/train" inp.txt \
        --epochs 1 --max-steps "$steps" --lr "$lr" > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$out" >/dev/null; then
        echo "FAILED ${label} (rc=$rc)"
        tail -n 20 "$out"
        echo "${hidden},${backend},${graph},${steps},,,FAIL" >> "$csv"
        return 1
    fi
    ms=$(grep -oE '[0-9]+ms' "$out" | tail -n 1 | tr -d 'ms')
    tok=$(grep -oE '[0-9]+ tok/s' "$out" | tail -n 1 | grep -oE '[0-9]+')
    loss=$(grep -oE 'avg_logged_loss=[0-9.]+' "$out" | tail -n 1 | sed 's/avg_logged_loss=//')
    echo "${hidden},${backend},${graph},${steps},${ms},${tok},${loss}" >> "$csv"
    echo "PASSED ${label}: ${ms}ms ${tok} tok/s loss=${loss}"
}

# 300 steps for H128/H256, 150 for H512 to fit time budget.
for graph in nograph graph; do
    run_one "h128_custom_${graph}"    128 custom    "$graph" build_h128 300 3e-4
    run_one "h256_cublas_tc_${graph}" 256 cublas_tc "$graph" build_h256 300 3e-4
    run_one "h512_cublas_tc_${graph}" 512 cublas_tc "$graph" build_h512 150 1e-4
done

echo "=== Summary CSV ==="
sort -t, -k1,1n -k2,2 -k3,3 "$csv" -o "$csv"
cat "$csv"

echo "=== DONE ${JOB} ==="
