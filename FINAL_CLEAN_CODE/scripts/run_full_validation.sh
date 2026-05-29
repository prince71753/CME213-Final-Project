#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/full_validation_%j.out
#SBATCH -e logs/full_validation_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

run_and_check() {
    local name="$1"
    shift
    local out="logs/full_validation_${SLURM_JOB_ID}_${name}.txt"

    echo "=== ${name} ==="
    "$@" 2>&1 | tee "${out}"
    if grep -E "FAIL|(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)" "${out}" >/dev/null; then
        echo "Validation failed in ${name}; see ${out}" >&2
        exit 1
    fi
}

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvcc --version | tail -n 1
mpicxx --showme:version || true

echo "=== Clean Build ==="
make clean
make all tests mpi

run_and_check "test_gemm" ./build/test_gemm
run_and_check "test_attention" ./build/test_attention
run_and_check "test_layernorm" ./build/test_layernorm
run_and_check "test_model_reference" ./build/test_model_reference
run_and_check "test_fusion_benchmark" ./build/test_fusion_benchmark

run_and_check "single_gpu_train_20_steps" \
    ./build/train inp.txt --epochs 1 --max-steps 20

run_and_check "mpi_2_host_staged_20_steps" \
    mpirun -np 2 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 20

run_and_check "mpi_4_blocking_direct_20_steps" \
    env CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi \
        inp.txt --epochs 1 --max-steps 20

run_and_check "mpi_4_overlap_pinned_20_steps" \
    env CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi \
        inp.txt --epochs 1 --max-steps 20 --overlap --bucket-kb 256

echo "=== Benchmark Summary ==="
./build/benchmark_all

echo "=== FULL VALIDATION PASSED ==="
