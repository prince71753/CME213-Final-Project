#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/gradient_sync_bench_%j.out
#SBATCH -e logs/gradient_sync_bench_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Build ==="
make mpi sync_bench

echo "=== backend=direct mode=blocking ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/benchmark_gradient_sync

echo "=== backend=host_staged mode=blocking ==="
CME213_CUDA_AWARE_MPI=0 mpirun -np 4 ./build_mpi/benchmark_gradient_sync

echo "=== backend=pinned mode=overlap ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/benchmark_gradient_sync --overlap

echo "=== DONE ==="
