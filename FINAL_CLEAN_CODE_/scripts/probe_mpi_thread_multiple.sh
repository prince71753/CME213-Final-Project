#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/mpi_thread_multiple_probe_%j.out
#SBATCH -e logs/mpi_thread_multiple_probe_%j.err
#SBATCH --time=00:05:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs build_probe "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1

JOB="${SLURM_JOB_ID:-local}"
BIN="build_probe/mpi_thread_multiple_probe"

echo "=== MPI_THREAD_MULTIPLE probe job=${JOB} ==="
hostname
mpicxx -std=c++17 -O2 tests/mpi_thread_multiple_probe.cpp -o "${BIN}"
mpirun -np 4 "./${BIN}"
echo "=== DONE MPI_THREAD_MULTIPLE probe job=${JOB} ==="
