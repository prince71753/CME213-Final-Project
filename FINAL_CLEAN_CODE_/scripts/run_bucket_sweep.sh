#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/bucket_sweep_%j.out
#SBATCH -e logs/bucket_sweep_%j.err
#SBATCH --time=00:30:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

STEPS="${STEPS:-100}"

echo "=== Build ==="
make all mpi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"

echo "=== Blocking direct allreduce baseline ==="
CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps "${STEPS}"

for bucket_kb in 0 64 256 1024; do
    echo "=== Bucketed host-staged Iallreduce: bucket_kb=${bucket_kb} ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt \
        --epochs 1 --max-steps "${STEPS}" --overlap --bucket-kb "${bucket_kb}"
done

echo "=== DONE ==="
