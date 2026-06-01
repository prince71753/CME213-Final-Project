#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/model_size_sweep_%j.out
#SBATCH -e logs/model_size_sweep_%j.err
#SBATCH --time=01:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

STEPS="${STEPS:-50}"
BUCKET_KB="${BUCKET_KB:-256}"
BUCKET_LIST="${BUCKET_LIST:-${BUCKET_KB}}"
BUCKET_LIST="${BUCKET_LIST//,/ }"
BUCKET_LIST="${BUCKET_LIST//:/ }"
LR="${LR:-3e-4}"
HIDDEN_LIST="${HIDDEN_LIST:-128 256 512}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "bucket_kb=${BUCKET_KB}"
echo "bucket_list=${BUCKET_LIST}"
echo "lr=${LR}"
echo "hidden_list=${HIDDEN_LIST}"

for hidden in ${HIDDEN_LIST}; do
    ff=$((4 * hidden))
    head_dim=$((hidden / 4))
    build_dir="build_h${hidden}"
    mpi_build_dir="build_mpi_h${hidden}"
    flags="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"

    echo "=== Build hidden=${hidden} ==="
    make all mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
        EXTRA_NVCC_FLAGS="${flags}"

    echo "=== hidden=${hidden}: blocking direct allreduce ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}"

    for bucket_kb in ${BUCKET_LIST}; do
        echo "=== hidden=${hidden}: pinned bucketed Iallreduce bucket_kb=${bucket_kb} ==="
        CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" --overlap \
            --bucket-kb "${bucket_kb}"
    done
done

echo "=== DONE ==="
