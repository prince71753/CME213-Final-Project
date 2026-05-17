#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/adaptive_sync_sweep_%j.out
#SBATCH -e logs/adaptive_sync_sweep_%j.err
#SBATCH --time=00:15:00

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
AUTO_MAX_MB="${AUTO_MAX_MB:-8}"
HIDDEN_LIST="${HIDDEN_LIST:-128 256 512}"
LR_DEFAULT="${LR_DEFAULT:-3e-4}"
LR_H512="${LR_H512:-1e-4}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "bucket_kb=${BUCKET_KB}"
echo "auto_max_mb=${AUTO_MAX_MB}"
echo "hidden_list=${HIDDEN_LIST}"

for hidden in ${HIDDEN_LIST}; do
    ff=$((4 * hidden))
    head_dim=$((hidden / 4))
    build_dir="build_h${hidden}"
    mpi_build_dir="build_mpi_h${hidden}"
    flags="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"
    lr="${LR_DEFAULT}"
    if [ "${hidden}" -eq 512 ]; then
        lr="${LR_H512}"
    fi

    echo "=== Build hidden=${hidden} ==="
    make all mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
        EXTRA_NVCC_FLAGS="${flags}"

    echo "=== hidden=${hidden}: sync_mode=blocking ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
        --sync-mode blocking

    echo "=== hidden=${hidden}: sync_mode=overlap bucket_kb=${BUCKET_KB} ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
        --sync-mode overlap --bucket-kb "${BUCKET_KB}"

    echo "=== hidden=${hidden}: sync_mode=auto auto_max_mb=${AUTO_MAX_MB} bucket_kb=${BUCKET_KB} ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
        --sync-mode auto --auto-overlap-max-mb "${AUTO_MAX_MB}" \
        --bucket-kb "${BUCKET_KB}"
done

echo "=== DONE ==="
