#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/final_repeated_sync_%j.out
#SBATCH -e logs/final_repeated_sync_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs
module load course/cme213/nvhpc/24.1

REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-50}"
BUCKET_KB="${BUCKET_KB:-256}"
AUTO_MAX_MB="${AUTO_MAX_MB:-8}"
HIDDEN_LIST="${HIDDEN_LIST:-128 256 512}"
LR_DEFAULT="${LR_DEFAULT:-3e-4}"
LR_H512="${LR_H512:-1e-4}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "repeats=${REPEATS}"
echo "steps=${STEPS}"
echo "bucket_kb=${BUCKET_KB}"
echo "auto_max_mb=${AUTO_MAX_MB}"
echo "hidden_list=${HIDDEN_LIST}"

run_case() {
    local hidden="$1"
    local mpi_build_dir="$2"
    local backend="$3"
    local mode="$4"
    local repeat="$5"
    local lr="$6"

    echo "=== hidden=${hidden}: backend=${backend} sync_mode=${mode} repeat=${repeat} ==="
    local cuda_aware=1
    if [ "${backend}" = "host_staged" ]; then
        cuda_aware=0
    fi

    if [ "${mode}" = "blocking" ]; then
        CME213_CUDA_AWARE_MPI="${cuda_aware}" mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
            --sync-mode blocking
    elif [ "${mode}" = "overlap" ]; then
        CME213_CUDA_AWARE_MPI="${cuda_aware}" mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
            --sync-mode overlap --bucket-kb "${BUCKET_KB}"
    else
        CME213_CUDA_AWARE_MPI="${cuda_aware}" mpirun -np 4 "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${lr}" \
            --sync-mode auto --auto-overlap-max-mb "${AUTO_MAX_MB}" \
            --bucket-kb "${BUCKET_KB}"
    fi
}

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

    for repeat in $(seq 1 "${REPEATS}"); do
        run_case "${hidden}" "${mpi_build_dir}" direct blocking "${repeat}" "${lr}"
        run_case "${hidden}" "${mpi_build_dir}" direct overlap "${repeat}" "${lr}"
        run_case "${hidden}" "${mpi_build_dir}" direct auto "${repeat}" "${lr}"
        if [ "${hidden}" -eq 512 ]; then
            run_case "${hidden}" "${mpi_build_dir}" host_staged blocking "${repeat}" "${lr}"
        fi
    done
done

echo "=== DONE ==="
