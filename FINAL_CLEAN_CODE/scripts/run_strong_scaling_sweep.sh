#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/strong_scaling_%j.out
#SBATCH -e logs/strong_scaling_%j.err
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
TOTAL_BATCH="${TOTAL_BATCH:-32}"
BUCKET_KB="${BUCKET_KB:-256}"
HIDDEN_LIST="${HIDDEN_LIST:-128 256}"
RANK_LIST="${RANK_LIST:-1 2 4}"
LR="${LR:-3e-4}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "steps=${STEPS}"
echo "total_batch=${TOTAL_BATCH}"
echo "bucket_kb=${BUCKET_KB}"
echo "hidden_list=${HIDDEN_LIST}"
echo "rank_list=${RANK_LIST}"

for hidden in ${HIDDEN_LIST}; do
    ff=$((4 * hidden))
    head_dim=$((hidden / 4))

    for ranks in ${RANK_LIST}; do
        local_batch=$((TOTAL_BATCH / ranks))
        build_dir="build_strong_h${hidden}_r${ranks}"
        mpi_build_dir="build_mpi_strong_h${hidden}_r${ranks}"
        flags="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim} -DBATCH_SIZE=${local_batch}"

        echo "=== Build hidden=${hidden} ranks=${ranks} local_batch=${local_batch} ==="
        make mpi BUILD_DIR="${build_dir}" MPI_BUILD_DIR="${mpi_build_dir}" \
            EXTRA_NVCC_FLAGS="${flags}"

        echo "=== strong hidden=${hidden} ranks=${ranks} local_batch=${local_batch} mode=blocking ==="
        CME213_CUDA_AWARE_MPI=1 mpirun -np "${ranks}" "./${mpi_build_dir}/train_mpi" \
            inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
            --sync-mode blocking

        if [ "${ranks}" -gt 1 ]; then
            echo "=== strong hidden=${hidden} ranks=${ranks} local_batch=${local_batch} mode=overlap ==="
            CME213_CUDA_AWARE_MPI=1 mpirun -np "${ranks}" "./${mpi_build_dir}/train_mpi" \
                inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}" \
                --sync-mode overlap --bucket-kb "${BUCKET_KB}"
        fi
    done
done

echo "=== DONE ==="
