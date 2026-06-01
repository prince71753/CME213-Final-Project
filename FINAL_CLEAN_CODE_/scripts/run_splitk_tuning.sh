#!/bin/bash
# Sweep the split-K block target for H256 weight-gradient GEMM shapes.
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/splitk_tuning_%j.out
#SBATCH -e logs/splitk_tuning_%j.err
#SBATCH --time=00:12:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results
module load course/cme213/nvhpc/24.1

TARGET_LIST="${TARGET_LIST:-144 216 360 432 576 720}"
CASES="${CASES:-splitk_dW1 splitk_dW2 splitk_qkv splitk_dWout}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "target_list=${TARGET_LIST}"
echo "cases=${CASES}"

for target in ${TARGET_LIST}; do
    build_dir="build_splitk_${target}_${SLURM_JOB_ID}"
    echo "=== Build target=${target} ==="
    make BUILD_DIR="${build_dir}" \
        EXTRA_NVCC_FLAGS="-DSPLITK_TARGET_BLOCKS=${target}" \
        "${build_dir}/profile_training_hotspots"

    for case_name in ${CASES}; do
        echo "=== splitk_tune target=${target} case=${case_name} ==="
        "./${build_dir}/profile_training_hotspots" --case "${case_name}"
    done
done

python3 scripts/summarize_splitk_tuning.py --job-id "${SLURM_JOB_ID}"
echo "=== Split-K Tuning Summary ==="
cat "results/splitk_tuning_${SLURM_JOB_ID}.csv"
echo "=== DONE ==="
