#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/single_gpu_profile_snapshot_%j.out
#SBATCH -e logs/single_gpu_profile_snapshot_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs profiles results "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${SLURM_JOB_ID:-local}"
HIDDEN="${HIDDEN:-256}"
FF=$((4 * HIDDEN))
HEAD_DIM=$((HIDDEN / 4))
BACKEND="${BACKEND:-cublas_tc}"
STEPS="${STEPS:-30}"
LR="${LR:-}"
LT_FUSION="${LT_FUSION:-0}"
BUILD_DIR="${BUILD_DIR:-build_profile_h${HIDDEN}_${JOB}}"

if [ -z "${LR}" ]; then
    if [ "${HIDDEN}" -eq 512 ]; then
        LR=1e-4
    else
        LR=3e-4
    fi
fi

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "hidden=${HIDDEN}"
echo "backend=${BACKEND}"
echo "steps=${STEPS}"
echo "lr=${LR}"
echo "lt_fusion=${LT_FUSION}"
echo "tmpdir=${TMPDIR}"

echo "=== Build hidden=${HIDDEN} ==="
make all BUILD_DIR="${BUILD_DIR}" \
    USE_NVTX=1 \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

tag="single_h${HIDDEN}_${BACKEND}_lt${LT_FUSION}_${JOB}"
profile="profiles/${tag}"

echo "=== Nsight Systems ${tag} ==="
if [ "${LT_FUSION}" = "1" ]; then
    cme213_clean_env CME213_GEMM_BACKEND="${BACKEND}" CME213_LT_FUSION=1 \
        nsys profile --trace=cuda,nvtx,osrt,cublas --sample=none --stats=true \
        --force-overwrite=true -o "${profile}" \
        "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}"
else
    cme213_clean_env CME213_GEMM_BACKEND="${BACKEND}" \
        nsys profile --trace=cuda,nvtx,osrt,cublas --sample=none --stats=true \
        --force-overwrite=true -o "${profile}" \
        "./${BUILD_DIR}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr "${LR}"
fi

echo "=== Extract NSys stats ${tag} ==="
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
    --format csv --output - "${profile}.sqlite" > "results/${tag}_nsys_stats.csv"
python3 scripts/extract_nsys_profile_summary.py \
    --sqlite "${profile}.sqlite" \
    --label "${tag}" \
    --out "results/${tag}_nsys_summary.csv"

echo "=== DONE job=${JOB} ==="
