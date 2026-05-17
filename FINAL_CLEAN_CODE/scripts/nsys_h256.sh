#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/nsys_h256_%j.out
#SBATCH -e logs/nsys_h256_%j.err
#SBATCH --time=00:10:00

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs profiles results
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${SLURM_JOB_ID:-local}"
STEPS="${STEPS:-30}"
TAG="${TAG:-baseline}"

hidden=256; ff=1024; head_dim=64
bdir="build_h${hidden}"
nsys_prefix="profiles/single_h${hidden}_${TAG}_${JOB}"

echo "=== Build H${hidden} ==="
make all BUILD_DIR="${bdir}" \
    USE_NVTX=1 EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"

echo "=== NSys profile H${hidden} steps=${STEPS} ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=true \
    --force-overwrite=true \
    -o "${nsys_prefix}" \
    "./${bdir}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr 3e-4

if [ -f "${nsys_prefix}.sqlite" ]; then
    csv="results/nsys_single_h${hidden}_${JOB}_kern.csv"
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output - "${nsys_prefix}.sqlite" > "${csv}"
    echo "=== Top CUDA kernels for H${hidden} ==="
    head -n 30 "${csv}" || true
fi

echo "=== DONE ${JOB} ==="
