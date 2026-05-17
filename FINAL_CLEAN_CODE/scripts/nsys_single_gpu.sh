#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/nsys_single_gpu_%j.out
#SBATCH -e logs/nsys_single_gpu_%j.err
#SBATCH --time=00:15:00

set -euo pipefail
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

profile_one() {
    local hidden="$1" ff="$2" head_dim="$3" steps="$4" lr="$5"
    local bdir="build_h${hidden}"
    local nsys_prefix="profiles/single_h${hidden}_${TAG}_${JOB}"

    echo "=== Build H${hidden} ==="
    make all BUILD_DIR="${bdir}" \
        USE_NVTX=1 \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"

    echo "=== NSys profile H${hidden} steps=${steps} ==="
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --stats=true \
        --force-overwrite=true \
        -o "${nsys_prefix}" \
        "./${bdir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}"

    # Convert stats to CSV via nsys stats. The .sqlite file is already there.
    if [ -f "${nsys_prefix}.sqlite" ]; then
        echo "=== Top CUDA kernels for H${hidden} ==="
        local csv="results/nsys_single_h${hidden}_${JOB}_kern.csv"
        nsys stats --report cuda_gpu_kern_sum --format csv \
            --output - "${nsys_prefix}.sqlite" > "${csv}"
        head -n 30 "${csv}" || true
    fi
}

profile_one 128 512  32 "${STEPS}" 3e-4
profile_one 256 1024 64 "${STEPS}" 3e-4

echo "=== DONE ${JOB} ==="
