#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/nsys_cublas_%j.out
#SBATCH -e logs/nsys_cublas_%j.err
#SBATCH --time=00:15:00

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
STEPS=30

profile_one() {
    local hidden="$1" ff="$2" head_dim="$3"
    local bdir="build_h${hidden}"

    echo "=== Build H${hidden} ==="
    make all BUILD_DIR="${bdir}" \
        USE_NVTX=1 \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"

    for backend in custom cublas cublas_tc; do
        local tag="profiles/single_h${hidden}_${backend}_${JOB}"
        echo "=== NSys H${hidden} ${backend} ==="
        cme213_clean_env CME213_GEMM_BACKEND="${backend}" nsys profile \
            --trace=cuda,nvtx,osrt,cublas --sample=none --stats=true \
            --force-overwrite=true -o "${tag}" \
            "./${bdir}/train" inp.txt --epochs 1 --max-steps "${STEPS}"
        nsys stats --report cuda_gpu_kern_sum --format csv \
            --output - "${tag}.sqlite" \
            > "results/nsys_h${hidden}_${backend}_${JOB}_kern.csv"
        echo "=== Top 15 H${hidden} ${backend} ==="
        head -n 16 "results/nsys_h${hidden}_${backend}_${JOB}_kern.csv" || true
        echo
    done
}

profile_one 128 512   32
profile_one 256 1024  64
profile_one 512 2048 128

echo "=== DONE ${JOB} ==="
