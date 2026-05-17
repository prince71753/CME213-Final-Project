#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/h512_redo_%j.out
#SBATCH -e logs/h512_redo_%j.err
#SBATCH --time=00:10:00

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1

JOB="${SLURM_JOB_ID:-local}"
HIDDEN=512
FF=2048
HEAD_DIM=128
BDIR=build_h512

echo "=== Build (correct H512 with head_dim=128) ==="
make all BUILD_DIR="${BDIR}" \
    EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${HIDDEN} -DFF_DIM=${FF} -DHEAD_DIM=${HEAD_DIM}"

echo "=== H512 single-GPU 50 steps ==="
"./${BDIR}/train" inp.txt --epochs 1 --max-steps 50 --lr 1e-4

echo "=== H512 single-GPU 100 steps ==="
"./${BDIR}/train" inp.txt --epochs 1 --max-steps 100 --lr 1e-4

echo "=== DONE ${JOB} ==="
