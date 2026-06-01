#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/profile_%j.out
#SBATCH -e logs/profile_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs profiles
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

echo "=== Kernel Profiling with Nsight Compute ==="
echo "tmpdir=${TMPDIR}"
make all tests USE_NVTX=1

echo "--- GEMM profiling ---"
ncu --set full --target-processes all --force-overwrite \
    -o "profiles/gemm_profile_${SLURM_JOB_ID}" \
    ./build/test_gemm

echo "--- Attention profiling ---"
ncu --set full --target-processes all --force-overwrite \
    -o "profiles/attention_profile_${SLURM_JOB_ID}" \
    ./build/test_attention

echo "--- LayerNorm profiling ---"
ncu --set full --target-processes all --force-overwrite \
    -o "profiles/layernorm_profile_${SLURM_JOB_ID}" \
    ./build/test_layernorm

echo "=== Timeline profiling with Nsight Systems ==="
nsys profile --trace=cuda,nvtx --sample=none --stats=true --force-overwrite=true \
    -o "profiles/train_timeline_${SLURM_JOB_ID}" \
    ./build/train inp.txt --epochs 1 --max-steps 50

echo "Profiles saved to profiles/"
