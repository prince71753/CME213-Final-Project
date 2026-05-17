#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o logs/null_variance_bench_%j.out
#SBATCH -e logs/null_variance_bench_%j.err
#SBATCH --time=01:30:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${JOB:-${SLURM_JOB_ID:-local}}"
CASE="${CASE:-h256_single_cublas_tc}"
REPEATS="${REPEATS:-50}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"

echo "=== Null variance benchmark ==="
echo "job=${JOB}"
echo "case=${CASE}"
echo "repeats=${REPEATS}"
echo "cooldown_sec=${COOLDOWN_SEC}"
hostname
cme213_log_gpu_state "null_start"

case "${CASE}" in
    h256_single_cublas_tc)
        env JOB="${JOB}_null_h256_single_cublas_tc" \
            HIDDEN=256 BACKENDS=cublas_tc REPEATS="${REPEATS}" \
            STEPS="${STEPS:-100}" COOLDOWN_SEC="${COOLDOWN_SEC}" \
            bash scripts/run_single_gpu_repeated_bench.sh
        ;;
    h256_mpi_blocking)
        env JOB="${JOB}_null_h256_mpi_blocking" \
            HIDDEN=256 REPEATS="${REPEATS}" STEPS="${STEPS:-50}" \
            BUCKETS_KB="" COOLDOWN_SEC="${COOLDOWN_SEC}" \
            GEMM_BACKEND="${GEMM_BACKEND:-auto}" \
            bash scripts/run_training_bucket_sweep.sh
        ;;
    h256_mpi_overlap_1024)
        env JOB="${JOB}_null_h256_mpi_overlap_1024" \
            HIDDEN=256 REPEATS="${REPEATS}" STEPS="${STEPS:-50}" \
            BUCKETS_KB=1024 COOLDOWN_SEC="${COOLDOWN_SEC}" \
            GEMM_BACKEND="${GEMM_BACKEND:-auto}" \
            bash scripts/run_training_bucket_sweep.sh
        ;;
    h512_mpi_blocking)
        env JOB="${JOB}_null_h512_mpi_blocking" \
            HIDDEN=512 REPEATS="${REPEATS}" STEPS="${STEPS:-50}" \
            BUCKETS_KB="" COOLDOWN_SEC="${COOLDOWN_SEC}" \
            GEMM_BACKEND="${GEMM_BACKEND:-auto}" \
            bash scripts/run_training_bucket_sweep.sh
        ;;
    *)
        echo "Unknown CASE='${CASE}'" >&2
        echo "Valid cases: h256_single_cublas_tc h256_mpi_blocking h256_mpi_overlap_1024 h512_mpi_blocking" >&2
        exit 2
        ;;
esac

cme213_log_gpu_state "null_end"
python3 scripts/summarize_null_variance.py --tag "${JOB}" || true
echo "=== DONE null variance job=${JOB} ==="
