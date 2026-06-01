#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/baseline_snapshot_%j.out
#SBATCH -e logs/baseline_snapshot_%j.err
#SBATCH --time=00:30:00

# Reproducible baseline snapshot of the current code state.
# Goals:
#   1. Build single-GPU and MPI binaries at H128/H256/H512.
#   2. Time each configuration for a meaningful number of steps.
#   3. Write a CSV row per measurement so post-change runs can diff.
# This must NOT be edited as optimizations are added; instead, after each
# change run scripts/baseline_snapshot.sh again and compare CSVs by job id.

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
TAG="${TAG:-baseline}"
STEPS_SINGLE="${STEPS_SINGLE:-100}"
STEPS_H512="${STEPS_H512:-50}"
STEPS_MPI="${STEPS_MPI:-100}"
LR_SMALL="${LR_SMALL:-3e-4}"
LR_H512="${LR_H512:-1e-4}"
BUCKET_KB="${BUCKET_KB:-256}"

OUT_CSV="results/baseline_snapshot_${JOB}.csv"
echo "tag,job,hidden,mode,ranks,steps,ms,tok_per_s,grad_sync_ms,checksum_span,valid" > "${OUT_CSV}"

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
nvcc --version | tail -n 1

parse_and_log() {
    # Args: tag, hidden, mode, ranks, log
    local tag="$1" hidden="$2" mode="$3" ranks="$4" log="$5"
    local ms tok grad span valid
    ms=$(grep -oE 'Epoch [0-9]+:.*[0-9]+ms' "$log" | tail -n 1 \
        | grep -oE '[0-9]+ms' | head -n 1 | tr -d 'ms') || ms=""
    tok=$(grep -oE '[0-9]+ tok/s' "$log" | tail -n 1 \
        | grep -oE '[0-9]+' | head -n 1) || tok=""
    grad=$(grep -oE 'avg_grad_(sync|finish)=[0-9.]+ms' "$log" \
        | tail -n 1 | grep -oE '[0-9.]+' | head -n 1) || grad=""
    span=$(grep -oE 'checksum_span=[0-9.e+-]+' "$log" \
        | tail -n 1 | sed 's/checksum_span=//') || span=""
    if grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$log" >/dev/null; then
        valid=no
    else
        valid=yes
    fi
    echo "${tag},${JOB},${hidden},${mode},${ranks},${STEPS_SINGLE},${ms},${tok},${grad},${span},${valid}" \
        >> "${OUT_CSV}"
    echo "[parsed] hidden=${hidden} mode=${mode} ranks=${ranks} ms=${ms} tok/s=${tok} valid=${valid}"
}

build_variant() {
    # Args: hidden ff head_dim build_dir mpi_build_dir
    local hidden="$1" ff="$2" head_dim="$3" bdir="$4" mdir="$5"
    echo "=== Build ${bdir} hidden=${hidden} ==="
    make all mpi BUILD_DIR="${bdir}" MPI_BUILD_DIR="${mdir}" \
        EXTRA_NVCC_FLAGS="-DHIDDEN_DIM=${hidden} -DFF_DIM=${ff} -DHEAD_DIM=${head_dim}"
}

# --- Single-GPU H128/H256/H512 ---
build_variant 128 512   32 build_h128 build_mpi_h128
build_variant 256 1024  64 build_h256 build_mpi_h256
build_variant 512 2048 128 build_h512 build_mpi_h512

run_single() {
    local hidden="$1" bdir="$2" steps="$3" lr="$4"
    local log="logs/baseline_snapshot_${JOB}_h${hidden}_single.out"
    echo "=== Single-GPU H${hidden} steps=${steps} ==="
    "./${bdir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}" \
        2>&1 | tee "${log}"
    parse_and_log "${TAG}" "${hidden}" "single" 1 "${log}"
}

run_single 128 build_h128 "${STEPS_SINGLE}" "${LR_SMALL}"
run_single 256 build_h256 "${STEPS_SINGLE}" "${LR_SMALL}"
run_single 512 build_h512 "${STEPS_H512}"   "${LR_H512}"

# --- 4-rank MPI: blocking + overlap at H128/H256 ---
run_mpi() {
    local hidden="$1" mdir="$2" mode_name="$3"; shift 3
    local log="logs/baseline_snapshot_${JOB}_h${hidden}_${mode_name}.out"
    echo "=== 4-rank H${hidden} ${mode_name} ==="
    CME213_CUDA_AWARE_MPI=1 mpirun -np 4 "./${mdir}/train_mpi" \
        inp.txt --epochs 1 --max-steps "${STEPS_MPI}" --lr "${LR_SMALL}" "$@" \
        2>&1 | tee "${log}"
    parse_and_log "${TAG}" "${hidden}" "${mode_name}" 4 "${log}"
}

run_mpi 128 build_mpi_h128 blocking --sync-mode blocking
run_mpi 128 build_mpi_h128 overlap  --sync-mode overlap --bucket-kb "${BUCKET_KB}"
run_mpi 256 build_mpi_h256 blocking --sync-mode blocking
run_mpi 256 build_mpi_h256 overlap  --sync-mode overlap --bucket-kb "${BUCKET_KB}"

echo "=== Baseline CSV ==="
cat "${OUT_CSV}"

echo "=== DONE ${JOB} ==="
