#!/bin/bash
# Profile the H256 GEMM shapes that dominate the Nsight Systems timeline.
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/profile_hotspots_%j.out
#SBATCH -e logs/profile_hotspots_%j.err
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

CASES=(
    splitk_dW1
    splitk_dW2
    splitk_qkv
    splitk_dWout
    bt_ff1
    fwd_ff1
    batched_qkv
)

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvcc --version | tail -n 1
ncu --version | head -n 1
echo "tmpdir=${TMPDIR}"

PROFILE_BUILD_DIR="build_hotspots_${SLURM_JOB_ID}"

echo "=== Build ==="
make BUILD_DIR="${PROFILE_BUILD_DIR}" \
    "${PROFILE_BUILD_DIR}/profile_training_hotspots"

for case_name in "${CASES[@]}"; do
    echo "=== Profiling ${case_name} ==="
    out_prefix="profiles/hotspot_${case_name}_${SLURM_JOB_ID}"
    log_path="logs/profile_hotspots_${SLURM_JOB_ID}_${case_name}.txt"
    set +e
    ncu --profile-from-start off \
        --set full \
        --target-processes all \
        --kernel-name-base demangled \
        --print-units base \
        --print-fp \
        -o "${out_prefix}" \
        ./"${PROFILE_BUILD_DIR}"/profile_training_hotspots --case "${case_name}" \
        2>&1 | tee "${log_path}"
    ncu_status=${PIPESTATUS[0]}
    set -e

    if [ "${ncu_status}" -eq 0 ] && [ -f "${out_prefix}.ncu-rep" ]; then
        ncu --import "${out_prefix}.ncu-rep" \
            --csv \
            --page raw \
            --print-units base \
            --print-fp \
            > "${out_prefix}_raw.csv"
    else
        echo "ncu failed for ${case_name}; keeping CUDA event timing only." | tee -a "${log_path}"
    fi
done

python3 scripts/summarize_hotspot_profiles.py \
    --job-id "${SLURM_JOB_ID}" \
    --output "results/hotspot_profile_${SLURM_JOB_ID}.csv"

cp "results/hotspot_profile_${SLURM_JOB_ID}.csv" results/hotspot_profile.csv

echo "=== Hotspot Profile Summary ==="
cat results/hotspot_profile.csv
echo "=== DONE ==="
