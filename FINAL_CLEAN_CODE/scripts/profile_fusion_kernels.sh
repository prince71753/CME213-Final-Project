#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o logs/profile_fusion_%j.out
#SBATCH -e logs/profile_fusion_%j.err
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
    bias_relu_unfused
    bias_relu_fused
    residual_ln_unfused
    residual_ln_fused
    ln_bwd_residual_unfused
    ln_bwd_residual_fused
)

echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvcc --version | tail -n 1
ncu --version | head -n 1
echo "tmpdir=${TMPDIR}"

PROFILE_BUILD_DIR="build_profile_${SLURM_JOB_ID}"

echo "=== Build ==="
make BUILD_DIR="${PROFILE_BUILD_DIR}" \
    "${PROFILE_BUILD_DIR}/test_fusion_benchmark" \
    "${PROFILE_BUILD_DIR}/test_model_reference" \
    "${PROFILE_BUILD_DIR}/profile_fusion_kernels"

echo "=== Correctness smoke ==="
./"${PROFILE_BUILD_DIR}"/test_fusion_benchmark
./"${PROFILE_BUILD_DIR}"/test_model_reference

NCU_ENABLED=1

for case_name in "${CASES[@]}"; do
    echo "=== Profiling ${case_name} ==="
    out_prefix="profiles/fusion_${case_name}_${SLURM_JOB_ID}"
    log_path="logs/profile_fusion_${SLURM_JOB_ID}_${case_name}.txt"
    if [ "${NCU_ENABLED}" -eq 1 ]; then
        set +e
        ncu --profile-from-start off \
            --set full \
            --target-processes all \
            --kernel-name-base demangled \
            --print-units base \
            --print-fp \
            -o "${out_prefix}" \
            ./"${PROFILE_BUILD_DIR}"/profile_fusion_kernels --case "${case_name}" \
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
            echo "ncu unavailable for ${case_name}; falling back to CUDA event timing." | tee -a "${log_path}"
            NCU_ENABLED=0
            ./"${PROFILE_BUILD_DIR}"/profile_fusion_kernels --case "${case_name}" \
                2>&1 | tee -a "${log_path}"
        fi
    else
        ./"${PROFILE_BUILD_DIR}"/profile_fusion_kernels --case "${case_name}" \
            2>&1 | tee "${log_path}"
    fi
done

python3 scripts/summarize_fusion_profiles.py \
    --job-id "${SLURM_JOB_ID}" \
    --output "results/fusion_profile_${SLURM_JOB_ID}.csv"

cp "results/fusion_profile_${SLURM_JOB_ID}.csv" results/fusion_profile.csv

echo "=== Fusion Profile Summary ==="
cat results/fusion_profile.csv
echo "=== DONE ==="
