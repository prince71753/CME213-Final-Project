#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/cublas_validate_%j.out
#SBATCH -e logs/cublas_validate_%j.err
#SBATCH --time=00:25:00

# Validate the cuBLAS GEMM backend.
# 1. Clean build with cuBLAS linked.
# 2. Run kernel-level GEMM tests with CME213_GEMM_BACKEND=cublas.
# 3. Run model reference with each backend.
# 4. Smoke training for 20 steps with each backend and compare loss.
# 5. Time 100 steps with each backend at H128 and H256.

set -uo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results profiles
mkdir -p "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"
module load course/cme213/nvhpc/24.1

JOB="${SLURM_JOB_ID:-local}"

echo "=== Clean build ==="
make clean
make all tests

run_check() {
    local name="$1"; shift
    local out="logs/cublas_validate_${JOB}_${name}.txt"
    echo "=== ${name} ==="
    "$@" > "${out}" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || grep -E 'FAIL|(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "${out}" >/dev/null; then
        echo "FAILED ${name} (rc=${rc}); tail follows:"
        tail -n 20 "${out}"
        return 1
    fi
    echo "PASSED ${name}"
    tail -n 5 "${out}"
}

BACKENDS=(custom cublas cublas_tc)

echo "=== Kernel tests (each backend) ==="
for b in "${BACKENDS[@]}"; do
    run_check "test_gemm_${b}" env CME213_GEMM_BACKEND=$b ./build/test_gemm
    run_check "test_attn_${b}" env CME213_GEMM_BACKEND=$b ./build/test_attention
    run_check "test_ln_${b}"   env CME213_GEMM_BACKEND=$b ./build/test_layernorm
done

echo "=== Model reference (each backend) ==="
for b in "${BACKENDS[@]}"; do
    run_check "model_ref_${b}" env CME213_GEMM_BACKEND=$b ./build/test_model_reference
done

echo "=== H128 100-step training (each backend) ==="
for b in "${BACKENDS[@]}"; do
    run_check "train_${b}_h128_100" env CME213_GEMM_BACKEND=$b ./build/train \
        inp.txt --epochs 1 --max-steps 100
done

echo "=== H256 100-step training (each backend) ==="
make all BUILD_DIR=build_h256 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
for b in "${BACKENDS[@]}"; do
    run_check "train_${b}_h256_100" env CME213_GEMM_BACKEND=$b ./build_h256/train \
        inp.txt --epochs 1 --max-steps 100
done

echo "=== H512 50-step training (each backend) ==="
make all BUILD_DIR=build_h512 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'
for b in "${BACKENDS[@]}"; do
    run_check "train_${b}_h512_50" env CME213_GEMM_BACKEND=$b ./build_h512/train \
        inp.txt --epochs 1 --max-steps 50 --lr 1e-4
done

echo "=== Summary CSV ==="
csv="results/cublas_validate_${JOB}.csv"
echo "hidden,backend,steps,ms,tok_per_s,loss,valid" > "${csv}"
for f in logs/cublas_validate_${JOB}_train_*; do
    name=$(basename "$f" .txt | sed "s/^cublas_validate_${JOB}_//")
    # name like train_cublas_tc_h128_100 or train_custom_h128_100
    rest=${name#train_}
    backend=$(echo "$rest" | sed -E 's/_h[0-9]+_[0-9]+$//')
    suffix=${rest#${backend}_}
    hidden=$(echo "$suffix" | awk -F'_' '{print $1}' | sed 's/h//')
    steps=$(echo "$suffix" | awk -F'_' '{print $2}')
    ms=$(grep -oE '[0-9]+ms' "$f" | tail -n 1 | tr -d 'ms')
    tok=$(grep -oE '[0-9]+ tok/s' "$f" | tail -n 1 | grep -oE '[0-9]+')
    loss=$(grep -oE 'avg_logged_loss=[0-9.]+' "$f" | tail -n 1 | sed 's/avg_logged_loss=//')
    valid=yes
    if grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$f" >/dev/null; then valid=no; fi
    echo "${hidden},${backend},${steps},${ms},${tok},${loss},${valid}" >> "${csv}"
done
sort -t, -k1,1n -k2,2 "${csv}" -o "${csv}"
cat "${csv}"

echo "=== DONE ${JOB} ==="
