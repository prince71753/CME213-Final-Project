#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/lt_validation_pack_%j.out
#SBATCH -e logs/lt_validation_pack_%j.err
#SBATCH --time=00:55:00
#
# Full validation after cuBLASLt FFN1 fusion:
#   - default + MPI builds (requires: module load course/cme213/nvhpc/24.1)
#   - GPU unit tests (test_gemm may fail M=512 vs tiled — logged, non-fatal)
#   - Throughput A/B: cublas_tc with CME213_LT_FUSION=1 (opt-in Lt) vs unset (default)
#   - Nsight Systems H256: same A/B, kernel CSV grep for bias_relu / fused_relu
#
set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
mkdir -p logs results profiles "${HOME}/tmp"
export TMPDIR="${HOME}/tmp"

module load course/cme213/nvhpc/24.1
source scripts/common_env.sh

JOB="${SLURM_JOB_ID:-local}"
SUMMARY="results/lt_validation_pack_${JOB}_summary.md"

{
  echo "# LT validation pack (job ${JOB})"
  echo "Host: $(hostname)  Date: $(date -Is)"
  echo
} > "${SUMMARY}"

echo "=== [1/5] Clean build (single-GPU) + tests ===" | tee -a "${SUMMARY}"
make clean
make -j8 USE_NVTX=1
make tests -j8 USE_NVTX=1

echo "=== [2/5] GPU unit tests ===" | tee -a "${SUMMARY}"
run_test() {
  local name="$1" cmd="$2"
  echo "--- ${name} ---"
  if eval "${cmd}" > "logs/${JOB}_${name}.txt" 2>&1; then
    echo "${name}: PASS" | tee -a "${SUMMARY}"
    tail -n 2 "logs/${JOB}_${name}.txt" | tee -a "${SUMMARY}"
  else
    echo "${name}: FAIL (exit $?)" | tee -a "${SUMMARY}"
    tail -n 15 "logs/${JOB}_${name}.txt" | tee -a "${SUMMARY}"
  fi
  echo | tee -a "${SUMMARY}"
}

run_test test_gemm './build/test_gemm'
run_test test_attention './build/test_attention'
run_test test_layernorm './build/test_layernorm'
run_test test_model_reference 'cme213_clean_env CME213_GEMM_BACKEND=cublas_tc ./build/test_model_reference'

echo "=== [3/5] MPI build (train_mpi) ===" | tee -a "${SUMMARY}"
rm -rf build_mpi
if make mpi -j8 USE_NVTX=1 > "logs/${JOB}_make_mpi.txt" 2>&1; then
  echo "make mpi: PASS" | tee -a "${SUMMARY}"
  ls -la build_mpi/train_mpi | tee -a "${SUMMARY}"
else
  echo "make mpi: FAIL" | tee -a "${SUMMARY}"
  tail -n 30 "logs/${JOB}_make_mpi.txt" | tee -a "${SUMMARY}"
fi
echo | tee -a "${SUMMARY}"

echo "=== [4/5] Throughput A/B: cublas_tc, LT fusion on vs off ===" | tee -a "${SUMMARY}"
rm -rf build_lt_h128 build_lt_h256 build_lt_h512
make all BUILD_DIR=build_lt_h128 USE_NVTX=1 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_lt_h256 USE_NVTX=1 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
make all BUILD_DIR=build_lt_h512 USE_NVTX=1 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

bench_pair() {
  local label="$1" bdir="$2" steps="$3" lr="$4"
  local base="logs/lt_bench_${JOB}_${label}"
  echo "### ${label}" >> "${SUMMARY}"
  for fusion in off on; do
    local out="${base}_lt_${fusion}.txt"
    echo "Running ${label} LT=${fusion} ..."
    if [[ "${fusion}" == "off" ]]; then
      # Default path: Lt fusion opt-out (omit CME213_LT_FUSION).
      if ! cme213_clean_env CME213_GEMM_BACKEND=cublas_tc \
          "./${bdir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}" \
          > "${out}" 2>&1; then
        echo "${label} LT=${fusion}: train FAILED" | tee -a "${SUMMARY}"
        tail -n 8 "${out}" | tee -a "${SUMMARY}"
        continue
      fi
    else
      if ! cme213_clean_env CME213_GEMM_BACKEND=cublas_tc CME213_LT_FUSION=1 \
          "./${bdir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}" \
          > "${out}" 2>&1; then
        echo "${label} LT=${fusion}: train FAILED" | tee -a "${SUMMARY}"
        tail -n 8 "${out}" | tee -a "${SUMMARY}"
        continue
      fi
    fi
    if grep -qiE '(^|[^a-z])(nan|inf)([^a-z]|$)' "${out}"; then
      echo "${label} LT=${fusion}: DIVERGED" | tee -a "${SUMMARY}"
      tail -n 6 "${out}" | tee -a "${SUMMARY}"
      continue
    fi
    local line
    local ltvar="(unset)"
    [[ "${fusion}" == "on" ]] && ltvar="CME213_LT_FUSION=1"
    line="$(grep 'tok/s' "${out}" | tail -n 1 || true)"
    echo "- LT fusion **${fusion}** (\`${ltvar}\`): \`${line}\`" >> "${SUMMARY}"
    echo "${line}"
  done
  echo >> "${SUMMARY}"
}

bench_pair h128 build_lt_h128 100 3e-4
bench_pair h256 build_lt_h256 100 3e-4
bench_pair h512 build_lt_h512 50  1e-4

echo "=== [5/5] Nsight Systems H256 (30 steps): LT on vs off, kernel grep ===" | tee -a "${SUMMARY}"
STEPS=30
H=256
FF=1024
HD=64
bdir="build_lt_h${H}"
for fusion in off on; do
  tag="profiles/nsys_h${H}_cublas_tc_lt_${fusion}_${JOB}"
  unset CME213_LT_FUSION || true
  if [[ "${fusion}" == "on" ]]; then
    export CME213_LT_FUSION=1
  fi
  export CME213_GEMM_BACKEND=cublas_tc
  echo "NSys LT=${fusion} ..."
  nsys profile --trace=cuda,nvtx,osrt,cublas --sample=none --stats=true --force-overwrite=true \
    -o "${tag}" \
    "./${bdir}/train" inp.txt --epochs 1 --max-steps "${STEPS}" --lr 3e-4
  csv="results/nsys_h${H}_lt_${fusion}_${JOB}_kern.csv"
  nsys stats --report cuda_gpu_kern_sum --format csv --output - "${tag}.sqlite" > "${csv}"
  echo "### Nsys H${H} LT=${fusion}" >> "${SUMMARY}"
  echo '```' >> "${SUMMARY}"
  { echo "--- rows matching bias_relu | fused_relu | relu_backward (ili) ---"
    grep -iE 'bias_relu|fused_relu|relu_backward' "${csv}" || echo "(no matches)"
    echo "--- top 12 kernels by time ---"
    head -n 13 "${csv}"
  } | tee -a "${SUMMARY}"
  echo '```' >> "${SUMMARY}"
  echo >> "${SUMMARY}"
done

echo "=== DONE job ${JOB} ===" | tee -a "${SUMMARY}"
echo "Summary written: ${SUMMARY}"
