#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:1
#SBATCH -o logs/cuda_graph_validate_%j.out
#SBATCH -e logs/cuda_graph_validate_%j.err
#SBATCH --time=00:25:00

set -uo pipefail
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

run_check() {
    local name="$1"; shift
    local out="logs/cuda_graph_validate_${JOB}_${name}.txt"
    echo "=== ${name} ==="
    "$@" > "${out}" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || grep -E 'FAIL|(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "${out}" >/dev/null; then
        echo "FAILED ${name} (rc=${rc}); tail follows:"
        tail -n 30 "${out}"
        return 1
    fi
    echo "PASSED ${name}"
    tail -n 5 "${out}"
}

echo "=== Clean build ==="
rm -rf build build_mpi build_h128 build_h256 build_h512
make all BUILD_DIR=build_h128 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
make all BUILD_DIR=build_h256 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
make all BUILD_DIR=build_h512 EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

for hidden in 128 256 512; do
    case $hidden in
        128) bdir=build_h128; backend=custom;    steps=100; lr=3e-4 ;;
        256) bdir=build_h256; backend=cublas_tc; steps=100; lr=3e-4 ;;
        512) bdir=build_h512; backend=cublas_tc; steps=50;  lr=1e-4 ;;
    esac
    echo "=== H${hidden} backend=${backend} ==="
    run_check "h${hidden}_${backend}_nograph" \
        env CME213_GEMM_BACKEND="${backend}" "./${bdir}/train" inp.txt \
            --epochs 1 --max-steps "${steps}" --lr "${lr}"
    run_check "h${hidden}_${backend}_graph" \
        env CME213_GEMM_BACKEND="${backend}" CME213_USE_CUDA_GRAPH=1 \
            "./${bdir}/train" inp.txt --epochs 1 --max-steps "${steps}" --lr "${lr}"
done

echo "=== Summary CSV ==="
csv="results/cuda_graph_validate_${JOB}.csv"
echo "hidden,backend,graph,steps,ms,tok_per_s,loss,valid" > "${csv}"
for f in logs/cuda_graph_validate_${JOB}_h*.txt; do
    base=$(basename "$f" .txt | sed "s/^cuda_graph_validate_${JOB}_//")
    hidden=$(echo "$base" | awk -F'_' '{print $1}' | sed 's/h//')
    # backend may have an underscore (cublas_tc); reconstruct
    # base = h<H>_<backend(maybe with _)>_<graph|nograph>
    graph=${base##*_}
    rest=${base%_*}
    backend=${rest#h${hidden}_}
    ms=$(grep -oE '[0-9]+ms' "$f" | tail -n 1 | tr -d 'ms')
    tok=$(grep -oE '[0-9]+ tok/s' "$f" | tail -n 1 | grep -oE '[0-9]+')
    loss=$(grep -oE 'avg_logged_loss=[0-9.]+' "$f" | tail -n 1 | sed 's/avg_logged_loss=//')
    valid=yes
    if grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$f" >/dev/null; then valid=no; fi
    echo "${hidden},${backend},${graph},,${ms},${tok},${loss},${valid}" >> "${csv}"
done
sort -t, -k1,1n -k2,2 -k3,3 "${csv}" -o "${csv}"
cat "${csv}"

echo "=== DONE ${JOB} ==="
