#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/cublas_mpi_validate_%j.out
#SBATCH -e logs/cublas_mpi_validate_%j.err
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
STEPS=100

echo "=== Build H128 MPI ==="
make mpi MPI_BUILD_DIR=build_mpi_h128 \
    EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=128 -DFF_DIM=512 -DHEAD_DIM=32'
echo "=== Build H256 MPI ==="
make mpi MPI_BUILD_DIR=build_mpi_h256 \
    EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=256 -DFF_DIM=1024 -DHEAD_DIM=64'
echo "=== Build H512 MPI ==="
make mpi MPI_BUILD_DIR=build_mpi_h512 \
    EXTRA_NVCC_FLAGS='-DHIDDEN_DIM=512 -DFF_DIM=2048 -DHEAD_DIM=128'

run_one() {
    local hidden="$1" mdir="$2" mode="$3" backend="$4" lr="$5"; shift 5
    local name="h${hidden}_${mode}_${backend}"
    local log="logs/cublas_mpi_validate_${JOB}_${name}.txt"
    echo "=== ${name} (${STEPS} steps) ==="
    env CME213_CUDA_AWARE_MPI=1 CME213_GEMM_BACKEND="${backend}" \
        mpirun -np 4 "./${mdir}/train_mpi" inp.txt \
            --epochs 1 --max-steps "${STEPS}" --lr "${lr}" "$@" \
            2>&1 | tee "${log}"
}

for hidden in 128 256 512; do
    if [ "${hidden}" = 512 ]; then
        lr=1e-4
    else
        lr=3e-4
    fi
    mdir="build_mpi_h${hidden}"
    for backend in custom cublas cublas_tc; do
        run_one "${hidden}" "${mdir}" blocking "${backend}" "${lr}" --sync-mode blocking
        run_one "${hidden}" "${mdir}" overlap  "${backend}" "${lr}" --sync-mode overlap --bucket-kb 256
    done
done

echo "=== Parse CSV ==="
csv="results/cublas_mpi_validate_${JOB}.csv"
echo "hidden,mode,backend,ms,tok_per_s,grad_sync_ms,checksum_span,valid" > "${csv}"
for f in logs/cublas_mpi_validate_${JOB}_h*.txt; do
    base=$(basename "$f" .txt)
    rest=${base#cublas_mpi_validate_${JOB}_}
    hidden=$(echo "$rest" | awk -F'_' '{print $1}' | sed 's/h//')
    mode=$(echo "$rest" | awk -F'_' '{print $2}')
    backend=$(echo "$rest" | awk -F'_' '{print $3}')
    ms=$(grep -oE '[0-9]+ms' "$f" | tail -n 1 | tr -d 'ms')
    tok=$(grep -oE '[0-9]+ tok/s' "$f" | tail -n 1 | grep -oE '[0-9]+')
    grad=$(grep -oE 'avg_grad_(sync|finish)=[0-9.]+ms' "$f" | tail -n 1 \
        | grep -oE '[0-9.]+' | head -n 1)
    span=$(grep -oE 'checksum_span=[0-9.e+-]+' "$f" | tail -n 1 \
        | sed 's/checksum_span=//')
    valid=yes
    if grep -E '(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)' "$f" >/dev/null; then valid=no; fi
    echo "${hidden},${mode},${backend},${ms},${tok},${grad},${span},${valid}" >> "${csv}"
done
cat "${csv}"

echo "=== DONE ${JOB} ==="
