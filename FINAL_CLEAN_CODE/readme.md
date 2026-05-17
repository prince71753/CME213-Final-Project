# final clean code package

this folder is the cleaned final package for the cuda, mpi, and openmp
mini-transformer training project.

## contents

- `include/`, `src/`, and `tests/`: cleaned source code.  each c++/cuda file
  has one short header comment and otherwise avoids explanatory comments.
- `scripts/`: build, validation, benchmark, and profiling scripts.
- `artifacts/results/`: csv and md result artifacts used for tables.
- `artifacts/logs/`: selected validation and benchmark logs.
- `artifacts/profile_summaries/`: selected profiling summaries.
- `plots/`: generated svg plots.
- `report/`: report draft and figures folder.
- `m4_writeup.md`: milestone 4 progress writeup.
- `final_product_summary.md`: final project summary and result map.
- `package_manifest.md`: package contents and validation notes.

## build

on the course cluster:

```bash
module load course/cme213/nvhpc/24.1
make -j8
make -j8 tests
make -j8 mpi
```

## validate

inside a gpu allocation or through slurm:

```bash
sbatch scripts/run_correctness_matrix.sh
```

for the longer validation script:

```bash
sbatch scripts/run_full_validation.sh
```

## run

single gpu:

```bash
./build/train inp.txt --epochs 1 --max-steps 100
```

four gpu mpi:

```bash
mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 100 --sync-mode overlap --bucket-kb 1024
```

openmp communication thread:

```bash
CME213_COMM_THREAD=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 100 --sync-mode overlap --bucket-kb 1024
```

experimental h512 ffn fp16 gemm:

```bash
CME213_FFN_FP16=1 ./build/train inp.txt --epochs 1 --max-steps 50 --lr 1e-4
```

## default result story

the final default single-gpu backend is `cublas_tc`.  the best distributed
result is the h256 openmp communication-thread path, which improves over pinned
overlap.  the h128 mpi special case uses custom kernels.  the h512 ffn fp16
path is kept only as an opt-in research result because it passed focused
reference checks but was not fully valid in the final repeated h512 training
sweep.
