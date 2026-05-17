# package manifest

## code

- `include/`: public headers and data structures.
- `src/`: training driver, model code, dataset code, kernels, and distributed
  synchronization code.
- `tests/`: correctness, benchmark, and profiling harnesses.
- `scripts/`: slurm, benchmark, validation, and plotting scripts.
- `Makefile`: single-gpu, test, mpi, and profiling builds.
- `inp.txt`: input text corpus used by the training driver.

## writeups

- `m4_writeup.md`: milestone 4 progress report.
- `final_product_summary.md`: compact final result summary.
- `final_clean_run_log.md`: final validation, benchmark, and profile run log.
- `readme.md`: build and run instructions for this package.

## artifacts

- `artifacts/results/`: csv and markdown result files.
- `artifacts/logs/`: selected logs copied from final validation, benchmark, and
  profiling runs.
- `artifacts/profile_summaries/`: selected nsight summary files and roofline
  csvs.
- `plots/`: svg figures generated during the project.
- `report/`: report draft workspace.

the final clean figures generated from the current summaries are:

- `plots/final_single_gpu_backend.svg`;
- `plots/final_h128_mpi_backend.svg`;
- `plots/final_h256_comm_modes.svg`;
- `plots/final_h512_comm_buckets.svg`;
- `plots/final_fusion_event_speedups.svg`;
- `plots/roofline_combined.svg`;
- `plots/roofline_fusion.svg`;
- `plots/strong_scaling_speedup.svg`;
- `plots/strong_scaling_efficiency.svg`.

the main consolidated result table is:

- `results/final_clean_main_table.csv`;
- `artifacts/results/final_clean_main_table.csv`.

## cleanup policy

the cleaned source files keep only one short lowercase header comment per
c++/cuda source file.  required api names, environment variable names, cuda
symbols, mpi symbols, and parser-dependent log strings are intentionally left
unchanged so the code still builds and the benchmark scripts still parse logs
correctly.

slurm scripts use `SLURM_SUBMIT_DIR` when available, so they can be submitted
from this package without depending on the original project path.

## final clean validation

the cleaned package was validated from this folder on may 17, 2026.

```bash
sbatch scripts/run_correctness_matrix.sh
```

clean-package validation:

- job `85267`;
- summary: `results/correctness_matrix_85267.md`;
- artifact copy: `artifacts/results/correctness_matrix_85267.md`;
- result: pass.

the validation job built `all`, `tests`, and `mpi`, then passed:

- `test_gemm`;
- `test_attention`;
- `test_layernorm`;
- `test_model_reference` for `custom`, `cublas`, `cublas_tc`, and
  `cublas_tc_lt`;
- single-gpu smoke checks;
- `--validate-config`;
- four-rank mpi blocking and pinned-overlap smoke checks.

## final benchmark and profile jobs

| job | purpose | primary artifact |
|---|---|---|
| `85264` | h512 single-gpu repeated benchmark | `results/single_gpu_repeated_bench_summary_final_clean_h512_single_warm_r2.csv` |
| `85265` | h128 single-gpu repeated benchmark | `results/single_gpu_repeated_bench_summary_final_clean_h128_single_warm_r2.csv` |
| `85266` | h256 single-gpu repeated benchmark | `results/single_gpu_repeated_bench_summary_final_clean_h256_single_warm_r2.csv` |
| `85267` | correctness matrix | `results/correctness_matrix_85267.md` |
| `85268` | h512 communication-thread bucket check | `results/training_bucket_sweep_summary_comm_thread_final_clean_h512_comm_buckets.csv` |
| `85269` | h128 mpi backend sweep | `results/training_bucket_sweep_summary_h128_mpi_backend_final_clean_h128_mpi.csv` |
| `85270` | h256 strong scaling | `results/strong_scaling_85270.csv` |
| `85271` | h256 communication mode sweep | `results/training_bucket_sweep_summary_comm_thread_final_clean_h256_comm.csv` |
| `85272` | h512 fp16 reference checks | `logs/h512_fp16_model_ref_85272.out` |
| `85273` | nsight compute hotspot profile | `results/hotspot_profile.csv` |
| `85274` | nsight compute fusion and roofline profile | `results/fusion_profile.csv` |
| `85275` | nsight systems h128 mpi backend pair | `results/h128_mpi_custom_85275_rank0_nsys_summary.csv` |
| `85276` | nsight systems h256 communication pair | `results/h256_comm_openmp_thread_85276_rank0_nsys_summary.csv` |

## power data

time-sampled nvml power data was not collected as a repeated final table.
start and end gpu states appear in some logs, but the final report should not
make an energy-per-token claim from those snapshots.
