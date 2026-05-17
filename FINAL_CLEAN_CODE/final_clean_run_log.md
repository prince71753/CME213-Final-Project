# final clean run log

date: may 17, 2026

folder: `cme213/final_project/FINAL_CLEAN_CODE`

## validation

job `85267` passed the final correctness matrix.  it built `all`, `tests`,
and `mpi`, then passed the gemm, attention, layernorm, model-reference,
validate-config, single-gpu smoke, and four-rank mpi smoke checks.

job `85272` passed focused h512 reference checks for the fp16 storage-only and
ffn fp16 paths.  these checks do not make fp16 a default, because repeated
training validity is the deciding criterion.

## repeated benchmarks

single-gpu warm summaries drop repeat 1.

| case | best default result |
|---|---|
| h128 single gpu | `cublas_tc`, `2.494m tok/s`, job `85265` |
| h256 single gpu | `auto`, `1.596m tok/s`, job `85266` |
| h512 single gpu | `cublas_tc`, `0.732m tok/s`, job `85264` |
| h128 four gpu | custom overlap, `4.971m tok/s`, job `85269` |
| h256 four gpu | openmp-thread overlap, `2.269m tok/s`, job `85271` |
| h512 four gpu | openmp-thread overlap at `2048 kb`, `0.879m tok/s`, job `85268` |

the h512 openmp result is a short three-repeat supportive check at `lr=5e-5`.
the h256 openmp result is the stronger headline distributed result.

## profiling

| job | profile |
|---|---|
| `85273` | nsight compute hotspot profile for tensor-core gemms |
| `85274` | nsight compute fusion profile and regenerated roofline data |
| `85275` | nsight systems h128 mpi custom vs cublas_tc pair |
| `85276` | nsight systems h256 direct vs pinned vs openmp communication pair |

the current roofline artifacts are:

- `results/roofline_fusion.csv`;
- `results/roofline_combined.csv`;
- `plots/roofline_fusion.svg`;
- `plots/roofline_combined.svg`.

the current final clean figures are in `plots/final_*.svg` and copied into
`report/figures/`.

## source of truth

use `results/final_clean_main_table.csv` for final numbers.  a copy is stored
at `artifacts/results/final_clean_main_table.csv`.

do not make a final energy-per-token claim.  repeated time-sampled nvml power
data was not collected.
