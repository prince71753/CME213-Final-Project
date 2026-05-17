# final product summary

this package contains the final cleaned cuda, mpi, and openmp implementation
for the mini-transformer training project.  the current final evidence is
consolidated in `results/final_clean_main_table.csv`.

## what was built

- a small character-level transformer training loop;
- custom cuda kernels for gemm variants, attention, layernorm, embeddings,
  activation functions, cross entropy, gradient clipping, and adam;
- cublas and cublas tensor-core backends for larger gemms;
- an mpi data-parallel training path with one rank per gpu;
- blocking allreduce, bucketed pinned overlap, and an openmp communication
  thread for bucketed gradient allreduce;
- correctness tests against cpu references and finite differences;
- benchmark and profiling scripts for single-gpu, multi-gpu, scaling, roofline,
  nsight systems, and nsight compute analysis.

## final clean-run evidence

| result | evidence |
|---|---|
| cleaned package correctness matrix passes | job `85267`, `results/correctness_matrix_85267.md` |
| h128 single-gpu best path is cublas tensor cores | job `85265`: `2.494m tok/s` mean for `cublas_tc`, repeat 1 dropped |
| h256 single-gpu best path is auto/cublas tensor cores | job `85266`: `1.596m tok/s` mean for `auto`, repeat 1 dropped |
| h512 single-gpu default stays cublas tensor cores | job `85264`: `0.732m tok/s` mean, all valid |
| h512 ffn fp16 is not a default | job `85264`: `0.761m tok/s` mean over valid runs but only `3/4` valid after warmup |
| h128 mpi prefers custom kernels | job `85269`: custom overlap `4.971m tok/s`, cublas_tc overlap `3.929m tok/s` |
| h256 openmp communication thread is the main distributed win | job `85271`: `2.269m tok/s`, `1.103x` over pinned overlap and `1.496x` over direct blocking |
| h512 openmp communication thread is promising but supportive | job `85268`: best short run `0.879m tok/s` at `2048 kb`, `3/3` valid |
| final strong-scaling check is documented | job `85270`, h256 fixed total batch; overlap helps but efficiency remains low |
| final ncu hotspot profile is current | job `85273`, `results/hotspot_profile.csv` |
| final fusion and roofline profile is current | job `85274`, `results/fusion_profile.csv`, `results/roofline_fusion.csv` |
| final nsys communication profile is current | job `85276`, direct vs pinned vs openmp summaries |
| final h128 mpi profile pair is current | job `85275`, custom vs cublas_tc summaries |

## kept defaults

- cublas tensor-core backend for single-gpu auto mode;
- custom backend for the h128 mpi special case;
- residual plus layernorm fusion;
- layernorm backward plus residual fusion;
- fused bias plus relu after final clean profiling;
- bucketed communication overlap;
- openmp communication-thread overlap for h256 distributed runs.

## not kept as defaults

- cublaslt epilogue fusion;
- nccl backend;
- deferred gradient averaging;
- fp16 storage-only;
- ffn fp16 gemm, because the h512 repeated training run was not fully valid;
- host-staged pinned overlap at h512.

## final report story

the strongest story is not that every idea worked.  the strongest story is that
the project built a correct distributed cuda workload, measured it carefully,
and used profiling to explain tradeoffs:

- tensor cores help larger single-gpu gemms;
- fusion reduces memory traffic, but the amount of end-to-end benefit depends
  on the surrounding work;
- h128 has a distributed backend crossover where custom kernels beat cublas_tc;
- h256 is the cleanest communication-overlap case;
- h512 exposes communication and numerical-stability limits;
- the openmp communication thread is a real hybrid mpi/openmp/cuda contribution.

## notes for final writing

- use `results/final_clean_main_table.csv` as the source of truth;
- use unprofiled repeated runs for throughput claims;
- use nsight systems and nsight compute for mechanism, not headline speed;
- do not make a power or energy claim, because no repeated time-sampled nvml
  table was collected;
- keep ffn fp16 as an experimental result, not a shipped default.
