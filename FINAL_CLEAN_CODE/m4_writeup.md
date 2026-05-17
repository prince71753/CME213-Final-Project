# milestone 4 progress report

group: kwame ocran, jad bitar

## 1. summary of progress

since milestone 3, the project moved from a single-gpu cuda training pipeline
to a distributed multi-gpu training system.  the model is still a deliberately
small character-level transformer with token embeddings, position embeddings,
one self-attention block, layer normalization, an mlp, residual paths, and an
output projection.  the goal is controlled systems measurement rather than
language-model quality.

the distributed version is complete for data-parallel training.  each mpi rank
owns one gpu, holds a full copy of the parameters, receives a different slice
of the batch stream, runs forward and backward locally, and synchronizes
gradients before adam updates.  the code supports blocking allreduce, bucketed
overlap through nonblocking mpi, and an opt-in openmp communication thread that
runs blocking cuda-aware allreduce calls on ready gradient buckets.

since the milestone 3 submission, the largest changes are the mpi training
driver, gradient bucket synchronization, cuda-aware mpi support, nvtx ranges,
the openmp communication-thread path, and repeated benchmark scripts with
validity checks.  the single-gpu code also gained a cublas tensor-core backend,
which is now the default for single-gpu runs.

## 2. distributed algorithm and data decomposition

the implementation uses data parallelism.  model parameters are replicated on
each rank, while batches are rank-sharded.  for a world size of four, rank zero
uses batches `0, 4, 8, ...`, rank one uses `1, 5, 9, ...`, and so on.  this
keeps the transformer kernels unchanged: each rank still runs a normal local
forward and backward pass.

after backward propagation, each rank owns gradients for the same parameter
tensors but computed from different data.  the required communication is
therefore a gradient allreduce over every parameter gradient.  after the
allreduce, each rank applies the same adam update and the replicated models
remain synchronized.

we considered model parallelism and two-dimensional tensor partitioning, but
they are not a good fit for this project size.  the hidden dimensions are
small enough that splitting individual matrices across ranks would add more
communication and code complexity than useful parallelism.  data parallelism
matches the project goal better because it isolates the systems question:
when can gradient communication be hidden behind backpropagation?

## 3. mpi implementation

the baseline distributed path uses `MPI_Allreduce` on gradients after the
backward pass.  this is simple and reliable, but all communication is exposed
after compute finishes.

the overlap path divides gradients into buckets.  once a bucket has been
produced by backward kernels, the code starts an asynchronous reduction for
that bucket while later backward kernels continue.  for portability on the
cluster, the original nonblocking overlap path stages through pinned host
memory when direct device-buffer nonblocking mpi is unreliable.

the strongest distributed addition is the openmp communication thread.  this
path requests `MPI_THREAD_MULTIPLE`, records cuda events when buckets become
ready, and lets a persistent openmp worker wait on the event and issue a
blocking cuda-aware `MPI_Allreduce` directly on the device buffer.  the main
thread continues backward computation and waits for all buckets before adam.
this avoids the pinned host staging cost while still creating overlap.

## 4. c++ wrapper and cuda integration

the host driver initializes mpi, chooses a gpu per rank, builds the local
transformer, and runs the same cuda forward and backward code used in the
single-gpu path.  the model stores parameters and gradients in contiguous
device buffers.  this layout makes gradient synchronization easier because the
distributed code can describe parameter ranges as contiguous buckets.

the cuda kernels did not need to be rewritten for mpi.  the main integration
work was around launch ordering, stream synchronization, bucket readiness, and
optimizer timing.  for the openmp thread, the communication worker calls
`cudaSetDevice`, waits on cuda events, and then performs the allreduce.  nvtx
ranges mark backward, gradient synchronization, openmp event waits, and openmp
allreduce calls so nsight systems can show the communication structure.

## 5. correctness testing

correctness is checked at several levels.  the single-gpu tests compare gemm,
attention, layernorm, fusion kernels, and the full model against cpu or simple
reference implementations.  the full model reference checks forward loss,
logits, selected finite-difference gradients, and the first adam update.

for the distributed path, short mpi smoke tests run with multiple ranks and
check that the checksum span across ranks stays near zero after training.  this
confirms that all ranks keep the replicated parameters synchronized.  recent
correctness matrix jobs passed for the default path, explicit cublas backends,
mpi blocking, mpi overlap, and the openmp communication-thread path.

## 6. performance, scaling, and profiling

the final single-gpu default is the cublas tensor-core backend.  after
environment cleanup, it is best for h128, h256, and h512 single-gpu runs.
there is one distributed exception: at h128 with four mpi ranks, explicit
custom kernels outperform cublas tensor cores.  the likely reason is that the
per-rank problem is small enough that fixed library overhead and mpi gating
matter more than tensor-core arithmetic throughput.

the cleanest distributed result is h256.  pinned bucketed overlap already
improves over exposed blocking communication, and the openmp communication
thread improves further by avoiding pinned host staging.  the final clean
package sweep, job `85271`, measured `2.269m tok/s` mean for openmp-thread
overlap versus `2.057m tok/s` for pinned overlap and `1.516m tok/s` for direct
blocking.  this is a `1.103x` improvement over pinned overlap and a `1.496x`
improvement over direct blocking.

h512 is the main limitation case.  host-staged nonblocking overlap becomes too
expensive because gradients are larger, and higher learning-rate h512 runs
showed occasional validity issues.  in the final clean short bucket check at
`lr=5e-5`, job `85268`, the openmp-thread path was valid in the tested cases
and reached `0.879m tok/s` at a `2048 kb` bucket.  this is useful supporting
evidence, but h256 remains the headline distributed result because it has the
cleaner repeated benchmark.

profiling supports these conclusions.  the h256 nsight systems profile shows
that the openmp worker spends time in `openmp_comm_mpi_allreduce`, while event
wait time is small, meaning the worker usually consumes buckets after compute
has made them ready.  nsight compute and roofline analysis show that fusion can
reduce memory traffic, but not every fused kernel reaches the memory roof.

## 7. discussion of approaches and bottlenecks

the most important lesson is that overlap and fusion are both conditional.
fusion helps when it removes meaningful global-memory traffic from a kernel
that is actually limited by memory movement.  in the final clean fusion
profile, bias plus relu, residual plus layernorm, and layernorm backward plus
residual all improve in event timing.  residual plus layernorm remains the
cleanest report example because it combines a clear traffic reduction with a
simple transformer operation.

communication overlap also has a sweet spot.  h128 is too small for much
communication hiding, h256 has enough communication to benefit and enough
remaining backward computation to hide it, and h512 exposes staging and
validity limits.  the openmp communication thread is useful because it gives a
direct-device communication path without relying on the broken nonblocking
device-buffer mpi path on this cluster.

the biggest future improvement would be a production-quality gpu collective
path such as nccl or a stable cuda-aware nonblocking mpi stack.  true mixed
precision is another future direction.  the ffn fp16 experiment passes focused
reference checks and can be faster on valid h512 runs, but the final repeated
h512 training sweep was not fully valid, so fp16 remains experimental rather
than a shipped default.

## 8. challenges and next steps

the hardest distributed challenge was not the allreduce call itself; it was
making measurements trustworthy.  several early results were polluted by
environment variables, learning-rate instability, cold-start outliers, or
script defaults.  the final workflow therefore uses clean environments,
exclusive allocations where possible, repeated runs, warmup dropping, validity
counts, and a manifest that ties results to job ids.

the final clean package now contains a master table,
`results/final_clean_main_table.csv`, plus copied artifacts under
`artifacts/`.  the remaining work is report writing: use the locked table,
the validation job, the nsight summaries, and the regenerated roofline plots
without adding new major features.
