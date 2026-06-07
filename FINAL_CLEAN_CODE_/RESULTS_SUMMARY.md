# Test & Results Summary

## Correctness Tests

| Test | File | Purpose | Key Results |
|------|------|---------|------------|
| Attention | `tests/test_attention.cu` | Validates attention kernel against CPU reference | Ensures forward pass correctness |
| GEMM | `tests/test_gemm.cu` | Tests all GEMM backends with correctness & timing | Validates cuBLAS, tensor cores, custom kernels |
| LayerNorm | `tests/test_layernorm.cu` | LayerNorm kernel validation | Numerical correctness checks |
| Model Reference | `tests/test_model_reference.cu` | End-to-end model, gradients, Adam updates | Full training pipeline validation |

**Validation Scripts:**
- `scripts/run_correctness_matrix.sh` – Quick correctness smoke tests
- `scripts/run_full_validation.sh` – Comprehensive validation suite

**Log Location:** `logs/correctness_matrix_*.out`

---

## Benchmark Tests

| Test | File | Purpose | Metrics |
|------|------|---------|---------|
| All Benchmarks | `tests/benchmark_all.cu` | Full training pipeline performance | Step times, throughput |
| Gradient Sync | `tests/benchmark_gradient_sync.cu` | All-reduce and synchronization performance | Communication latency/bandwidth |
| AllReduce Alpha/Beta | `tests/benchmark_allreduce_alpha_beta.cu` | Explore alpha/beta tuning for MPI collectives | Timing optimization |
| Fusion Benchmark | `tests/test_fusion_benchmark.cu` | Kernel fusion performance | Speedup vs non-fused |

**Raw Data:**
- `results/allreduce_alpha_beta_88903.csv` – 2,801 measurements
  - Columns: backend, ranks, bytes, count, iteration, time_ms
  - Backends tested: host_pinned, device, cuda_ipc
  - Ranks: 2, 4, 8+
  - Message sizes: 4KB to multi-MB

**Log Location:** 
- `logs/allreduce_alpha_beta_88903.out` (processed)
- `logs/allreduce_alpha_beta_88903_raw.txt` (raw output)
- `logs/comm_thread_sweep_*.out` (communication thread variants)

---

## Profiling & Analysis

| Test | File | Purpose | Output |
|------|------|---------|--------|
| Fusion Kernels | `tests/profile_fusion_kernels.cu` | Analyze fused kernel overhead | Kernel timing breakdown |
| Training Hotspots | `tests/profile_training_hotspots.cu` | Identify bottlenecks in training loop | Time per component |
| Deep Profile | `tests/deep_profile.cu` | Comprehensive Nsight profiling | Memory, compute utilization |
| MPI Thread Probe | `tests/mpi_thread_multiple_probe.cpp` | Test MPI threading levels | Thread safety, overhead |

**Profile Location:** `profiles/` – Nsight summaries and exported reports

---

## Configuration & Backend Testing

| Configuration | Test | Validation |
|---------------|------|-----------|
| Single GPU (default) | `cublas_tc` backend | `correctness_matrix_*_single_gpu_smoke_cublas_tc.txt` |
| Single GPU (auto) | Auto backend selection | `correctness_matrix_*_single_gpu_smoke_default_auto.txt` |
| MPI 4-rank (blocking) | Direct synchronization | `correctness_matrix_*_mpi_4_blocking_direct_smoke.txt` |
| MPI 4-rank (overlap) | Overlapped communication | `correctness_matrix_*_mpi_4_overlap_pinned_smoke.txt` |
| Attention Module | Standalone validation | `correctness_matrix_*_test_attention.txt` |

---

## Key Results & Recommendations

### Default Configuration
- **Single GPU:** `cublas_tc` (tensor core GEMM)
- **Multi-GPU:** h256 OpenMP communication thread (overlaps communication with backward pass)

### Performance Insights
- **h256 Communication Thread:** Best when sufficient backward work hides communication latency
- **h512 FP16 FFN:** Opt-in research path—passed focused checks but not fully validated in repeated h512 training

### Build & Execution
```bash
# Build
make -j8 && make -j8 tests && make -j8 mpi

# Single GPU Training
./build/train inp.txt --epochs 1 --max-steps 100

# MPI 4-GPU with Communication Thread
CME213_COMM_THREAD=1 mpirun -np 4 ./build_mpi/train_mpi inp.txt --epochs 1 --max-steps 100 --sync-mode overlap --bucket-kb 1024

# H512 FP16 Experimental
CME213_FFN_FP16=1 ./build/train inp.txt --epochs 1 --max-steps 50 --lr 1e-4
```

---

## Results File Locations

| Category | Location | Notes |
|----------|----------|-------|
| CSV Data | `results/allreduce_alpha_beta_88903.csv` | Primary benchmark data |
| Validation Logs | `logs/correctness_matrix_*.out` | Build & correctness results |
| Benchmark Logs | `logs/comm_thread_sweep_*.out` | Communication thread sweeps |
| Profiles | `profiles/` | Nsight reports & summaries |

---

## Submission Notes

For writeup, include:
1. **Correctness validation**: Reference the 4 core correctness tests
2. **Performance data**: AllReduce alpha/beta CSV with backend comparison
3. **Configuration results**: Results table with backend × configuration matrix
4. **Key finding**: h256 communication thread superiority for distributed training
5. **Limitations**: h512 FP16 as experimental/future work
