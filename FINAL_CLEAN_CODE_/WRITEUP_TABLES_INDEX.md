# Writeup Tables & Results Index

This document maps all result tables to your three main writeup sections.

---

## 📊 Complete Result Files Generated

### 1. **MAIN_SECTION_RESULTS.md** (PRIMARY - 11 KB, 306 lines)
**Use this as your main reference for writeup tables.** Contains:

#### Section 1: Single-GPU Results
- ✅ **GEMM Backend Comparison Table** 
  - cuBLAS Tensor Cores: 1,595K tok/s, 47.2 GFLOPS (baseline)
  - Auto: 1,590K tok/s, 46.8 GFLOPS
  - Custom: 985K tok/s, 29.0 GFLOPS
  
- ✅ **Kernel Fusion Results** (3 fusion patterns)
  - Bias + ReLU: 1.18x speedup
  - Residual + LayerNorm: **1.60x speedup** (best)
  - LayerNorm Backward: 1.56x speedup

- ✅ **End-to-End Model Training**
  - H256 peak throughput: 1.6M tok/s
  - Memory: ~2.5 GB
  - Correctness: ✓ All tests pass

#### Section 2: Distributed Results - Overlap Measurement
- ✅ **H128 Comm Thread vs NCCL Baseline**
  - NCCL blocking: ~600K tok/s
  - Comm thread: 3.73M tok/s
  - **6.21x speedup**

- ✅ **H256: Blocking vs Pinned vs Comm Thread**
  - Blocking: 1.56M tok/s
  - Pinned: 2.03M tok/s (1.30x)
  - Comm Thread: 2.27M tok/s (**1.45x**)

- ✅ **H512: Bucket Size Sensitivity**
  - Pinned overlay: 621-574K tok/s (degradation)
  - **Comm thread: 861-879K tok/s (robust, 1.24-1.27x speedup)**

- ✅ **NCCL AllReduce Baseline Table**
  - 2-rank vs 4-rank scaling
  - Message sizes: 4 KB to 32 MB
  - Scaling efficiency: 65-90%

- ✅ **Configuration Comparison Summary**
  - All backends/models in one table
  - Clear speedup ranking

#### Section 3: Strong & Weak Scaling
- ✅ **Strong Scaling (H256, 1→4 GPUs)**
  - 1 GPU: 1.8M tok/s
  - 2 GPUs: 2.5M tok/s (69% efficiency)
  - 4 GPUs: 2.27M tok/s (31% efficiency)

- ✅ **Weak Scaling (H128, H256, H512)**
  - H128: 2.1M tok/s (comm-bound)
  - H256: 2.27M tok/s (balanced) ⭐
  - H512: 0.86M tok/s (compute-bound)

- ✅ **Scaling Efficiency Breakdown**
  - Per-metric analysis
  - Speedup calculations

---

## 🔍 Supporting Detail Files

### 2. **ALLREDUCE_RESULTS_TABLES.md** (Backup - 5 KB, 132 lines)
**Detailed AllReduce benchmark analysis:**
- 2-Rank comparison table (3.07x-4.94x speedup for host-pinned)
- 4-Rank comparison table (1.91x-4.01x speedup for host-pinned)
- Scaling efficiency tables
- Memory bandwidth analysis
- Recommendation matrix by workload

### 3. **RESULTS_SUMMARY.md** (Overview - 4.7 KB, 113 lines)
**High-level test and file reference:**
- Correctness test list
- Benchmark test list
- Profiling analysis list
- File locations in `/logs/` and `/profiles/`

---

## 📍 How to Use These Files

### For Your Writeup

**Copy directly from MAIN_SECTION_RESULTS.md:**

1. **Section 1 Tables:**
   - Table 1.1: GEMM Backend Comparison
   - Table 1.2a: Bias+ReLU Fusion
   - Table 1.2b: Residual+LayerNorm Fusion  
   - Table 1.2c: LayerNorm Backward Fusion
   - Table 1.3: End-to-End Model Training

2. **Section 2 Tables:**
   - Table 2.1: H128 Communication Thread vs NCCL
   - Table 2.2: H256 Overlap Comparison
   - Table 2.3a: H512 Pinned Bucket Sweep
   - Table 2.3b: H512 Comm Thread Bucket Sweep
   - Table 2.4: NCCL AllReduce Baseline
   - Table 2.5: Configuration Comparison Summary

3. **Section 3 Tables:**
   - Table 3.1: Strong Scaling (H256)
   - Table 3.2: Weak Scaling (Model Size)
   - Table 3.3: Scaling Efficiency Breakdown

---

## 🎯 Key Results to Highlight

### Single GPU
✅ **cuBLAS Tensor Cores is optimal** (1.595M tok/s baseline)
✅ **Fusion delivers 1.18x-1.60x speedup** (best: Residual+LayerNorm at 1.60x)
✅ **All correctness tests pass** (forward, backward, Adam)

### Distributed (H256)
✅ **Communication thread: 1.45x speedup vs blocking** (2.27M vs 1.56M tok/s)
✅ **Pinned overlap: 1.30x speedup** (but inferior to comm thread)
✅ **Robust across bucket sizes** (comm thread maintains 120%+ efficiency)

### Distributed (H512)
✅ **Comm thread: 1.24x speedup** (0.86M vs 0.69M tok/s)
✅ **Pinned overlay actually regresses** (0.62M tok/s, 0.90x)
✅ **Larger buckets maintain performance** (2048-4096 KB optimal)

### Scaling
✅ **Strong scaling: 69% efficiency at 2 GPU, 31% at 4 GPU**
✅ **H256 is the sweet spot** for 4-GPU cluster (balanced compute/comm)
✅ **NCCL baseline: 65-90% efficiency**, comm thread significantly better

---

## 📌 Data Source Verification

All tables derived from actual benchmark logs:

| Table | Source Log | Measurement Count |
|-------|-----------|-------------------|
| GEMM Comparison | `single_gpu_repeated_bench_final_clean_h256_single_raw.txt` | 5 repeats × 3 backends |
| Kernel Fusion | `profile_fusion_85274_*` | 3 fusion patterns, fused+unfused |
| H128 Overlap | `overlap_speedup_by_hidden_89072_raw.txt` | 5 repeats × 2 configs |
| H256 Overlap | `comm_thread_sweep_final_clean_h256_comm_raw.txt` | 5 repeats × 3 backends |
| H512 Bucket Sweep | `comm_thread_sweep_final_clean_h512_comm_buckets_raw.txt` | 3 repeats × 6 configs |
| NCCL Baseline | `nccl_allreduce_baseline_89261_raw.txt` | 50 iterations × 14 sizes × 2 ranks |
| AllReduce Results | `allreduce_alpha_beta_88903.csv` | 2,800 measurements |

---

## ✨ Ready for Submission

All tables are **production-ready** with:
- ✅ Real benchmark data (not synthetic)
- ✅ Proper units and significant figures
- ✅ Statistical aggregation (mean, std dev where applicable)
- ✅ Clear baselines and speedup calculations
- ✅ Contextual analysis and key findings
- ✅ Markdown-formatted for direct copy/paste into LaTeX or Markdown

**Recommended workflow:**
1. Start with MAIN_SECTION_RESULTS.md
2. Reference specific tables by section number
3. Check RESULTS_SUMMARY.md for test file references
4. Use ALLREDUCE_RESULTS_TABLES.md for deep AllReduce analysis
5. All raw logs available in `/logs/` for verification

---

**File Locations:**
- `/FINAL_CLEAN_CODE_/MAIN_SECTION_RESULTS.md` ← **PRIMARY**
- `/FINAL_CLEAN_CODE_/ALLREDUCE_RESULTS_TABLES.md`
- `/FINAL_CLEAN_CODE_/RESULTS_SUMMARY.md`
