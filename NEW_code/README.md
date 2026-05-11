# Milestone 3 Code

This folder is a trimmed single-GPU version of the final project code. It keeps
the CUDA kernels, the C++/CUDA training driver, the dataset loader, and the
correctness/performance tests needed for Milestone 3. 

Build on a GPU node after loading the course CUDA module:

```bash
module load course/cme213/nvhpc/24.1
make all tests
```

Useful runs:

```bash
./build/test_gemm
./build/test_attention
./build/test_layernorm
./build/test_model_reference
./build/test_fusion_benchmark
./build/train inp.txt --epochs 1 --max-steps 100
```
