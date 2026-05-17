# Correctness matrix (85263)

Host: hpcc-gpu-5-2
Date: 2026-05-17T12:54:46-07:00

- make tests mpi: PASS
  mpicxx -o build_mpi/train_mpi build_mpi/kernels/gemm.o build_mpi/kernels/attention.o build_mpi/kernels/layernorm.o build_mpi/kernels/activations.o build_mpi/kernels/embedding.o build_mpi/model/transformer.o build_mpi/main.o build_mpi/data/dataset.o build_mpi/model/distributed.o -pthread -L/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/lib -Wl,-rpath -Wl,/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/lib -Wl,--enable-new-dtags -lmpi -L/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/REDIST/cuda/12.3/targets/x86_64-linux/lib -lcudart \
      -L/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib -Wl,-rpath,/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib -lcublas -lcublasLt \
      -L/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/comm_libs/12.3/nccl/lib -Xlinker -rpath -Xlinker /home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1/comm_libs/12.3/nccl/lib -lnccl  -fopenmp

- test_gemm: PASS
    naive:  0.199 ms (1345.7 GFLOP/s)
    tiled:  0.022 ms (11970.0 GFLOP/s)  speedup: 8.89x
  

- test_attention: PASS
    tiled  max error: 7.450581e-08  PASS
    naive:  0.0407 ms
    tiled:  0.1262 ms  speedup: 0.32x

- test_layernorm: PASS
  rows=2048, cols=128
    max error: 7.152557e-07  PASS
    time: 0.0070 ms  effective bandwidth: 300.3 GB/s

- test_model_reference custom: PASS
    adam first-step clip_scale=1.000000 max_update_err=1.812e-09 PASS
  
  === MODEL REFERENCE TEST PASSED ===

- test_model_reference cublas: PASS
    adam first-step clip_scale=1.000000 max_update_err=1.812e-09 PASS
  
  === MODEL REFERENCE TEST PASSED ===

- test_model_reference cublas_tc: PASS
    adam first-step clip_scale=1.000000 max_update_err=1.812e-09 PASS
  
  === MODEL REFERENCE TEST PASSED ===

- test_model_reference cublas_tc_lt: PASS
    adam first-step clip_scale=1.000000 max_update_err=1.812e-09 PASS
  
  === MODEL REFERENCE TEST PASSED ===

- single_gpu_smoke default_auto: FAIL (exit 127)
  env: ‘./build/train’: No such file or directory

- single_gpu_smoke cublas_tc: FAIL (exit 127)
  env: ‘./build/train’: No such file or directory

- train_validate_config: FAIL (exit 127)
  env: ‘./build/train’: No such file or directory

- mpi_4_blocking_direct_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  45ms  910586 tok/s  avg_grad_sync=1.679ms  checksum_span=0.000e+00

- mpi_4_overlap_pinned_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  45ms  909330 tok/s  avg_grad_start=0.048ms  avg_grad_finish=2.085ms  checksum_span=0.000e+00

## Result
FAIL
