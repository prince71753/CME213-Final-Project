# Correctness matrix (85251)

Host: hpcc-gpu-5-2
Date: 2026-05-17T05:19:18-07:00

- make tests mpi: PASS
  make: Nothing to be done for 'tests'.
  make: Nothing to be done for 'mpi'.

- test_gemm: PASS
    naive:  0.200 ms (1344.4 GFLOP/s)
    tiled:  0.022 ms (12093.4 GFLOP/s)  speedup: 9.00x
  

- test_attention: PASS
    tiled  max error: 7.450581e-08  PASS
    naive:  0.0407 ms
    tiled:  0.1260 ms  speedup: 0.32x

- test_layernorm: PASS
  rows=2048, cols=128
    max error: 7.152557e-07  PASS
    time: 0.0070 ms  effective bandwidth: 301.2 GB/s

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

- single_gpu_smoke default_auto: PASS
  Batches: total=536 local_per_rank=536 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1608
  Epoch 1: avg_logged_loss=4.1608  steps/rank=5  25ms  404314 tok/s

- single_gpu_smoke cublas_tc: PASS
  Batches: total=536 local_per_rank=536 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1608
  Epoch 1: avg_logged_loss=4.1608  steps/rank=5  26ms  397845 tok/s

- train_validate_config: PASS
    (none set)
    GEMM requested=auto auto_policy=auto:cublas_tc_current_clean_benchmark
  Configuration validation: PASS

- mpi_4_blocking_direct_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  44ms  927042 tok/s  avg_grad_sync=2.079ms  checksum_span=0.000e+00

- mpi_4_overlap_pinned_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  44ms  926723 tok/s  avg_grad_start=0.706ms  avg_grad_finish=1.250ms  checksum_span=0.000e+00

## Result
PASS
