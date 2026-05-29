# Correctness matrix (87896)

Host: hpcc-gpu-5-1
Date: 2026-05-28T11:02:52-07:00

- make all tests mpi: PASS
  make: Nothing to be done for 'all'.
  make: Nothing to be done for 'tests'.
  make: Nothing to be done for 'mpi'.

- test_gemm: PASS
    naive:  0.201 ms (1333.0 GFLOP/s)
    tiled:  0.022 ms (12014.6 GFLOP/s)  speedup: 9.01x
  

- test_attention: PASS
    tiled  max error: 7.450581e-08  PASS
    naive:  0.0410 ms
    tiled:  0.1269 ms  speedup: 0.32x

- test_layernorm: PASS
  rows=2048, cols=128
    max error: 7.152557e-07  PASS
    time: 0.0070 ms  effective bandwidth: 298.8 GB/s

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
  Epoch 1: avg_logged_loss=4.1608  steps/rank=5  26ms  388455 tok/s

- single_gpu_smoke cublas_tc: PASS
  Batches: total=536 local_per_rank=536 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1608
  Epoch 1: avg_logged_loss=4.1608  steps/rank=5  27ms  378903 tok/s

- train_validate_config: PASS
    (none set)
    GEMM requested=auto auto_policy=auto:cublas_tc_current_clean_benchmark
  Configuration validation: PASS

- mpi_4_blocking_direct_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  48ms  855913 tok/s  avg_grad_sync=1.942ms  checksum_span=0.000e+00

- mpi_4_overlap_pinned_smoke: PASS
  Batches: total=536 local_per_rank=134 used_per_epoch=5 dropped=0
    epoch 1 | step    5/5 | loss 4.1611
  Epoch 1: avg_logged_loss=4.1611  steps/rank=5  47ms  869888 tok/s  avg_grad_start=0.736ms  avg_grad_finish=1.270ms  checksum_span=0.000e+00

## Result
PASS
