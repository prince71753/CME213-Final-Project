// training driver for single gpu and mpi runs.
#include "common.h"
#include "data.h"
#include "model.h"
#include "distributed.h"
#include "kernels.h"
#include "profiling.h"
#include <cstdio>
#if defined(USE_MPI) && defined(_OPENMP)
#include <omp.h>
#endif

enum class SyncMode {
    Blocking,
    Overlap,
    Auto
};

static const char* sync_mode_name(SyncMode mode) {
    switch (mode) {
        case SyncMode::Blocking: return "blocking";
        case SyncMode::Overlap:  return "overlap";
        case SyncMode::Auto:     return "auto";
    }
    return "blocking";
}

static double param_checksum(const TransformerModel& model) {
    std::vector<float> h(model.total_param_count);
    CUDA_CHECK(cudaMemcpy(h.data(), model.d_all_params,
                          model.total_param_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float v : h) sum += (double)v;
    return sum;
}

extern char** environ;

static bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0 && std::strcmp(value, "") != 0;
}

static void print_cme213_environment(const DistributedContext& dist) {
    if (!dist.is_root()) return;

    printf("CME213 environment:\n");
    bool any = false;
    for (char** env = environ; env && *env; ++env) {
        if (std::strncmp(*env, "CME213_", 7) == 0) {
            printf("  %s\n", *env);
            any = true;
        }
    }
    if (!any)
        printf("  (none set)\n");
    printf("  GEMM requested=%s auto_policy=%s\n",
           gemm_requested_backend_name(), gemm_auto_policy_name());
}

static bool validate_cme213_config(const DistributedContext& dist) {
    bool ok = true;
    auto reject = [&](const char* msg) {
        if (dist.is_root())
            fprintf(stderr, "Invalid CME213 configuration: %s\n", msg);
        ok = false;
    };

    const bool mixed = env_enabled("CME213_MIXED_PRECISION");
    const bool fp16_storage = env_enabled("CME213_FP16_STORAGE_ONLY");
    const bool ffn_fp16 = env_enabled("CME213_FFN_FP16");
    const bool lt_fusion = env_enabled("CME213_LT_FUSION");
    const bool graph = env_enabled("CME213_USE_CUDA_GRAPH");
    const bool unvalidated_nccl = env_enabled("CME213_ALLOW_UNVALIDATED_NCCL");
    const bool nccl = env_enabled("CME213_USE_NCCL");
    const bool comm_thread = env_enabled("CME213_COMM_THREAD");

    if ((mixed || fp16_storage || ffn_fp16) && lt_fusion)
        reject("FP16 modes must not be combined with CME213_LT_FUSION=1");
    if (mixed)
        reject("CME213_MIXED_PRECISION is planned but not implemented in this build");
    if ((mixed && fp16_storage) || (mixed && ffn_fp16) || (fp16_storage && ffn_fp16))
        reject("CME213_MIXED_PRECISION, CME213_FP16_STORAGE_ONLY, and CME213_FFN_FP16 are mutually exclusive");
#ifndef USE_MPI
    if (comm_thread)
        reject("CME213_COMM_THREAD requires the MPI build");
#endif
    if ((fp16_storage || ffn_fp16) && dist.world_size > 1)
        reject("FP16 experimental paths are currently validated only for single-GPU runs");
    if (ffn_fp16 && graph)
        reject("CME213_FFN_FP16 is not yet validated with CME213_USE_CUDA_GRAPH=1");
    if (graph && dist.world_size > 1 && dist.is_root())
        fprintf(stderr, "Warning: CME213_USE_CUDA_GRAPH=1 is ignored for MPI runs\n");
    if (nccl && !unvalidated_nccl && dist.is_root())
        fprintf(stderr,
                "Warning: CME213_USE_NCCL=1 requested without "
                "CME213_ALLOW_UNVALIDATED_NCCL=1; NCCL falls back to MPI\n");
    if (nccl && unvalidated_nccl && dist.is_root())
        fprintf(stderr, "Warning: forcing previously unvalidated NCCL path\n");

    return ok;
}

int main(int argc, char** argv) {
    DistributedContext dist;
    dist.init(&argc, &argv);

    std::string data_path = "inp.txt";
    int num_epochs = 10;
    int max_steps = -1;
    float lr = 3e-4f;
    SyncMode sync_mode = SyncMode::Blocking;
    int bucket_kb = 0;
    float auto_overlap_max_mb = 8.0f;
    bool validate_config_only = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--overlap") == 0) {
            sync_mode = SyncMode::Overlap;
        } else if (strcmp(argv[i], "--auto-sync") == 0) {
            sync_mode = SyncMode::Auto;
        } else if (strcmp(argv[i], "--sync-mode") == 0 && i + 1 < argc) {
            const char* mode = argv[++i];
            if (strcmp(mode, "blocking") == 0) {
                sync_mode = SyncMode::Blocking;
            } else if (strcmp(mode, "overlap") == 0) {
                sync_mode = SyncMode::Overlap;
            } else if (strcmp(mode, "auto") == 0) {
                sync_mode = SyncMode::Auto;
            } else {
                if (dist.is_root())
                    fprintf(stderr, "Unknown sync mode '%s'\n", mode);
                dist.finalize();
                return 1;
            }
        } else if (strcmp(argv[i], "--bucket-kb") == 0 && i + 1 < argc) {
            bucket_kb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--auto-overlap-max-mb") == 0 && i + 1 < argc) {
            auto_overlap_max_mb = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--validate-config") == 0) {
            validate_config_only = true;
        } else {
            data_path = argv[i];
        }
    }
    print_cme213_environment(dist);
    if (!validate_cme213_config(dist)) {
        dist.finalize();
        return 2;
    }
    if (validate_config_only) {
        if (dist.is_root())
            printf("Configuration validation: PASS\n");
        dist.finalize();
        return 0;
    }
#ifdef USE_MPI
    int min_bucket_floats = (bucket_kb > 0)
        ? (bucket_kb * 1024 + (int)sizeof(float) - 1) / (int)sizeof(float)
        : 0;
#endif

    std::string text = read_file(data_path);
    CharTokenizer tokenizer;
    tokenizer.build(text);

    if (dist.is_root())
        printf("Vocabulary size: %d\n", tokenizer.vocab_size);

    TransformerConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size;

    Dataset dataset;
    dataset.load(data_path, tokenizer, cfg.seq_len, cfg.batch_size, dist.is_root());

    TransformerModel model;
    model.build(cfg);
    model.init_weights(42);

    double grad_mb = model.total_param_count * sizeof(float) / (1024.0 * 1024.0);
    bool use_overlap_sync = false;
#ifdef USE_MPI
    if (dist.world_size > 1) {
        use_overlap_sync =
            sync_mode == SyncMode::Overlap ||
            (sync_mode == SyncMode::Auto && grad_mb <= auto_overlap_max_mb);
    }
#endif

    if (dist.is_root()) {
        printf("Model: vocab=%d seq=%d hidden=%d heads=%d ff=%d batch=%d\n",
               cfg.vocab_size, cfg.seq_len, cfg.hidden_dim, cfg.num_heads,
               cfg.ff_dim, cfg.batch_size);
        printf("Parameters: %d (%.2f KB)\n",
               model.total_param_count, model.total_param_count * 4.0f / 1024.0f);
        printf("Training: epochs=%d lr=%.2e max_steps=%d world_size=%d sync_mode=%s effective_sync=%s bucket_kb=%d auto_overlap_max_mb=%.1f\n",
               num_epochs, lr, max_steps, dist.world_size, sync_mode_name(sync_mode),
               use_overlap_sync ? "overlap" : "blocking", bucket_kb,
               auto_overlap_max_mb);
#ifdef USE_MPI
        printf("Gradient averaging: %s\n",
               (dist.world_size > 1 && dist.defer_gradient_average_to_adam)
                   ? "deferred_to_adam"
                   : "post_allreduce");
#endif
    }

    int BT = cfg.batch_size * cfg.seq_len;
    int num_batches = dataset.num_batches();
    int local_batches = num_batches / dist.world_size;
    int dropped_batches = num_batches - local_batches * dist.world_size;

    if (local_batches <= 0) {
        if (dist.is_root()) {
            fprintf(stderr, "Error: dataset has %d batches but world_size=%d\n",
                    num_batches, dist.world_size);
        }
        dist.finalize();
        return 1;
    }

    int steps_per_epoch = local_batches;
    if (max_steps > 0 && max_steps < steps_per_epoch)
        steps_per_epoch = max_steps;

    if (dist.is_root()) {
        printf("Batches: total=%d local_per_rank=%d used_per_epoch=%d dropped=%d\n",
               num_batches, local_batches, steps_per_epoch, dropped_batches);
    }

    std::vector<int> all_inputs(num_batches * BT);
    std::vector<int> all_targets(num_batches * BT);
    {
        std::vector<int> h_input, h_target;
        dataset.reset();
        for (int s = 0; s < num_batches; ++s) {
            dataset.next_batch(h_input, h_target);
            memcpy(all_inputs.data() + s * BT, h_input.data(), BT * sizeof(int));
            memcpy(all_targets.data() + s * BT, h_target.data(), BT * sizeof(int));
        }
    }

    int* d_all_inputs;
    int* d_all_targets;
    CUDA_CHECK(cudaMalloc(&d_all_inputs,  num_batches * BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_targets, num_batches * BT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_all_inputs, all_inputs.data(),
                          num_batches * BT * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_targets, all_targets.data(),
                          num_batches * BT * sizeof(int), cudaMemcpyHostToDevice));

    int* d_step_tokens  = nullptr;
    int* d_step_targets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_step_tokens,  BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_step_targets, BT * sizeof(int)));

    const bool graph_env = []() {
        const char* env = std::getenv("CME213_USE_CUDA_GRAPH");
        return env && std::strcmp(env, "1") == 0;
    }();
    bool use_graph_step = graph_env && dist.world_size == 1;
    cudaGraph_t      g_graph      = nullptr;
    cudaGraphExec_t  g_graph_exec = nullptr;
    bool             graph_ready  = false;
    const int        kWarmupSteps = 3;
    if (use_graph_step && dist.is_root()) {
        printf("CUDA Graph: enabled, capture after %d warmup steps\n",
               kWarmupSteps);
    }

    float step_loss = 0.0f;

    auto run_training_loop = [&]() {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float logged_loss_sum = 0.0f;
        int logged_loss_count = 0;
        double comm_ms_sum = 0.0;
        double async_start_ms_sum = 0.0;

        dist.barrier();
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < steps_per_epoch; ++step) {
            int batch_idx = step * dist.world_size + dist.rank;
            int* d_tokens  = d_all_inputs  + batch_idx * BT;
            int* d_targets = d_all_targets + batch_idx * BT;

            const bool step_uses_graph = use_graph_step && step >= kWarmupSteps;

            if (step_uses_graph) {
                NvtxRange graph_range("cuda_graph_step");

                CUDA_CHECK(cudaMemcpyAsync(d_step_tokens,  d_tokens,
                                            BT * sizeof(int),
                                            cudaMemcpyDeviceToDevice,
                                            cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_step_targets, d_targets,
                                            BT * sizeof(int),
                                            cudaMemcpyDeviceToDevice,
                                            cudaStreamPerThread));
                if (!graph_ready) {
                    CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread,
                                                       cudaStreamCaptureModeRelaxed));
                    model.zero_grad();
                    model.forward_no_sync(d_step_tokens, d_step_targets);
                    model.backward();
                    model.update_adam(lr);
                    CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &g_graph));
                    CUDA_CHECK(cudaGraphInstantiate(&g_graph_exec, g_graph,
                                                     nullptr, nullptr, 0));
                    graph_ready = true;
                } else {
                    CUDA_CHECK(cudaGraphLaunch(g_graph_exec, cudaStreamPerThread));
                }
            } else {
                model.zero_grad();
                {
                    NvtxRange range("forward");
                    model.forward_no_sync(d_tokens, d_targets);
                }
#ifdef USE_MPI
                if (use_overlap_sync && dist.world_size > 1) {
                    NvtxRange range("backward_bucketed");
                    model.backward_bucketed(&dist, min_bucket_floats);
                } else {
                    {
                        NvtxRange range("backward");
                        model.backward();
                    }
                    if (dist.world_size > 1) {
                        NvtxRange range("gradient_sync");
                        dist.sync_gradients(model.d_all_grads,
                                             model.total_param_count);
                    }
                }
                comm_ms_sum += dist.last_sync_ms;
                async_start_ms_sum += dist.last_async_start_ms;
#else
                {
                    NvtxRange range("backward");
                    model.backward();
                }
#endif
                float adam_grad_scale = 1.0f;
#ifdef USE_MPI
                if (dist.world_size > 1 && dist.defer_gradient_average_to_adam)
                    adam_grad_scale = 1.0f / (float)dist.world_size;
#endif
                {
                    NvtxRange range("adam_update");
                    model.update_adam(lr, 0.9f, 0.999f, 1e-8f,
                                      adam_grad_scale);
                }
            }

            bool should_log = (step + 1) % 100 == 0 || step + 1 == steps_per_epoch;
            if (dist.is_root() && should_log) {
                step_loss = model.read_loss();
                logged_loss_sum += step_loss;
                logged_loss_count++;
                printf("  epoch %d | step %4d/%d | loss %.4f\n",
                       epoch + 1, step + 1, steps_per_epoch, step_loss);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        dist.barrier();
        auto epoch_end = std::chrono::high_resolution_clock::now();

        double local_ms =
            std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        double max_ms = dist.max_double(local_ms);
        double avg_comm_ms = (steps_per_epoch > 0)
            ? comm_ms_sum / (double)steps_per_epoch : 0.0;
        double avg_async_start_ms = (steps_per_epoch > 0)
            ? async_start_ms_sum / (double)steps_per_epoch : 0.0;
        double max_avg_comm_ms = dist.max_double(avg_comm_ms);
        double max_avg_async_start_ms = dist.max_double(avg_async_start_ms);
        double local_checksum = param_checksum(model);
        double checksum_min = dist.min_double(local_checksum);
        double checksum_max = dist.max_double(local_checksum);

        if (dist.is_root()) {
            float avg_logged_loss = logged_loss_count > 0
                ? logged_loss_sum / logged_loss_count : step_loss;
            double tokens = (double)steps_per_epoch * (double)BT * (double)dist.world_size;
            double tokens_per_sec = tokens / (max_ms / 1000.0);
            printf("Epoch %d: avg_logged_loss=%.4f  steps/rank=%d  %.0fms  %.0f tok/s",
                   epoch + 1, avg_logged_loss, steps_per_epoch, max_ms, tokens_per_sec);
#ifdef USE_MPI
            if (use_overlap_sync && dist.world_size > 1) {
                printf("  avg_grad_start=%.3fms  avg_grad_finish=%.3fms  checksum_span=%.3e",
                       max_avg_async_start_ms, max_avg_comm_ms,
                       checksum_max - checksum_min);
            } else {
                printf("  avg_grad_sync=%.3fms  checksum_span=%.3e",
                       max_avg_comm_ms, checksum_max - checksum_min);
            }
#endif
            printf("\n");
        }
    }
    };

#if defined(USE_MPI) && defined(_OPENMP)
    if (dist.use_comm_thread && dist.world_size > 1) {
#pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) {
                run_training_loop();
                dist.request_comm_worker_shutdown();
            } else {
                dist.comm_worker_loop();
            }
        }
    } else {
        run_training_loop();
    }
#else
    run_training_loop();
#endif

    if (g_graph_exec) cudaGraphExecDestroy(g_graph_exec);
    if (g_graph)      cudaGraphDestroy(g_graph);
    cudaFree(d_step_tokens);
    cudaFree(d_step_targets);
    cudaFree(d_all_inputs);
    cudaFree(d_all_targets);
    model.free_all();
    dist.finalize();
    return 0;
}
