// Single-GPU CUDA code for the mini-Transformer.
#include "common.h"
#include "data.h"
#include "model.h"
#include <cstdio>

extern bool use_fused;
double results[2] = {0.0, 0.0};  // [0]=unfused, [1]=fused
double tokens_per_sec = 0.0;

int main(int argc, char** argv) {
    std::string data_path = "inp.txt";
    int num_epochs = 10;
    int max_steps = -1;
    float lr = 3e-4f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else {
            data_path = argv[i];
        }
    }

    std::string text = read_file(data_path);
    CharTokenizer tokenizer;
    tokenizer.build(text);

    TransformerConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size;

    Dataset dataset;
    dataset.load(data_path, tokenizer, cfg.seq_len, cfg.batch_size, false);

    TransformerModel model;
    model.build(cfg);
    model.init_weights(42);

    int BT = cfg.batch_size * cfg.seq_len;
    int num_batches = dataset.num_batches();
    int steps_per_epoch = num_batches;
    if (max_steps > 0 && max_steps < steps_per_epoch) {
        steps_per_epoch = max_steps;
    }

    if (steps_per_epoch <= 0) {
        fprintf(stderr, "not enough data for one batch\n");
        return 1;
    }
    auto tokens = tokenizer.encode(text);
    printf("vocab=%d tokens=%zu\n", tokenizer.vocab_size, tokens.size());
    printf("model: seq=%d hidden=%d heads=%d ff=%d batch=%d params=%d\n",
           cfg.seq_len, cfg.hidden_dim, cfg.num_heads, cfg.ff_dim,
           cfg.batch_size, model.total_param_count);
    printf("run: epochs=%d steps=%d lr=%.1e\n", num_epochs, steps_per_epoch, lr);

    std::vector<int> all_inputs(num_batches * BT);
    std::vector<int> all_targets(num_batches * BT);
    std::vector<int> h_input;
    std::vector<int> h_target;
    dataset.reset();
    for (int s = 0; s < num_batches; ++s) {
        dataset.next_batch(h_input, h_target);
        memcpy(all_inputs.data() + s * BT, h_input.data(), BT * sizeof(int));
        memcpy(all_targets.data() + s * BT, h_target.data(), BT * sizeof(int));
    }

    int* d_all_inputs = nullptr;
    int* d_all_targets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_all_inputs, num_batches * BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_targets, num_batches * BT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_all_inputs, all_inputs.data(),
                          num_batches * BT * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_targets, all_targets.data(),num_batches * BT * sizeof(int), cudaMemcpyHostToDevice));

    for (int mode = 0; mode < 2; ++mode) {
        use_fused = (mode == 1);
        printf("\n===== RUN: %s =====\n", use_fused ? "FUSED" : "UNFUSED");
        model.free_all();
        model.build(cfg);
        model.init_weights(42);
        // warmup loop for better timing
        for (int i = 0; i < 5; ++i) {
            int* d_tokens = d_all_inputs;
            int* d_targets = d_all_targets;

            model.zero_grad();
            model.forward_no_sync(d_tokens, d_targets);
            model.backward();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float loss = 0.0f;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int step = 0; step < steps_per_epoch; ++step) {
                int* d_tokens = d_all_inputs + step * BT;
                int* d_targets = d_all_targets + step * BT;

                model.zero_grad();
                model.forward_no_sync(d_tokens, d_targets);
                model.backward();
                model.update_adam(lr);

                if ((step + 1) % 100 == 0 || step + 1 == steps_per_epoch) {
                    loss = model.read_loss();
                    printf("step %d/%d loss %.4f\n", step + 1, steps_per_epoch, loss);
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            tokens_per_sec = (double)steps_per_epoch * (double)BT / (ms / 1000.0);
            
            printf("epoch %d: %.0f ms, %.0f tok/s, loss %.4f\n",
                epoch + 1, ms, tokens_per_sec, loss);
        }
        results[mode] = tokens_per_sec;
        printf("RESULT (%s): %.0f tok/s\n", use_fused ? "FUSED" : "UNFUSED", tokens_per_sec);
}

    printf("\n========== FINAL COMPARISON ==========\n");
    printf("UNFUSED: %.0f tok/s\n", results[0]);
    printf("FUSED:   %.0f tok/s\n", results[1]);
    printf("SPEEDUP: %.2fx\n", results[1] / results[0]);

    cudaFree(d_all_inputs);
    cudaFree(d_all_targets);
    model.free_all();
    return 0;
}
