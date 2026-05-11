// Single-GPU CUDA code for the mini-Transformer.
#include "common.h"
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>

static float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static void embedding_forward_cpu(const float* table, const int* tokens,
                                  float* out, int batch_seq, int dim) {
    for (int i = 0; i < batch_seq; ++i)
        for (int j = 0; j < dim; ++j)
            out[i * dim + j] = table[tokens[i] * dim + j];
}

static void embedding_backward_cpu(const float* grad_out, const int* tokens,
                                   float* grad_table, int batch_seq, int dim,
                                   int vocab_size) {
    std::fill(grad_table, grad_table + vocab_size * dim, 0.0f);
    for (int i = 0; i < batch_seq; ++i) {
        int tok = tokens[i];
        for (int j = 0; j < dim; ++j)
            grad_table[tok * dim + j] += grad_out[i * dim + j];
    }
}

static void pos_embedding_cpu(const float* pos_table, float* embeddings,
                              int batch, int seq, int dim) {
    for (int b = 0; b < batch; ++b)
        for (int s = 0; s < seq; ++s)
            for (int d = 0; d < dim; ++d)
                embeddings[(b * seq + s) * dim + d] +=
                    pos_table[s * dim + d];
}

int main() {
    printf("=== Embedding Correctness & Performance Test ===\n\n");

    int vocab_size = 512, embed_dim = 128;
    int batch = 16, seq_len = 32;
    int batch_seq = batch * seq_len;
    printf("vocab_size=%d, embed_dim=%d, batch=%d, seq=%d\n",
           vocab_size, embed_dim, batch, seq_len);

    std::vector<float> h_table(vocab_size * embed_dim);
    std::vector<int> h_tokens(batch_seq);
    std::vector<float> h_out_ref(batch_seq * embed_dim);
    std::vector<float> h_out_gpu(batch_seq * embed_dim);

    srand(42);
    for (auto& v : h_table) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& t : h_tokens) t = rand() % vocab_size;

    // CPU reference
    embedding_forward_cpu(h_table.data(), h_tokens.data(),
                          h_out_ref.data(), batch_seq, embed_dim);

    // GPU computation
    float *d_table, *d_out;
    int *d_tokens;
    CUDA_CHECK(cudaMalloc(&d_table, vocab_size * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, batch_seq * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tokens, batch_seq * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), vocab_size * embed_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), batch_seq * sizeof(int),
                          cudaMemcpyHostToDevice));

    embedding_forward(d_table, d_tokens, d_out, batch_seq, embed_dim);

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, batch_seq * embed_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float err = max_abs_diff(h_out_ref.data(), h_out_gpu.data(), batch_seq * embed_dim);
    printf("  embedding forward: vocab=%d dim=%d batch_seq=%d err=%.3e %s\n",
           vocab_size, embed_dim, batch_seq, err, err < 1e-5f ? "PASS" : "FAIL");

    // Performance
    GpuTimer timer;
    int warmup = 5, iters = 100;
    for (int i = 0; i < warmup; ++i)
        embedding_forward(d_table, d_tokens, d_out, batch_seq, embed_dim);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i)
        embedding_forward(d_table, d_tokens, d_out, batch_seq, embed_dim);
    timer.toc();
    float ms = timer.elapsed_ms() / iters;
    float bytes_read = batch_seq * embed_dim * sizeof(float) + batch_seq * sizeof(int) +   batch_seq * embed_dim * sizeof(float);   
    float bw = bytes_read / (ms / 1e3f) / 1e9f;
    printf("  time: %.4f ms  bandwidth: %.1f GB/s\n", ms, bw);

    // Test embedding backward
    printf("\n=== Embedding Backward Test ===\n");
    std::vector<float> h_grad_out(batch_seq * embed_dim);
    std::vector<float> h_grad_ref(vocab_size * embed_dim);
    std::vector<float> h_grad_gpu(vocab_size * embed_dim);

    for (auto& v : h_grad_out) v = (float)rand() / RAND_MAX - 0.5f;

    embedding_backward_cpu(h_grad_out.data(), h_tokens.data(),
                          h_grad_ref.data(), batch_seq, embed_dim, vocab_size);

    float *d_grad_out, *d_grad_table;
    CUDA_CHECK(cudaMalloc(&d_grad_out, batch_seq * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_table, vocab_size * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_table, 0, vocab_size * embed_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out.data(),
                          batch_seq * embed_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    embedding_backward(d_grad_out, d_tokens, d_grad_table, batch_seq, embed_dim);

    CUDA_CHECK(cudaMemcpy(h_grad_gpu.data(), d_grad_table,
                          vocab_size * embed_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float err_bwd = max_abs_diff(h_grad_ref.data(), h_grad_gpu.data(),
                                 vocab_size * embed_dim);
    printf("  embedding backward: err=%.3e %s\n",
           err_bwd, err_bwd < 1e-5f ? "PASS" : "FAIL");

    // Performance
    for (int i = 0; i < warmup; ++i)
        embedding_backward(d_grad_out, d_tokens, d_grad_table, batch_seq, embed_dim);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i)
        embedding_backward(d_grad_out, d_tokens, d_grad_table, batch_seq, embed_dim);
    timer.toc();
    float ms_bwd = timer.elapsed_ms() / iters;
    printf("  time: %.4f ms\n", ms_bwd);

    // Test positional embedding
    printf("\n=== Positional Embedding Test ===\n");
    std::vector<float> h_pos_table(seq_len * embed_dim);
    std::vector<float> h_emb_ref(batch_seq * embed_dim);
    std::vector<float> h_emb_gpu(batch_seq * embed_dim);

    for (auto& v : h_pos_table) v = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < batch_seq * embed_dim; ++i)
        h_emb_ref[i] = h_out_ref[i];
    for (int i = 0; i < batch_seq * embed_dim; ++i)
        h_emb_gpu[i] = h_out_gpu[i];

    pos_embedding_cpu(h_pos_table.data(), h_emb_ref.data(), batch, seq_len, embed_dim);

    float *d_pos_table;
    CUDA_CHECK(cudaMalloc(&d_pos_table, seq_len * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pos_table, h_pos_table.data(),
                          seq_len * embed_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    pos_embedding_forward(d_pos_table, d_out, batch, seq_len, embed_dim);

    CUDA_CHECK(cudaMemcpy(h_emb_gpu.data(), d_out,
                          batch_seq * embed_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float err_pos = max_abs_diff(h_emb_ref.data(), h_emb_gpu.data(),
                                 batch_seq * embed_dim);
    printf("  positional embedding: err=%.3e %s\n",
           err_pos, err_pos < 1e-5f ? "PASS" : "FAIL");

    bool ok = (err < 1e-5f) && (err_bwd < 1e-5f) && (err_pos < 1e-5f);
    printf("\n=== EMBEDDING TEST %s ===\n", ok ? "PASSED" : "FAILED");

    cudaFree(d_table); cudaFree(d_out); cudaFree(d_tokens);
    cudaFree(d_grad_out); cudaFree(d_grad_table);
    cudaFree(d_pos_table);

    return ok ? 0 : 1;
}
