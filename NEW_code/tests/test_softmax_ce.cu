// Single-GPU CUDA code for the mini-Transformer.
#include "common.h"
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

static float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static float max_rel_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i) {
        float rel = fabsf(a[i] - b[i]) / (fabsf(a[i]) + 1e-8f);
        d = fmaxf(d, rel);
    }
    return d;
}

static void softmax_cpu(const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = x + r * cols;
        float* row_out = out + r * cols;
        
        float max_val = row_in[0];
        for (int c = 1; c < cols; ++c)
            max_val = fmaxf(max_val, row_in[c]);
        
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }
        
        for (int c = 0; c < cols; ++c)
            row_out[c] /= sum;
    }
}

static float cross_entropy_cpu(const float* logits, const int* targets,
                               float* grad, int batch, int vocab) {
    float total_loss = 0.0f;
    for (int i = 0; i < batch; ++i) {
        int target = targets[i];
        
        // Find max for numerical stability
        float max_logit = logits[i * vocab];
        for (int j = 1; j < vocab; ++j)
            max_logit = fmaxf(max_logit, logits[i * vocab + j]);
        
        // Compute softmax and loss
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab; ++j) {
            sum_exp += expf(logits[i * vocab + j] - max_logit);
        }
        float log_sum_exp = logf(sum_exp) + max_logit;
        float loss = log_sum_exp - logits[i * vocab + target];
        total_loss += loss;
        
        // Compute gradient
        for (int j = 0; j < vocab; ++j) {
            float softmax_j = expf(logits[i * vocab + j] - max_logit) / sum_exp;
            grad[i * vocab + j] = softmax_j;
            if (j == target)
                grad[i * vocab + j] -= 1.0f;
        }
    }
    return total_loss / batch;
}

static bool test_softmax() {
    printf("=== Softmax Test ===\n");
    int rows = 256, cols = 512;
    int total = rows * cols;
    
    std::vector<float> h_in(total);
    std::vector<float> h_out_ref(total);
    std::vector<float> h_out_gpu(total);
    
    srand(42);
    for (auto& v : h_in) v = (float)rand() / RAND_MAX - 0.5f;
    
    // CPU
    softmax_cpu(h_in.data(), h_out_ref.data(), rows, cols);
    
    // GPU
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    softmax_forward(d_in, d_out, rows, cols);
    
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    float err = max_abs_diff(h_out_ref.data(), h_out_gpu.data(), total);
    printf("  softmax: rows=%d cols=%d err=%.3e %s\n",
           rows, cols, err, err < 1e-5f ? "PASS" : "FAIL");
    
    // Performance
    GpuTimer timer;
    int warmup = 5, iters = 100;
    for (int i = 0; i < warmup; ++i)
        softmax_forward(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i)
        softmax_forward(d_in, d_out, rows, cols);
    timer.toc();
    float ms = timer.elapsed_ms() / iters;
    printf("  time: %.4f ms\n", ms);
    
    cudaFree(d_in); cudaFree(d_out);
    return err < 1e-5f;
}

static bool test_cross_entropy() {
    printf("\n=== Cross-Entropy Loss Test ===\n");
    int batch = 128, vocab = 4096;
    int total = batch * vocab;
    
    std::vector<float> h_logits(total);
    std::vector<int> h_targets(batch);
    std::vector<float> h_grad_ref(total);
    std::vector<float> h_grad_gpu(total);
    
    srand(42);
    for (auto& v : h_logits) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& t : h_targets) t = rand() % vocab;
    
    // CPU reference
    float loss_ref = cross_entropy_cpu(h_logits.data(), h_targets.data(),
                                       h_grad_ref.data(), batch, vocab);
    
    // GPU
    float *d_logits, *d_grad;
    int *d_targets;
    CUDA_CHECK(cudaMalloc(&d_logits, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, batch * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    float loss_gpu = cross_entropy_forward(d_logits, d_targets, d_grad, batch, vocab);
    
    CUDA_CHECK(cudaMemcpy(h_grad_gpu.data(), d_grad, total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    float grad_err = max_abs_diff(h_grad_ref.data(), h_grad_gpu.data(), total);
    float loss_err = fabsf(loss_ref - loss_gpu) / (fabsf(loss_ref) + 1e-8f);
    
    printf("  cross_entropy: batch=%d vocab=%d\n", batch, vocab);
    printf("    loss_ref=%.6f loss_gpu=%.6f loss_err=%.3e\n",
           loss_ref, loss_gpu, loss_err);
    printf("    grad_err=%.3e %s\n",
           grad_err, grad_err < 1e-4f ? "PASS" : "FAIL");
    
    // Performance
    GpuTimer timer;
    int warmup = 5, iters = 50;
    for (int i = 0; i < warmup; ++i)
        cross_entropy_forward(d_logits, d_targets, d_grad, batch, vocab);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i)
        cross_entropy_forward(d_logits, d_targets, d_grad, batch, vocab);
    timer.toc();
    float ms = timer.elapsed_ms() / iters;
    float bytes = total * 4 + batch * 4;
    float bw = bytes / (ms / 1e3f) / 1e9f;
    printf("  time: %.4f ms  bandwidth: %.1f GB/s\n", ms, bw);
    
    cudaFree(d_logits); cudaFree(d_grad); cudaFree(d_targets);
    return grad_err < 1e-4f && loss_err < 1e-2f;
}

int main() {
    printf("=== Softmax & Cross-Entropy Correctness & Performance Test ===\n\n");
    
    bool ok = true;
    ok = test_softmax() && ok;
    ok = test_cross_entropy() && ok;
    
    printf("\n=== SOFTMAX/CE TEST %s ===\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
