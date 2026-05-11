// Single-GPU CUDA code for the mini-Transformer.
#include "common.h"
#include "kernels.h"
#include <cmath>
#include <cstdio>
#include <vector>

static void fill_pattern(std::vector<float>& v, float scale, float shift = 0.0f) {
    for (int i = 0; i < (int)v.size(); ++i)
        v[i] = scale * sinf(0.013f * (float)i) +
               0.5f * scale * cosf(0.021f * (float)(i + 7)) + shift;
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float d = 0.0f;
    for (int i = 0; i < (int)a.size(); ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static void report_bandwidth(const char* name, float ms, long long bytes_unfused,
                             long long bytes_fused, float speedup) {
    printf("  [%s]\n", name);
    float bw_unfused = bytes_unfused / (ms / 1e3f) / 1e9f;
    float bw_fused = bytes_fused / (ms / 1e3f) / 1e9f;
    // Theoretical speedup based on memory reduction
    float theoretical_speedup = ((float)bytes_unfused) / bytes_fused;
    printf("    bandwidth unfused: %.2f GB/s | fused: %.2f GB/s\n",
       bw_unfused, bw_fused);
    printf("    memory unfused: %.1f MB | fused: %.1f MB (%.1f%% reduction)\n",
           bytes_unfused / 1e6f, bytes_fused / 1e6f,
           (1.0f - (float)bytes_fused / bytes_unfused) * 100.0f);
    printf("    achieved speedup: %.2fx | theoretical (mem-bound): %.2fx\n",
           speedup, theoretical_speedup);
}

static float time_bias_relu_unfused(float* d_work, const float* d_bias,
                                    float* d_out, int rows, int cols,
                                    int iters) {
    GpuTimer timer;
    for (int i = 0; i < 10; ++i) {
        bias_add(d_work, d_bias, rows, cols);
        relu_forward(d_work, d_out, rows * cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        bias_add(d_work, d_bias, rows, cols);
        relu_forward(d_work, d_out, rows * cols);
    }
    timer.toc();
    return timer.elapsed_ms() / iters;
}

static float time_bias_relu_fused(float* d_work, const float* d_bias,
                                  float* d_out, int rows, int cols,
                                  int iters) {
    GpuTimer timer;
    for (int i = 0; i < 10; ++i)
        bias_relu_forward(d_work, d_bias, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i)
        bias_relu_forward(d_work, d_bias, d_out, rows, cols);
    timer.toc();
    return timer.elapsed_ms() / iters;
}

static bool test_bias_relu() {
    int rows = 2048, cols = 512, n = rows * cols, iters = 200;
    std::vector<float> h_in(n), h_bias(cols);
    fill_pattern(h_in, 0.1f);
    fill_pattern(h_bias, 0.001f);

    float *d_base, *d_unfused_work, *d_fused_work, *d_bias, *d_unfused, *d_fused;
    CUDA_CHECK(cudaMalloc(&d_base, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unfused_work, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fused_work, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unfused, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fused, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_unfused_work, d_base, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_fused_work, d_base, n * sizeof(float), cudaMemcpyDeviceToDevice));
    bias_add(d_unfused_work, d_bias, rows, cols);
    relu_forward(d_unfused_work, d_unfused, n);
    bias_relu_forward(d_fused_work, d_bias, d_fused, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_unfused(n), h_fused(n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_unfused.data(), d_unfused, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_fused, n * sizeof(float), cudaMemcpyDeviceToHost));
    float err = max_abs_diff(h_unfused, h_fused);

    CUDA_CHECK(cudaMemcpy(d_unfused_work, d_base, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_fused_work, d_base, n * sizeof(float), cudaMemcpyDeviceToDevice));
    float ms_unfused = time_bias_relu_unfused(d_unfused_work, d_bias, d_unfused, rows, cols, iters);
    float ms_fused = time_bias_relu_fused(d_fused_work, d_bias, d_fused, rows, cols, iters);

    printf("  bias+ReLU rows=%d cols=%d err=%.3e %s\n",
           rows, cols, err, err < 1e-6f ? "PASS" : "FAIL");
    printf("    unfused %.4f ms | fused %.4f ms | speedup %.2fx\n", ms_unfused, ms_fused, ms_unfused / ms_fused);
    
        
    long long bytes_unfused = 4LL * n * sizeof(float);
    long long bytes_fused   = 2LL * n * sizeof(float);
    report_bandwidth("bias+ReLU", ms_unfused, bytes_unfused, bytes_fused, ms_unfused / ms_fused);


    cudaFree(d_base); cudaFree(d_unfused_work); cudaFree(d_fused_work);
    cudaFree(d_bias); cudaFree(d_unfused); cudaFree(d_fused);
    return err < 1e-6f;
}

static float time_residual_ln_unfused(const float* d_a, const float* d_b,
                                      const float* d_gamma, const float* d_beta,
                                      float* d_res, float* d_out,
                                      float* d_mean, float* d_inv,
                                      int rows, int cols, int iters) {
    GpuTimer timer;
    for (int i = 0; i < 10; ++i) {
        residual_add(d_a, d_b, d_res, rows * cols);
        layernorm_forward_save(d_res, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        residual_add(d_a, d_b, d_res, rows * cols);
        layernorm_forward_save(d_res, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);
    }
    timer.toc();
    return timer.elapsed_ms() / iters;
}

static float time_residual_ln_fused(const float* d_a, const float* d_b,
                                    const float* d_gamma, const float* d_beta,
                                    float* d_res, float* d_out,
                                    float* d_mean, float* d_inv,
                                    int rows, int cols, int iters) {
    GpuTimer timer;
    for (int i = 0; i < 10; ++i)
        residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta,
                                        d_res, d_out, d_mean, d_inv, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i)
        residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta,
                                        d_res, d_out, d_mean, d_inv, rows, cols);
    timer.toc();
    return timer.elapsed_ms() / iters;
}

static bool test_residual_layernorm_forward() {
    int rows = 2048, cols = 128, n = rows * cols, iters = 300;
    std::vector<float> h_a(n), h_b(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_a, 0.1f);
    fill_pattern(h_b, 0.08f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.02f);

    float *d_a, *d_b, *d_gamma, *d_beta;
    float *d_res_u, *d_out_u, *d_mean_u, *d_inv_u;
    float *d_res_f, *d_out_f, *d_mean_f, *d_inv_f;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_u, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_u, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_f, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_f, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    residual_add(d_a, d_b, d_res_u, n);
    layernorm_forward_save(d_res_u, d_gamma, d_beta, d_out_u, d_mean_u, d_inv_u, rows, cols);
    residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta,
                                    d_res_f, d_out_f, d_mean_f, d_inv_f, rows, cols);

    std::vector<float> h_out_u(n), h_out_f(n), h_res_u(n), h_res_f(n);
    CUDA_CHECK(cudaMemcpy(h_out_u.data(), d_out_u, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_f.data(), d_out_f, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_u.data(), d_res_u, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_f.data(), d_res_f, n * sizeof(float), cudaMemcpyDeviceToHost));
    float out_err = max_abs_diff(h_out_u, h_out_f);
    float res_err = max_abs_diff(h_res_u, h_res_f);

    float ms_unfused = time_residual_ln_unfused(d_a, d_b, d_gamma, d_beta,
                                                d_res_u, d_out_u, d_mean_u, d_inv_u,
                                                rows, cols, iters);
    float ms_fused = time_residual_ln_fused(d_a, d_b, d_gamma, d_beta,
                                            d_res_f, d_out_f, d_mean_f, d_inv_f,
                                            rows, cols, iters);

    printf("  residual+LayerNorm fwd rows=%d cols=%d res_err=%.3e out_err=%.3e %s\n",
           rows, cols, res_err, out_err,
           (res_err < 1e-6f && out_err < 1e-5f) ? "PASS" : "FAIL");
    printf("    unfused %.4f ms | fused %.4f ms | speedup %.2fx\n",
           ms_unfused, ms_fused, ms_unfused / ms_fused);

    long long bytes_unfused = 4LL * n * sizeof(float) + 2LL * cols * sizeof(float);
    long long bytes_fused   = 3LL * n * sizeof(float) + 2LL * cols * sizeof(float);
    report_bandwidth("residual+LayerNorm", ms_unfused, bytes_unfused, bytes_fused, ms_unfused / ms_fused);
    

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_res_u); cudaFree(d_out_u); cudaFree(d_mean_u); cudaFree(d_inv_u);
    cudaFree(d_res_f); cudaFree(d_out_f); cudaFree(d_mean_f); cudaFree(d_inv_f);
    return res_err < 1e-6f && out_err < 1e-5f;
}

static bool test_layernorm_backward_residual() {
    int rows = 2048, cols = 128, n = rows * cols, iters = 200;
    std::vector<float> h_x(n), h_grad(n), h_residual(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_x, 0.1f);
    fill_pattern(h_grad, 0.05f);
    fill_pattern(h_residual, 0.02f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.01f);

    float *d_x, *d_grad, *d_gamma, *d_beta, *d_out, *d_mean, *d_inv;
    float *d_dx, *d_res_u, *d_res_f, *d_dg_u, *d_db_u, *d_dg_f, *d_db_f;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dg_u, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_u, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dg_f, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_f, cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, h_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward_save(d_x, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);

    CUDA_CHECK(cudaMemcpy(d_res_u, h_residual.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res_f, h_residual.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dg_u, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_u, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dg_f, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_f, 0, cols * sizeof(float)));
    layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                       d_dx, d_dg_u, d_db_u, rows, cols);
    residual_add(d_res_u, d_dx, d_res_u, n);
    layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                d_res_f, d_dg_f, d_db_f, rows, cols);

    std::vector<float> h_res_u(n), h_res_f(n), h_dg_u(cols), h_dg_f(cols), h_db_u(cols), h_db_f(cols);
    CUDA_CHECK(cudaMemcpy(h_res_u.data(), d_res_u, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_f.data(), d_res_f, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dg_u.data(), d_dg_u, cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dg_f.data(), d_dg_f, cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db_u.data(), d_db_u, cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db_f.data(), d_db_f, cols * sizeof(float), cudaMemcpyDeviceToHost));
    float res_err = max_abs_diff(h_res_u, h_res_f);
    float dg_err = max_abs_diff(h_dg_u, h_dg_f);
    float db_err = max_abs_diff(h_db_u, h_db_f);

    GpuTimer timer;
    for (int i = 0; i < 10; ++i) {
        layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                           d_dx, d_dg_u, d_db_u, rows, cols);
        residual_add(d_res_u, d_dx, d_res_u, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                           d_dx, d_dg_u, d_db_u, rows, cols);
        residual_add(d_res_u, d_dx, d_res_u, n);
    }
    timer.toc();
    float ms_unfused = timer.elapsed_ms() / iters;

    for (int i = 0; i < 10; ++i)
        layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                    d_res_f, d_dg_f, d_db_f, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iters; ++i)
        layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                    d_res_f, d_dg_f, d_db_f, rows, cols);
    timer.toc();
    float ms_fused = timer.elapsed_ms() / iters;

    bool ok = res_err < 1e-5f && dg_err < 1e-4f && db_err < 1e-4f;
    printf("  LayerNorm bwd+residual rows=%d cols=%d res_err=%.3e dgamma_err=%.3e dbeta_err=%.3e %s\n",
           rows, cols, res_err, dg_err, db_err, ok ? "PASS" : "FAIL");
    printf("    unfused %.4f ms | fused %.4f ms | speedup %.2fx\n",
           ms_unfused, ms_fused, ms_unfused / ms_fused);

    cudaFree(d_x); cudaFree(d_grad); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_out); cudaFree(d_mean); cudaFree(d_inv); cudaFree(d_dx);
    cudaFree(d_res_u); cudaFree(d_res_f); cudaFree(d_dg_u); cudaFree(d_db_u);
    cudaFree(d_dg_f); cudaFree(d_db_f);
    return ok;
}

int main() {
    printf("=== Fusion Correctness & Performance Test ===\n\n");
    bool ok = true;
    ok = test_bias_relu() && ok;
    ok = test_residual_layernorm_forward() && ok;
    ok = test_layernorm_backward_residual() && ok;
    printf("\n=== FUSION TEST %s ===\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
