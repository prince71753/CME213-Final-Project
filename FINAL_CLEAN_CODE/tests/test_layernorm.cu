// layer normalization correctness test.
#include "common.h"
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>

static void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                          float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[r*cols + c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = x[r*cols + c] - mean;
            var += d * d;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; ++c)
            out[r*cols + c] = (x[r*cols + c] - mean) * inv_std * gamma[c] + beta[c];
    }
}

int main() {
    printf("=== LayerNorm Correctness & Performance Test ===\n\n");

    int rows = 2048, cols = 128;
    float eps = 1e-5f;
    printf("rows=%d, cols=%d\n", rows, cols);

    std::vector<float> hx(rows*cols), hgamma(cols, 1.0f), hbeta(cols, 0.0f);
    std::vector<float> hout_ref(rows*cols), hout_gpu(rows*cols);
    srand(42);
    for (auto& v : hx) v = (float)rand()/RAND_MAX - 0.5f;

    layernorm_cpu(hx.data(), hgamma.data(), hbeta.data(),
                  hout_ref.data(), rows, cols, eps);

    float *dx, *dgamma, *dbeta, *dout;
    CUDA_CHECK(cudaMalloc(&dx, rows*cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dgamma, cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbeta, cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, rows*cols*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dgamma, hgamma.data(), cols*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta, hbeta.data(), cols*sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward(dx, dgamma, dbeta, dout, rows, cols, eps);
    CUDA_CHECK(cudaMemcpy(hout_gpu.data(), dout, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < rows*cols; ++i)
        max_err = fmaxf(max_err, fabsf(hout_ref[i] - hout_gpu[i]));
    printf("  max error: %.6e  %s\n", max_err, max_err < 1e-3 ? "PASS" : "FAIL");

    GpuTimer timer;
    int warmup = 5, iters = 100;
    for (int i = 0; i < warmup; ++i)
        layernorm_forward(dx, dgamma, dbeta, dout, rows, cols, eps);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i)
        layernorm_forward(dx, dgamma, dbeta, dout, rows, cols, eps);
    timer.toc();
    float ms = timer.elapsed_ms() / iters;
    float bw = 2.0f * rows * cols * sizeof(float) / (ms / 1e3f) / 1e9f;
    printf("  time: %.4f ms  effective bandwidth: %.1f GB/s\n", ms, bw);

    cudaFree(dx); cudaFree(dgamma); cudaFree(dbeta); cudaFree(dout);
    return 0;
}
