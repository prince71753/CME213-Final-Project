// Single-GPU CUDA code for the mini-Transformer.
#include "common.h"
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>

static void attention_cpu(const float* Q, const float* K, const float* V,
                          float* out, int BH, int seq, int hd) {
    float scale = 1.0f / sqrtf((float)hd);
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < seq; ++i) {

            std::vector<float> scores(seq);
            float max_s = -1e30f;
            for (int j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += Q[bh*seq*hd + i*hd + d] * K[bh*seq*hd + j*hd + d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }

            float sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                scores[j] = expf(scores[j] - max_s);
                sum += scores[j];
            }
            for (int j = 0; j < seq; ++j) scores[j] /= sum;

            for (int d = 0; d < hd; ++d) {
                float v = 0.0f;
                for (int j = 0; j < seq; ++j)
                    v += scores[j] * V[bh*seq*hd + j*hd + d];
                out[bh*seq*hd + i*hd + d] = v;
            }
        }
    }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

int main() {
    printf("=== Attention Correctness & Performance Test ===\n\n");

    int BH = 4, seq = 64, hd = 32;
    int total = BH * seq * hd;
    printf("BH=%d, seq=%d, head_dim=%d\n", BH, seq, hd);

    std::vector<float> hQ(total), hK(total), hV(total);
    std::vector<float> hOut_ref(total), hOut_naive(total), hOut_tiled(total);
    srand(42);
    for (auto& v : hQ) v = (float)rand()/RAND_MAX - 0.5f;
    for (auto& v : hK) v = (float)rand()/RAND_MAX - 0.5f;
    for (auto& v : hV) v = (float)rand()/RAND_MAX - 0.5f;

    attention_cpu(hQ.data(), hK.data(), hV.data(), hOut_ref.data(), BH, seq, hd);

    float *dQ, *dK, *dV, *dOut;
    CUDA_CHECK(cudaMalloc(&dQ, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, total*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), total*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), total*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), total*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dOut, 0, total*sizeof(float)));
    attention_naive(dQ, dK, dV, dOut, BH, seq, hd);
    CUDA_CHECK(cudaMemcpy(hOut_naive.data(), dOut, total*sizeof(float), cudaMemcpyDeviceToHost));
    float err_naive = max_abs_diff(hOut_ref.data(), hOut_naive.data(), total);
    printf("  naive  max error: %.6e  %s\n", err_naive, err_naive < 1e-3 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaMemset(dOut, 0, total*sizeof(float)));
    attention_tiled(dQ, dK, dV, dOut, BH, seq, hd);
    CUDA_CHECK(cudaMemcpy(hOut_tiled.data(), dOut, total*sizeof(float), cudaMemcpyDeviceToHost));
    float err_tiled = max_abs_diff(hOut_ref.data(), hOut_tiled.data(), total);
    printf("  tiled  max error: %.6e  %s\n", err_tiled, err_tiled < 1e-3 ? "PASS" : "FAIL");

    GpuTimer timer;
    int warmup = 5, iters = 50;

    for (int i = 0; i < warmup; ++i) attention_naive(dQ, dK, dV, dOut, BH, seq, hd);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i) attention_naive(dQ, dK, dV, dOut, BH, seq, hd);
    timer.toc();
    float ms_naive = timer.elapsed_ms() / iters;

    for (int i = 0; i < warmup; ++i) attention_tiled(dQ, dK, dV, dOut, BH, seq, hd);
    cudaDeviceSynchronize();
    timer.tic();
    for (int i = 0; i < iters; ++i) attention_tiled(dQ, dK, dV, dOut, BH, seq, hd);
    timer.toc();
    float ms_tiled = timer.elapsed_ms() / iters;

    printf("  naive:  %.4f ms\n", ms_naive);
    printf("  tiled:  %.4f ms  speedup: %.2fx\n", ms_tiled, ms_naive / ms_tiled);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
    return 0;
}
