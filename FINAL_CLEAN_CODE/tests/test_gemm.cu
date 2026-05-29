// gemm correctness and timing test.
#include "common.h"
#include "kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void gemm_cpu(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

int main() {
    printf("=== GEMM Correctness & Performance Test ===\n\n");

    int sizes[][3] = {{64,64,64}, {128,128,128}, {256,256,256}, {512,512,512}};

    for (auto& sz : sizes) {
        int M = sz[0], N = sz[1], K = sz[2];
        printf("M=%d, N=%d, K=%d\n", M, N, K);

        std::vector<float> hA(M*K), hB(K*N), hC_ref(M*N), hC_naive(M*N), hC_tiled(M*N);
        srand(42);
        for (auto& v : hA) v = (float)rand()/RAND_MAX - 0.5f;
        for (auto& v : hB) v = (float)rand()/RAND_MAX - 0.5f;

        gemm_cpu(hA.data(), hB.data(), hC_ref.data(), M, N, K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(dC, 0, M*N*sizeof(float)));
        gemm_naive(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaMemcpy(hC_naive.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost));
        float err_naive = max_abs_diff(hC_ref.data(), hC_naive.data(), M*N);
        printf("  naive  max error: %.6e  %s\n", err_naive, err_naive < 1e-3 ? "PASS" : "FAIL");

        CUDA_CHECK(cudaMemset(dC, 0, M*N*sizeof(float)));
        gemm_tiled(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaMemcpy(hC_tiled.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost));
        float err_tiled = max_abs_diff(hC_ref.data(), hC_tiled.data(), M*N);
        float tolerance = getenv("CME213_STRICT_FP32") ? 1e-3f : 5e-3f;
        printf("  tiled  max error: %.6e  %s\n", err_tiled,err_tiled < tolerance ? "PASS" : "FAIL");

        GpuTimer timer;
        int warmup = 3, iters = 20;

        for (int i = 0; i < warmup; ++i) gemm_naive(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        timer.tic();
        for (int i = 0; i < iters; ++i) gemm_naive(dA, dB, dC, M, N, K);
        timer.toc();
        float ms_naive = timer.elapsed_ms() / iters;

        for (int i = 0; i < warmup; ++i) gemm_tiled(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        timer.tic();
        for (int i = 0; i < iters; ++i) gemm_tiled(dA, dB, dC, M, N, K);
        timer.toc();
        float ms_tiled = timer.elapsed_ms() / iters;

        float gflops = 2.0f * M * N * K / 1e9f;
        printf("  naive:  %.3f ms (%.1f GFLOP/s)\n", ms_naive, gflops / (ms_naive / 1e3f));
        printf("  tiled:  %.3f ms (%.1f GFLOP/s)  speedup: %.2fx\n",
               ms_tiled, gflops / (ms_tiled / 1e3f), ms_naive / ms_tiled);
        printf("\n");

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
