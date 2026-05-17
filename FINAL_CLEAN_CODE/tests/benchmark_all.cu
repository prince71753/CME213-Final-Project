// kernel benchmark harness.
#include "common.h"
#include "kernels.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>

static float max_err(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; ++i) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static void rand_init(std::vector<float>& v, unsigned seed = 42) {
    srand(seed);
    for (auto& x : v) x = (float)rand() / RAND_MAX - 0.5f;
}

void benchmark_gemm() {
    printf("========================================\n");
    printf("  GEMM Benchmark: Naive vs Tiled\n");
    printf("========================================\n");
    printf("%6s %6s %6s | %10s %10s | %10s %10s | %7s\n",
           "M", "N", "K", "naive(ms)", "GFLOP/s", "tiled(ms)", "GFLOP/s", "speedup");
    printf("------+------+------+------------+------------+------------+------------+--------\n");

    int sizes[][3] = {
        {32,32,32}, {64,64,64}, {128,128,128}, {256,256,256},
        {512,512,512}, {1024,1024,1024}, {2048,128,128}, {2048,512,128}
    };

    for (auto& sz : sizes) {
        int M = sz[0], N = sz[1], K = sz[2];
        float gflops = 2.0f * M * N * K / 1e9f;

        std::vector<float> hA(M*K), hB(K*N);
        rand_init(hA); rand_init(hB);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

        GpuTimer timer;
        int warmup = 5, iters = 30;

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

        printf("%6d %6d %6d | %10.4f %10.1f | %10.4f %10.1f | %6.2fx\n",
               M, N, K, ms_naive, gflops/(ms_naive/1e3f),
               ms_tiled, gflops/(ms_tiled/1e3f), ms_naive/ms_tiled);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    printf("\n");
}

void benchmark_attention() {
    printf("========================================\n");
    printf("  Attention Benchmark: Naive vs Tiled\n");
    printf("========================================\n");
    printf("%4s %4s %3s | %10s %10s | %7s\n",
           "BH", "seq", "hd", "naive(ms)", "tiled(ms)", "speedup");
    printf("----+----+---+------------+------------+--------\n");

    int configs[][3] = {{4,32,32}, {4,64,32}, {4,128,32}, {8,64,32}, {8,128,64}};

    for (auto& c : configs) {
        int BH = c[0], seq = c[1], hd = c[2];
        int total = BH * seq * hd;

        std::vector<float> hQ(total), hK(total), hV(total);
        rand_init(hQ); rand_init(hK, 123); rand_init(hV, 456);

        float *dQ, *dK, *dV, *dOut;
        CUDA_CHECK(cudaMalloc(&dQ, total*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK, total*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV, total*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, total*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), total*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), total*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data(), total*sizeof(float), cudaMemcpyHostToDevice));

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

        printf("%4d %4d %3d | %10.4f %10.4f | %6.2fx\n",
               BH, seq, hd, ms_naive, ms_tiled, ms_naive/ms_tiled);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
    }
    printf("\n");
}

void benchmark_layernorm() {
    printf("========================================\n");
    printf("  LayerNorm Benchmark\n");
    printf("========================================\n");
    printf("%6s %6s | %10s %12s\n", "rows", "cols", "time(ms)", "BW(GB/s)");
    printf("------+------+------------+-------------\n");

    int configs[][2] = {{1024,64}, {1024,128}, {2048,128}, {4096,128}, {2048,256}, {2048,512}};

    for (auto& c : configs) {
        int rows = c[0], cols = c[1];
        int n = rows * cols;

        std::vector<float> hx(n), hg(cols, 1.0f), hb(cols, 0.0f);
        rand_init(hx);

        float *dx, *dg, *db, *dout;
        CUDA_CHECK(cudaMalloc(&dx, n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dg, cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db, cols*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dout, n*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), cols*sizeof(float), cudaMemcpyHostToDevice));

        GpuTimer timer;
        int warmup = 10, iters = 200;
        for (int i = 0; i < warmup; ++i)
            layernorm_forward(dx, dg, db, dout, rows, cols);
        cudaDeviceSynchronize();
        timer.tic();
        for (int i = 0; i < iters; ++i)
            layernorm_forward(dx, dg, db, dout, rows, cols);
        timer.toc();
        float ms = timer.elapsed_ms() / iters;
        float bw = 2.0f * n * sizeof(float) / (ms / 1e3f) / 1e9f;

        printf("%6d %6d | %10.4f %12.1f\n", rows, cols, ms, bw);

        cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dout);
    }
    printf("\n");
}

void roofline_analysis() {
    printf("========================================\n");
    printf("  Roofline Analysis (RTX 6000 Turing)\n");
    printf("========================================\n");
    printf("Peak FP32:     16.3 TFLOPS\n");
    printf("Peak BW:       672 GB/s\n");
    printf("Ridge point:   16300/672 = %.1f FLOP/byte\n\n",
           16300.0f / 672.0f);

    printf("%-20s | %12s | %12s | %12s | %s\n",
           "Kernel", "AI(FLOP/B)", "Achieved", "Ceiling", "Bound");
    printf("--------------------+-------------+-------------+-------------+----------\n");

    {
        float flops = 2.0f * 512 * 512 * 512;
        float bytes = 3.0f * 512 * 512 * 4;
        float ai = flops / bytes;
        float achieved_tflops = 2.127f;
        float bw_ceiling = 672.0f * ai / 1e3f;
        const char* bound = (achieved_tflops < bw_ceiling) ? "compute" : "memory";
        printf("%-20s | %12.1f | %10.1f T | %10.1f T | %s\n",
               "GEMM tiled 512", ai, achieved_tflops, fminf(bw_ceiling, 16.3f), bound);
    }

    {
        int rows = 2048, cols = 128;
        float flops = 5.0f * rows * cols;
        float bytes = 3.0f * rows * cols * 4;
        float ai = flops / bytes;
        float achieved_bw = 292.7f;
        float achieved_tflops = achieved_bw * ai / 1e3f;
        const char* bound = "memory";
        printf("%-20s | %12.1f | %8.3f T   | %10.1f T | %s\n",
               "LayerNorm 2048x128", ai, achieved_tflops, 672.0f * ai / 1e3f, bound);
    }

    {
        int seq = 64, hd = 32;
        float flops = 2.0f * seq * seq * hd + seq * seq;
        float bytes = 3.0f * seq * hd * 4 + seq * hd * 4;
        float ai = flops / bytes;
        printf("%-20s | %12.1f | %10s   | %10.1f T | %s\n",
               "Attention 64x32", ai, "(varies)", fminf(672.0f * ai / 1e3f, 16.3f),
               ai > 24.3f ? "compute" : "memory");
    }

    printf("\n");
}

void benchmark_cublas_comparison() {
    printf("==============================================================\n");
    printf("  Custom GEMM vs cuBLAS — Transformer-Relevant Matrix Sizes\n");
    printf("==============================================================\n");
    printf("%-24s | %6s %6s %6s | %10s %10s | %10s %10s | %7s\n",
           "Operation", "M", "N", "K", "custom(ms)", "GFLOP/s", "cuBLAS(ms)", "GFLOP/s", "ratio");
    printf("------------------------+------+------+------+------------+------------+------------+------------+--------\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    struct GemmCase {
        const char* name;
        int M, N, K;
    };
    GemmCase cases[] = {
        {"QKV/Wo proj", 2048, 128, 128},
        {"FFN W1 fwd", 2048, 512, 128},
        {"FFN W2 fwd", 2048, 128, 512},
        {"Logits fwd", 2048, 65, 128},
        {"Large GEMM", 1024, 1024, 1024},
        {"XL GEMM", 2048, 2048, 2048},
    };

    float alpha = 1.0f, beta = 0.0f;

    for (auto& c : cases) {
        int M = c.M, N = c.N, K = c.K;
        float gflops = 2.0f * M * N * K / 1e9f;

        std::vector<float> hA(M * K), hB(K * N);
        rand_init(hA); rand_init(hB, 123);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        GpuTimer timer;
        int warmup = 10, iters = 50;

        for (int i = 0; i < warmup; ++i) gemm_tiled(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        timer.tic();
        for (int i = 0; i < iters; ++i) gemm_tiled(dA, dB, dC, M, N, K);
        timer.toc();
        float ms_custom = timer.elapsed_ms() / iters;

        for (int i = 0; i < warmup; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        cudaDeviceSynchronize();
        timer.tic();
        for (int i = 0; i < iters; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        timer.toc();
        float ms_cublas = timer.elapsed_ms() / iters;

        printf("%-24s | %6d %6d %6d | %10.4f %10.1f | %10.4f %10.1f | %6.1f%%\n",
               c.name, M, N, K,
               ms_custom, gflops / (ms_custom / 1e3f),
               ms_cublas, gflops / (ms_cublas / 1e3f),
               (ms_cublas / ms_custom) * 100.0f);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    cublasDestroy(handle);
    printf("\nNote: ratio = cuBLAS_time / custom_time × 100%%. Higher = custom is faster.\n");
    printf("      100%% = same speed, <100%% = cuBLAS is faster.\n\n");
}

int main() {
    benchmark_gemm();
    benchmark_attention();
    benchmark_layernorm();
    roofline_analysis();
    benchmark_cublas_comparison();
    return 0;
}
