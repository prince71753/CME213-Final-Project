// Training hotspot profiling driver for GEMM and backward kernels.
#include "common.h"
#include "kernels.h"
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstring>

__global__ void fill_values_kernel(float* x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = (i * 17 + 23) & 255;
        x[i] = scale * ((float)v / 255.0f - 0.5f);
    }
}

static void fill_values(float* x, int n, float scale) {
    int block = 256;
    fill_values_kernel<<<(n + block - 1) / block, block>>>(x, n, scale);
    CUDA_CHECK_LAST();
}

struct HotspotCase {
    const char* name;
    int kind;
    int m;
    int n;
    int k;
    int batch;
};

static HotspotCase get_case(const char* name) {
    HotspotCase cases[] = {
        {"splitk_dW1", 0, 256, 1024, 2048, 1},
        {"splitk_dW2", 0, 1024, 256, 2048, 1},
        {"splitk_qkv", 0, 256, 256, 2048, 1},
        {"splitk_dWout", 0, 256, 65, 2048, 1},
        {"bt_ff1", 1, 2048, 256, 1024, 1},
        {"fwd_ff1", 2, 2048, 1024, 256, 1},
        {"batched_qkv", 3, 2048, 256, 256, 3},
    };
    for (auto& c : cases)
        if (strcmp(name, c.name) == 0)
            return c;
    fprintf(stderr, "Unknown case '%s'\n", name);
    fprintf(stderr, "Cases: splitk_dW1 splitk_dW2 splitk_qkv splitk_dWout bt_ff1 fwd_ff1 batched_qkv\n");
    exit(1);
}

static void run_case(const HotspotCase& c, const float* A, const float* B,
                     float* C) {
    if (c.kind == 0) {
        gemm_splitk_AT_acc(A, B, C, c.m, c.n, c.k);
    } else if (c.kind == 1) {
        gemm_tiled_BT(A, B, C, c.m, c.n, c.k);
    } else if (c.kind == 2) {
        gemm_tiled(A, B, C, c.m, c.n, c.k);
    } else {
        int stride_a = 0;
        int stride_b = c.k * c.n;
        int stride_c = c.m * c.n;
        gemm_batched(A, B, C, c.m, c.n, c.k, c.batch,
                     stride_a, stride_b, stride_c);
    }
}

int main(int argc, char** argv) {
    const char* case_name = "splitk_dW1";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0 && i + 1 < argc)
            case_name = argv[++i];
    }

    HotspotCase c = get_case(case_name);
    int a_count = (c.kind == 3) ? c.m * c.k : c.k * c.m;
    if (c.kind == 1 || c.kind == 2)
        a_count = c.m * c.k;
    int b_count = c.k * c.n * c.batch;
    int out_count = c.m * c.n * c.batch;

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, a_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, b_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, out_count * sizeof(float)));
    fill_values(A, a_count, 0.05f);
    fill_values(B, b_count, 0.04f);
    CUDA_CHECK(cudaMemset(C, 0, out_count * sizeof(float)));

    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemset(C, 0, out_count * sizeof(float)));
        run_case(c, A, B, C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 20;
    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(C, 0, out_count * sizeof(float)));
        run_case(c, A, B, C);
    }
    timer.toc();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(C, 0, out_count * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStart();
    run_case(c, A, B, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();

    double flops = 2.0 * (double)c.m * (double)c.n * (double)c.k *
                   (double)c.batch;
    double runtime_us = (double)timer.elapsed_ms() * 1000.0 / (double)iters;
    double gflops = flops / (runtime_us * 1000.0);
    printf("HOTSPOT_TIMING case=%s m=%d n=%d k=%d batch=%d runtime_us=%.3f flops=%.0f gflops=%.3f\n",
           c.name, c.m, c.n, c.k, c.batch, runtime_us, flops, gflops);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
