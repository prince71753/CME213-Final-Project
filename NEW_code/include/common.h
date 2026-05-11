// Single-GPU CUDA code for the mini-Transformer.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_map>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#ifndef VOCAB_SIZE
#define VOCAB_SIZE 65
#endif

#ifndef SEQ_LEN
#define SEQ_LEN 64
#endif

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 128
#endif

#ifndef NUM_HEADS
#define NUM_HEADS 4
#endif

#ifndef HEAD_DIM
#define HEAD_DIM (HIDDEN_DIM / NUM_HEADS)
#endif

#ifndef FF_DIM
#define FF_DIM (4 * HIDDEN_DIM)
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 32
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic()  { CUDA_CHECK(cudaEventRecord(start, 0)); }
    void toc()  { CUDA_CHECK(cudaEventRecord(stop, 0));
                   CUDA_CHECK(cudaEventSynchronize(stop)); }
    float elapsed_ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
