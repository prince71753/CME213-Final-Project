// gemm tuning harness.
#include "common.h"
#include "kernels.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>

static void rand_init(std::vector<float>& v, unsigned seed = 42) {
    srand(seed);
    for (auto& x : v) x = (float)rand() / RAND_MAX - 0.5f;
}

template <int BM_, int BN_, int BK_, int TM_, int TN_>
__global__ __launch_bounds__((BM_ / TM_) * (BN_ / TN_))
void gemm_tune_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    constexpr int THREADS = (BM_ / TM_) * (BN_ / TN_);
    __shared__ float sA[2][BK_][BM_ + 1];
    __shared__ float sB[2][BK_][BN_ + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN_ / TN_);
    const int ty   = tid / (BN_ / TN_);
    const int brow = blockIdx.y * BM_;
    const int bcol = blockIdx.x * BN_;

    float rc[TM_][TN_] = {};
    const int num_k = (K + BK_ - 1) / BK_;

    {
        constexpr int LOAD_A = (BM_ * BK_ + THREADS - 1) / THREADS;
        constexpr int LOAD_B = (BK_ * BN_ + THREADS - 1) / THREADS;
        #pragma unroll
        for (int l = 0; l < LOAD_A; ++l) {
            int idx = tid + l * THREADS;
            if (idx < BM_ * BK_) {
                int r = idx / BK_, c = idx % BK_;
                int gr = brow + r, gc = c;
                sA[0][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
        }
        #pragma unroll
        for (int l = 0; l < LOAD_B; ++l) {
            int idx = tid + l * THREADS;
            if (idx < BK_ * BN_) {
                int r = idx / BN_, c = idx % BN_;
                int gr = r, gc = bcol + c;
                sB[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < num_k) {
            constexpr int LOAD_A = (BM_ * BK_ + THREADS - 1) / THREADS;
            constexpr int LOAD_B = (BK_ * BN_ + THREADS - 1) / THREADS;
            int k_off = (kt + 1) * BK_;
            #pragma unroll
            for (int l = 0; l < LOAD_A; ++l) {
                int idx = tid + l * THREADS;
                if (idx < BM_ * BK_) {
                    int r = idx / BK_, c = idx % BK_;
                    int gr = brow + r, gc = k_off + c;
                    sA[nxt][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < LOAD_B; ++l) {
                int idx = tid + l * THREADS;
                if (idx < BK_ * BN_) {
                    int r = idx / BN_, c = idx % BN_;
                    int gr = k_off + r, gc = bcol + c;
                    sB[nxt][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < BK_; ++k) {
            float rA[TM_], rB[TN_];
            #pragma unroll
            for (int i = 0; i < TM_; ++i) rA[i] = sA[cur][k][ty * TM_ + i];
            #pragma unroll
            for (int j = 0; j < TN_; ++j) rB[j] = sB[cur][k][tx * TN_ + j];
            #pragma unroll
            for (int i = 0; i < TM_; ++i)
                #pragma unroll
                for (int j = 0; j < TN_; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM_; ++i) {
        int grow = brow + ty * TM_ + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN_; ++j) {
            int gcol = bcol + tx * TN_ + j;
            if (gcol >= N) continue;
            C[grow * N + gcol] = rc[i][j];
        }
    }
}

template <int BM_, int BN_, int BK_, int TM_, int TN_>
__global__ __launch_bounds__((BM_ / TM_) * (BN_ / TN_))
void gemm_tune_vec4_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    constexpr int THREADS = (BM_ / TM_) * (BN_ / TN_);
    __shared__ float sA[2][BK_][BM_ + 1];
    __shared__ float sB[2][BK_][BN_ + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN_ / TN_);
    const int ty   = tid / (BN_ / TN_);
    const int brow = blockIdx.y * BM_;
    const int bcol = blockIdx.x * BN_;

    float rc[TM_][TN_] = {};
    const int num_k = (K + BK_ - 1) / BK_;

    constexpr int A_ELEMS = BM_ * BK_;
    constexpr int B_ELEMS = BK_ * BN_;
    constexpr int A_VEC4 = (A_ELEMS + 3) / 4;
    constexpr int B_VEC4 = (B_ELEMS + 3) / 4;
    constexpr int A_LOADS = (A_VEC4 + THREADS - 1) / THREADS;
    constexpr int B_LOADS = (B_VEC4 + THREADS - 1) / THREADS;

    auto load_A_tile = [&](int buf, int k_off) {
        #pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            int vid = tid + l * THREADS;
            if (vid < A_VEC4) {
                int elem = vid * 4;
                int r = elem / BK_;
                int c = elem % BK_;
                int gr = brow + r;
                int gc = k_off + c;
                if (gr < M && gc + 3 < K && (c % 4) == 0) {
                    float4 v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
                    sA[buf][c    ][r] = v.x;
                    sA[buf][c + 1][r] = v.y;
                    sA[buf][c + 2][r] = v.z;
                    sA[buf][c + 3][r] = v.w;
                } else {
                    for (int d = 0; d < 4 && (elem + d) < A_ELEMS; ++d) {
                        int rr = (elem + d) / BK_;
                        int cc = (elem + d) % BK_;
                        int grr = brow + rr, gcc = k_off + cc;
                        sA[buf][cc][rr] = (grr < M && gcc < K) ? A[grr * K + gcc] : 0.0f;
                    }
                }
            }
        }
    };

    auto load_B_tile = [&](int buf, int k_off) {
        #pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            int vid = tid + l * THREADS;
            if (vid < B_VEC4) {
                int elem = vid * 4;
                int r = elem / BN_;
                int c = elem % BN_;
                int gr = k_off + r;
                int gc = bcol + c;
                if (gr < K && gc + 3 < N && (c % 4) == 0) {
                    float4 v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
                    sB[buf][r][c    ] = v.x;
                    sB[buf][r][c + 1] = v.y;
                    sB[buf][r][c + 2] = v.z;
                    sB[buf][r][c + 3] = v.w;
                } else {
                    for (int d = 0; d < 4 && (elem + d) < B_ELEMS; ++d) {
                        int rr = (elem + d) / BN_;
                        int cc = (elem + d) % BN_;
                        int grr = k_off + rr, gcc = bcol + cc;
                        sB[buf][rr][cc] = (grr < K && gcc < N) ? B[grr * N + gcc] : 0.0f;
                    }
                }
            }
        }
    };

    load_A_tile(0, 0);
    load_B_tile(0, 0);
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt + 1 < num_k) {
            load_A_tile(nxt, (kt + 1) * BK_);
            load_B_tile(nxt, (kt + 1) * BK_);
        }

        #pragma unroll
        for (int k = 0; k < BK_; ++k) {
            float rA[TM_], rB[TN_];
            #pragma unroll
            for (int i = 0; i < TM_; ++i) rA[i] = sA[cur][k][ty * TM_ + i];
            #pragma unroll
            for (int j = 0; j < TN_; ++j) rB[j] = sB[cur][k][tx * TN_ + j];
            #pragma unroll
            for (int i = 0; i < TM_; ++i)
                #pragma unroll
                for (int j = 0; j < TN_; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM_; ++i) {
        int grow = brow + ty * TM_ + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN_; ++j) {
            int gcol = bcol + tx * TN_ + j;
            if (gcol >= N) continue;
            C[grow * N + gcol] = rc[i][j];
        }
    }
}

template <int BM_, int BN_, int BK_, int TM_, int TN_, bool VEC4>
float bench_config(const float* dA, const float* dB, float* dC,
                   int M, int N, int K, int warmup, int iters) {
    constexpr int THREADS = (BM_ / TM_) * (BN_ / TN_);
    dim3 block(THREADS);
    dim3 grid((N + BN_ - 1) / BN_, (M + BM_ - 1) / BM_);

    for (int i = 0; i < warmup; ++i) {
        if constexpr (VEC4)
            gemm_tune_vec4_kernel<BM_, BN_, BK_, TM_, TN_><<<grid, block>>>(dA, dB, dC, M, N, K);
        else
            gemm_tune_kernel<BM_, BN_, BK_, TM_, TN_><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();

    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        if constexpr (VEC4)
            gemm_tune_vec4_kernel<BM_, BN_, BK_, TM_, TN_><<<grid, block>>>(dA, dB, dC, M, N, K);
        else
            gemm_tune_kernel<BM_, BN_, BK_, TM_, TN_><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    timer.toc();
    return timer.elapsed_ms() / iters;
}

float bench_cublas(cublasHandle_t handle, const float* dA, const float* dB, float* dC,
                   int M, int N, int K, int warmup, int iters) {
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < warmup; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();

    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < iters; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    timer.toc();
    return timer.elapsed_ms() / iters;
}

float max_error(float* dC, float* dRef, int n) {
    std::vector<float> hC(n), hRef(n);
    cudaMemcpy(hC.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hRef.data(), dRef, n * sizeof(float), cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (int i = 0; i < n; ++i) err = fmaxf(err, fabsf(hC[i] - hRef[i]));
    return err;
}

struct GemmShape { const char* name; int M, N, K; };

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    GemmShape shapes[] = {
        {"QKV/Wo (2048x128x128)", 2048, 128, 128},
        {"FFN W1 (2048x512x128)", 2048, 512, 128},
        {"FFN W2 (2048x128x512)", 2048, 128, 512},
        {"Logits (2048x65x128)",  2048,  65, 128},
    };

    int warmup = 10, iters = 100;

    printf("================================================================\n");
    printf("  GEMM Tuning Sweep — All Transformer-Relevant Sizes\n");
    printf("================================================================\n\n");

    for (auto& s : shapes) {
        int M = s.M, N = s.N, K = s.K;
        float gflops = 2.0f * M * N * K / 1e9f;

        std::vector<float> hA(M * K), hB(K * N);
        rand_init(hA); rand_init(hB, 123);

        float *dA, *dB, *dC, *dRef;
        cudaMalloc(&dA, M * K * sizeof(float));
        cudaMalloc(&dB, K * N * sizeof(float));
        cudaMalloc(&dC, M * N * sizeof(float));
        cudaMalloc(&dRef, M * N * sizeof(float));
        cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        float ms_cublas = bench_cublas(handle, dA, dB, dRef, M, N, K, warmup, iters);

        printf("--- %s ---\n", s.name);
        printf("  cuBLAS: %.4f ms = %.0f GFLOP/s\n\n", ms_cublas, gflops / (ms_cublas / 1e3f));
        printf("  %-36s | %6s | %5s | %10s %10s | %6s | %s\n",
               "Config", "blocks", "thrd", "ms", "GFLOP/s", "vs cuB", "err");
        printf("  ------------------------------------+--------+-------+------------+------------+--------+-------\n");

        {
            for (int i = 0; i < warmup; ++i) gemm_tiled(dA, dB, dC, M, N, K);
            cudaDeviceSynchronize();
            GpuTimer timer; timer.tic();
            for (int i = 0; i < iters; ++i) gemm_tiled(dA, dB, dC, M, N, K);
            timer.toc();
            float ms = timer.elapsed_ms() / iters;
            float err = max_error(dC, dRef, M*N);
            printf("  %-36s | %6s | %5s | %10.4f %10.0f | %5.1f%% | %.1e\n",
                   "** PRODUCTION gemm_tiled **", "—", "256",
                   ms, gflops/(ms/1e3f), ms_cublas/ms*100.0f, err);
        }

        #define TEST_CONFIG(BM_,BN_,BK_,TM_,TN_,V4,LABEL) { \
            float ms = bench_config<BM_,BN_,BK_,TM_,TN_,V4>(dA, dB, dC, M, N, K, warmup, iters); \
            float err = max_error(dC, dRef, M*N); \
            int blocks = ((N+BN_-1)/BN_) * ((M+BM_-1)/BM_); \
            int threads = (BM_/TM_)*(BN_/TN_); \
            float ratio = ms_cublas / ms * 100.0f; \
            printf("  %-36s | %6d | %5d | %10.4f %10.0f | %5.1f%% | %.1e\n", \
                   LABEL, blocks, threads, ms, gflops/(ms/1e3f), ratio, err); \
        }

        TEST_CONFIG(64,64,16,4,4,false,"BM64 BN64 BK16 TM4 TN4")
        TEST_CONFIG(64,64,16,8,4,false,"BM64 BN64 BK16 TM8 TN4")
        TEST_CONFIG(64,64,8,8,4,false, "BM64 BN64 BK8  TM8 TN4")
        TEST_CONFIG(128,64,8,8,4,false,"BM128 BN64 BK8 TM8 TN4")
        TEST_CONFIG(128,64,16,8,4,false,"BM128 BN64 BK16 TM8 TN4")

        if (N % 4 == 0) {
            TEST_CONFIG(64,64,16,4,4,true, "BM64 BN64 BK16 TM4 TN4 vec4")
            TEST_CONFIG(64,64,16,8,4,true, "BM64 BN64 BK16 TM8 TN4 vec4")
            TEST_CONFIG(64,64,8,8,4,true,  "BM64 BN64 BK8  TM8 TN4 vec4")
            TEST_CONFIG(128,64,8,8,4,true, "BM128 BN64 BK8 TM8 TN4 vec4")
            TEST_CONFIG(128,64,16,8,4,true,"BM128 BN64 BK16 TM8 TN4 vec4")
        }

        #undef TEST_CONFIG

        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }

    cublasDestroy(handle);
    return 0;
}
