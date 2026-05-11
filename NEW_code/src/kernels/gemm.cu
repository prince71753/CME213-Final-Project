// Single-GPU CUDA code for the mini-Transformer.
#include "kernels.h"

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void gemm_naive(const float* A, const float* B, float* C,
                int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define NTHREADS ((BM/TM) * (BN/TN))

__global__ __launch_bounds__(NTHREADS)
void gemm_reg_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);

    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};

    const int num_k = (K + BK - 1) / BK;

    {
        const int k_off = 0;
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = brow + r, gc = k_off + c;
            sA[0][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = k_off + r, gc = bcol + c;
            sB[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < num_k) {
            const int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int r = idx / BK, c = idx % BK;
                int gr = brow + r, gc = k_off + c;
                sA[nxt][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int r = idx / BN, c = idx % BN;
                int gr = k_off + r, gc = bcol + c;
                sB[nxt][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            if (accumulate)
                C[grow * N + gcol] += rc[i][j];
            else
                C[grow * N + gcol] = rc[i][j];
        }
    }
}

#define BM2 64
#define BN2 64
#define BK2 16
#define TM2 8
#define TN2 4
#define NTHREADS2 ((BM2/TM2) * (BN2/TN2))

__global__ __launch_bounds__(NTHREADS2)
void gemm_v2_kernel(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK2][BM2 + 1];
    __shared__ float sB[2][BK2][BN2 + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN2 / TN2);
    const int ty   = tid / (BN2 / TN2);
    const int brow = blockIdx.y * BM2;
    const int bcol = blockIdx.x * BN2;

    float rc[TM2][TN2] = {};
    const int num_k = (K + BK2 - 1) / BK2;

    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BK2, c = elem % BK2;
        int gr = brow + r, gc = c;
        float4 v = (gr < M && gc + 3 < K)
            ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
            : make_float4(
                (gr < M && gc   < K) ? A[gr * K + gc]   : 0.0f,
                (gr < M && gc+1 < K) ? A[gr * K + gc+1] : 0.0f,
                (gr < M && gc+2 < K) ? A[gr * K + gc+2] : 0.0f,
                (gr < M && gc+3 < K) ? A[gr * K + gc+3] : 0.0f);
        sA[0][c][r] = v.x; sA[0][c+1][r] = v.y;
        sA[0][c+2][r] = v.z; sA[0][c+3][r] = v.w;
    }
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BN2, c = elem % BN2;
        int gr = r, gc = bcol + c;
        float4 v = (gr < K && gc + 3 < N)
            ? *reinterpret_cast<const float4*>(&B[gr * N + gc])
            : make_float4(
                (gr < K && gc   < N) ? B[gr * N + gc]   : 0.0f,
                (gr < K && gc+1 < N) ? B[gr * N + gc+1] : 0.0f,
                (gr < K && gc+2 < N) ? B[gr * N + gc+2] : 0.0f,
                (gr < K && gc+3 < N) ? B[gr * N + gc+3] : 0.0f);
        sB[0][r][c] = v.x; sB[0][r][c+1] = v.y;
        sB[0][r][c+2] = v.z; sB[0][r][c+3] = v.w;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;

        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK2;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BK2, c = elem % BK2;
                int gr = brow + r, gc = k_off + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
                    : make_float4(
                        (gr < M && gc   < K) ? A[gr * K + gc]   : 0.0f,
                        (gr < M && gc+1 < K) ? A[gr * K + gc+1] : 0.0f,
                        (gr < M && gc+2 < K) ? A[gr * K + gc+2] : 0.0f,
                        (gr < M && gc+3 < K) ? A[gr * K + gc+3] : 0.0f);
                sA[nxt][c][r] = v.x; sA[nxt][c+1][r] = v.y;
                sA[nxt][c+2][r] = v.z; sA[nxt][c+3][r] = v.w;
            }
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BN2, c = elem % BN2;
                int gr = k_off + r, gc = bcol + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *reinterpret_cast<const float4*>(&B[gr * N + gc])
                    : make_float4(
                        (gr < K && gc   < N) ? B[gr * N + gc]   : 0.0f,
                        (gr < K && gc+1 < N) ? B[gr * N + gc+1] : 0.0f,
                        (gr < K && gc+2 < N) ? B[gr * N + gc+2] : 0.0f,
                        (gr < K && gc+3 < N) ? B[gr * N + gc+3] : 0.0f);
                sB[nxt][r][c] = v.x; sB[nxt][r][c+1] = v.y;
                sB[nxt][r][c+2] = v.z; sB[nxt][r][c+3] = v.w;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK2; ++k) {
            float rA[TM2], rB[TN2];
            #pragma unroll
            for (int i = 0; i < TM2; ++i) rA[i] = sA[cur][k][ty * TM2 + i];
            #pragma unroll
            for (int j = 0; j < TN2; ++j) rB[j] = sB[cur][k][tx * TN2 + j];
            #pragma unroll
            for (int i = 0; i < TM2; ++i)
                #pragma unroll
                for (int j = 0; j < TN2; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM2; ++i) {
        int grow = brow + ty * TM2 + i;
        if (grow >= M) continue;
        int gcol = bcol + tx * TN2;
        if (gcol + TN2 - 1 < N) {
            if (accumulate) {
                float4 old = *reinterpret_cast<float4*>(&C[grow * N + gcol]);
                old.x += rc[i][0]; old.y += rc[i][1];
                old.z += rc[i][2]; old.w += rc[i][3];
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) = old;
            } else {
                float4 v = make_float4(rc[i][0], rc[i][1], rc[i][2], rc[i][3]);
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) = v;
            }
        } else {
            for (int j = 0; j < TN2; ++j) {
                int gc = gcol + j;
                if (gc < N) {
                    if (accumulate) C[grow * N + gc] += rc[i][j];
                    else            C[grow * N + gc]  = rc[i][j];
                }
            }
        }
    }
}

static void launch_gemm(const float* A, const float* B, float* C,
                         int M, int N, int K, bool acc) {
    if (K % 4 == 0 && N % 4 == 0) {
        dim3 block(NTHREADS2);
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        gemm_v2_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    } else {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_reg_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    }
    CUDA_CHECK_LAST();
}

void gemm_tiled(const float* A, const float* B, float* C,
                int M, int N, int K) {
    launch_gemm(A, B, C, M, N, K, false);
}

void gemm_tiled_acc(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    launch_gemm(A, B, C, M, N, K, true);
}

__global__ __launch_bounds__(NTHREADS)
void gemm_AT_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};
    const int num_k = (K + BK - 1) / BK;

    {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int m = idx % BM, k = idx / BM;
            int gm = brow + m, gk = k;
            sA[0][k][m] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = r, gc = bcol + c;
            sB[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int m = idx % BM, k = idx / BM;
                int gm = brow + m, gk = k_off + k;
                sA[nxt][k][m] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int r = idx / BN, c = idx % BN;
                int gr = k_off + r, gc = bcol + c;
                sB[nxt][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            if (accumulate) C[grow * N + gcol] += rc[i][j];
            else            C[grow * N + gcol]  = rc[i][j];
        }
    }
}

__global__ __launch_bounds__(NTHREADS2)
void gemm_v2_AT_kernel(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK2][BM2 + 1];
    __shared__ float sB[2][BK2][BN2 + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN2 / TN2);
    const int ty   = tid / (BN2 / TN2);
    const int brow = blockIdx.y * BM2;
    const int bcol = blockIdx.x * BN2;

    float rc[TM2][TN2] = {};
    const int num_k = (K + BK2 - 1) / BK2;

    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int m = elem % BM2, k = elem / BM2;
        int gm = brow + m, gk = k;
        float4 v = (gm + 3 < M && gk < K)
            ? *reinterpret_cast<const float4*>(&A[gk * M + gm])
            : make_float4(
                (gm   < M && gk < K) ? A[gk * M + gm]   : 0.0f,
                (gm+1 < M && gk < K) ? A[gk * M + gm+1] : 0.0f,
                (gm+2 < M && gk < K) ? A[gk * M + gm+2] : 0.0f,
                (gm+3 < M && gk < K) ? A[gk * M + gm+3] : 0.0f);
        sA[0][k][m] = v.x; sA[0][k][m+1] = v.y;
        sA[0][k][m+2] = v.z; sA[0][k][m+3] = v.w;
    }
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BN2, c = elem % BN2;
        int gr = r, gc = bcol + c;
        float4 v = (gr < K && gc + 3 < N)
            ? *reinterpret_cast<const float4*>(&B[gr * N + gc])
            : make_float4(
                (gr < K && gc   < N) ? B[gr * N + gc]   : 0.0f,
                (gr < K && gc+1 < N) ? B[gr * N + gc+1] : 0.0f,
                (gr < K && gc+2 < N) ? B[gr * N + gc+2] : 0.0f,
                (gr < K && gc+3 < N) ? B[gr * N + gc+3] : 0.0f);
        sB[0][r][c] = v.x; sB[0][r][c+1] = v.y;
        sB[0][r][c+2] = v.z; sB[0][r][c+3] = v.w;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK2;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int m = elem % BM2, k = elem / BM2;
                int gm = brow + m, gk = k_off + k;
                float4 v = (gm + 3 < M && gk < K)
                    ? *reinterpret_cast<const float4*>(&A[gk * M + gm])
                    : make_float4(
                        (gm   < M && gk < K) ? A[gk * M + gm]   : 0.0f,
                        (gm+1 < M && gk < K) ? A[gk * M + gm+1] : 0.0f,
                        (gm+2 < M && gk < K) ? A[gk * M + gm+2] : 0.0f,
                        (gm+3 < M && gk < K) ? A[gk * M + gm+3] : 0.0f);
                sA[nxt][k][m] = v.x; sA[nxt][k][m+1] = v.y;
                sA[nxt][k][m+2] = v.z; sA[nxt][k][m+3] = v.w;
            }
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BN2, c = elem % BN2;
                int gr = k_off + r, gc = bcol + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *reinterpret_cast<const float4*>(&B[gr * N + gc])
                    : make_float4(
                        (gr < K && gc   < N) ? B[gr * N + gc]   : 0.0f,
                        (gr < K && gc+1 < N) ? B[gr * N + gc+1] : 0.0f,
                        (gr < K && gc+2 < N) ? B[gr * N + gc+2] : 0.0f,
                        (gr < K && gc+3 < N) ? B[gr * N + gc+3] : 0.0f);
                sB[nxt][r][c] = v.x; sB[nxt][r][c+1] = v.y;
                sB[nxt][r][c+2] = v.z; sB[nxt][r][c+3] = v.w;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK2; ++k) {
            float rA[TM2], rB[TN2];
            #pragma unroll
            for (int i = 0; i < TM2; ++i) rA[i] = sA[cur][k][ty * TM2 + i];
            #pragma unroll
            for (int j = 0; j < TN2; ++j) rB[j] = sB[cur][k][tx * TN2 + j];
            #pragma unroll
            for (int i = 0; i < TM2; ++i)
                #pragma unroll
                for (int j = 0; j < TN2; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM2; ++i) {
        int grow = brow + ty * TM2 + i;
        if (grow >= M) continue;
        int gcol = bcol + tx * TN2;
        if (gcol + TN2 - 1 < N) {
            if (accumulate) {
                float4 old = *reinterpret_cast<float4*>(&C[grow * N + gcol]);
                old.x += rc[i][0]; old.y += rc[i][1];
                old.z += rc[i][2]; old.w += rc[i][3];
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) = old;
            } else {
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) =
                    make_float4(rc[i][0], rc[i][1], rc[i][2], rc[i][3]);
            }
        } else {
            for (int j = 0; j < TN2; ++j) {
                int gc = gcol + j;
                if (gc < N) {
                    if (accumulate) C[grow * N + gc] += rc[i][j];
                    else            C[grow * N + gc]  = rc[i][j];
                }
            }
        }
    }
}

static void launch_gemm_AT(const float* A, const float* B, float* C,
                            int M, int N, int K, bool acc) {
    if (M % 4 == 0 && N % 4 == 0) {
        dim3 block(NTHREADS2);
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        gemm_v2_AT_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    } else {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_AT_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    }
    CUDA_CHECK_LAST();
}

void gemm_tiled_AT(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    launch_gemm_AT(A, B, C, M, N, K, false);
}

void gemm_tiled_AT_acc(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    launch_gemm_AT(A, B, C, M, N, K, true);
}

__global__ __launch_bounds__(NTHREADS)
void gemm_splitk_AT_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K, int k_per_split) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    int k_start = blockIdx.z * k_per_split;
    int k_end   = min(k_start + k_per_split, K);

    float rc[TM][TN] = {};

    int first_kt = k_start / BK;
    int last_kt  = (k_end + BK - 1) / BK;
    if (first_kt >= last_kt) goto store;

    {
        int k_off = first_kt * BK;
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BM * BK) {
                int m = idx % BM, k = idx / BM;
                int gm = brow + m, gk = k_off + k;
                sA[0][k][m] = (gm < M && gk < k_end) ? A[gk * M + gm] : 0.0f;
            }
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BK * BN) {
                int r = idx / BN, c = idx % BN;
                int gr = k_off + r, gc = bcol + c;
                sB[0][r][c] = (gr < k_end && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = first_kt; kt < last_kt; ++kt) {
        int cur = (kt - first_kt) & 1;
        int nxt = 1 - cur;

        if (kt + 1 < last_kt) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BM * BK) {
                    int m = idx % BM, k = idx / BM;
                    int gm = brow + m, gk = k_off + k;
                    sA[nxt][k][m] = (gm < M && gk < k_end) ? A[gk * M + gm] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BK * BN) {
                    int r = idx / BN, c = idx % BN;
                    int gr = k_off + r, gc = bcol + c;
                    sB[nxt][r][c] = (gr < k_end && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

store:
    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            atomicAdd(&C[grow * N + gcol], rc[i][j]);
        }
    }
}

void gemm_splitk_AT_acc(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    int grid_mn = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);

    int target_blocks = 144;
    int num_splits = max(1, target_blocks / max(1, grid_mn));

    int k_per_split = ((K + num_splits - 1) / num_splits + BK - 1) / BK * BK;
    num_splits = (K + k_per_split - 1) / k_per_split;

    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, num_splits);
    gemm_splitk_AT_kernel<<<grid, block>>>(A, B, C, M, N, K, k_per_split);
    CUDA_CHECK_LAST();
}

__global__ __launch_bounds__(NTHREADS)
void gemm_BT_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};
    const int num_k = (K + BK - 1) / BK;

    {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = brow + r, gc = c;
            sA[0][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            int k = idx % BK, n = idx / BK;
            int gk = k, gn = bcol + n;
            sB[0][k][n] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int r = idx / BK, c = idx % BK;
                int gr = brow + r, gc = k_off + c;
                sA[nxt][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                int k = idx % BK, n = idx / BK;
                int gk = k_off + k, gn = bcol + n;
                sB[nxt][k][n] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            if (accumulate)
                C[grow * N + gcol] += rc[i][j];
            else
                C[grow * N + gcol] = rc[i][j];
        }
    }
}

__global__ __launch_bounds__(NTHREADS2)
void gemm_v2_BT_kernel(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K, bool accumulate) {
    __shared__ float sA[2][BK2][BM2 + 1];
    __shared__ float sB[2][BK2][BN2 + 1];

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN2 / TN2);
    const int ty   = tid / (BN2 / TN2);
    const int brow = blockIdx.y * BM2;
    const int bcol = blockIdx.x * BN2;

    float rc[TM2][TN2] = {};
    const int num_k = (K + BK2 - 1) / BK2;

    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BK2, c = elem % BK2;
        int gr = brow + r, gc = c;
        float4 v = (gr < M && gc + 3 < K)
            ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
            : make_float4(
                (gr < M && gc   < K) ? A[gr * K + gc]   : 0.0f,
                (gr < M && gc+1 < K) ? A[gr * K + gc+1] : 0.0f,
                (gr < M && gc+2 < K) ? A[gr * K + gc+2] : 0.0f,
                (gr < M && gc+3 < K) ? A[gr * K + gc+3] : 0.0f);
        sA[0][c][r] = v.x; sA[0][c+1][r] = v.y;
        sA[0][c+2][r] = v.z; sA[0][c+3][r] = v.w;
    }
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int k = elem % BK2, n = elem / BK2;
        int gk = k, gn = bcol + n;
        float4 v = (gn < N && gk + 3 < K)
            ? *reinterpret_cast<const float4*>(&B[gn * K + gk])
            : make_float4(
                (gn < N && gk   < K) ? B[gn * K + gk]   : 0.0f,
                (gn < N && gk+1 < K) ? B[gn * K + gk+1] : 0.0f,
                (gn < N && gk+2 < K) ? B[gn * K + gk+2] : 0.0f,
                (gn < N && gk+3 < K) ? B[gn * K + gk+3] : 0.0f);
        sB[0][k][n] = v.x; sB[0][k+1][n] = v.y;
        sB[0][k+2][n] = v.z; sB[0][k+3][n] = v.w;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK2;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BK2, c = elem % BK2;
                int gr = brow + r, gc = k_off + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
                    : make_float4(
                        (gr < M && gc   < K) ? A[gr * K + gc]   : 0.0f,
                        (gr < M && gc+1 < K) ? A[gr * K + gc+1] : 0.0f,
                        (gr < M && gc+2 < K) ? A[gr * K + gc+2] : 0.0f,
                        (gr < M && gc+3 < K) ? A[gr * K + gc+3] : 0.0f);
                sA[nxt][c][r] = v.x; sA[nxt][c+1][r] = v.y;
                sA[nxt][c+2][r] = v.z; sA[nxt][c+3][r] = v.w;
            }
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int k = elem % BK2, n = elem / BK2;
                int gk = k_off + k, gn = bcol + n;
                float4 v = (gn < N && gk + 3 < K)
                    ? *reinterpret_cast<const float4*>(&B[gn * K + gk])
                    : make_float4(
                        (gn < N && gk   < K) ? B[gn * K + gk]   : 0.0f,
                        (gn < N && gk+1 < K) ? B[gn * K + gk+1] : 0.0f,
                        (gn < N && gk+2 < K) ? B[gn * K + gk+2] : 0.0f,
                        (gn < N && gk+3 < K) ? B[gn * K + gk+3] : 0.0f);
                sB[nxt][k][n] = v.x; sB[nxt][k+1][n] = v.y;
                sB[nxt][k+2][n] = v.z; sB[nxt][k+3][n] = v.w;
            }
        }

        #pragma unroll
        for (int k = 0; k < BK2; ++k) {
            float rA[TM2], rB[TN2];
            #pragma unroll
            for (int i = 0; i < TM2; ++i) rA[i] = sA[cur][k][ty * TM2 + i];
            #pragma unroll
            for (int j = 0; j < TN2; ++j) rB[j] = sB[cur][k][tx * TN2 + j];
            #pragma unroll
            for (int i = 0; i < TM2; ++i)
                #pragma unroll
                for (int j = 0; j < TN2; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM2; ++i) {
        int grow = brow + ty * TM2 + i;
        if (grow >= M) continue;
        int gcol = bcol + tx * TN2;
        if (gcol + TN2 - 1 < N) {
            if (accumulate) {
                float4 old = *reinterpret_cast<float4*>(&C[grow * N + gcol]);
                old.x += rc[i][0]; old.y += rc[i][1];
                old.z += rc[i][2]; old.w += rc[i][3];
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) = old;
            } else {
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) =
                    make_float4(rc[i][0], rc[i][1], rc[i][2], rc[i][3]);
            }
        } else {
            for (int j = 0; j < TN2; ++j) {
                int gc = gcol + j;
                if (gc < N) {
                    if (accumulate) C[grow * N + gc] += rc[i][j];
                    else            C[grow * N + gc]  = rc[i][j];
                }
            }
        }
    }
}

static void launch_gemm_BT(const float* A, const float* B, float* C,
                            int M, int N, int K, bool acc) {
    if (K % 4 == 0 && N % 4 == 0) {
        dim3 block(NTHREADS2);
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        gemm_v2_BT_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    } else {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_BT_kernel<<<grid, block>>>(A, B, C, M, N, K, acc);
    }
    CUDA_CHECK_LAST();
}

void gemm_tiled_BT(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    launch_gemm_BT(A, B, C, M, N, K, false);
}

void gemm_tiled_BT_acc(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    launch_gemm_BT(A, B, C, M, N, K, true);
}

__global__ __launch_bounds__(NTHREADS)
void gemm_batched_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K,
                          int stride_a, int stride_b, int stride_c) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int batch = blockIdx.z;
    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float* Cb       = C + batch * stride_c;

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};
    const int num_k = (K + BK - 1) / BK;

    {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BM * BK) {
                int r = idx / BK, c = idx % BK;
                int gr = brow + r, gc = c;
                sA[0][c][r] = (gr < M && gc < K) ? Ab[gr * K + gc] : 0.0f;
            }
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BK * BN) {
                int r = idx / BN, c = idx % BN;
                int gr = r, gc = bcol + c;
                sB[0][r][c] = (gr < K && gc < N) ? Bb[gr * N + gc] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BM * BK) {
                    int r = idx / BK, c = idx % BK;
                    int gr = brow + r, gc = k_off + c;
                    sA[nxt][c][r] = (gr < M && gc < K) ? Ab[gr * K + gc] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BK * BN) {
                    int r = idx / BN, c = idx % BN;
                    int gr = k_off + r, gc = bcol + c;
                    sB[nxt][r][c] = (gr < K && gc < N) ? Bb[gr * N + gc] : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            Cb[grow * N + gcol] = rc[i][j];
        }
    }
}

__global__ __launch_bounds__(NTHREADS)
void gemm_batched_AT_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K,
                             int stride_a, int stride_b, int stride_c) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int batch = blockIdx.z;
    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float* Cb       = C + batch * stride_c;

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};
    const int num_k = (K + BK - 1) / BK;

    {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BM * BK) {
                int m = idx % BM, k = idx / BM;
                int gm = brow + m, gk = k;
                sA[0][k][m] = (gm < M && gk < K) ? Ab[gk * M + gm] : 0.0f;
            }
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BK * BN) {
                int r = idx / BN, c = idx % BN;
                int gr = r, gc = bcol + c;
                sB[0][r][c] = (gr < K && gc < N) ? Bb[gr * N + gc] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BM * BK) {
                    int m = idx % BM, k = idx / BM;
                    int gm = brow + m, gk = k_off + k;
                    sA[nxt][k][m] = (gm < M && gk < K) ? Ab[gk * M + gm] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BK * BN) {
                    int r = idx / BN, c = idx % BN;
                    int gr = k_off + r, gc = bcol + c;
                    sB[nxt][r][c] = (gr < K && gc < N) ? Bb[gr * N + gc] : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            Cb[grow * N + gcol] = rc[i][j];
        }
    }
}

__global__ __launch_bounds__(NTHREADS)
void gemm_batched_BT_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int stride_a, int stride_b, int stride_c) {
    __shared__ float sA[2][BK][BM + 1];
    __shared__ float sB[2][BK][BN + 1];

    const int batch = blockIdx.z;
    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float* Cb       = C + batch * stride_c;

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN / TN);
    const int ty   = tid / (BN / TN);
    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    float rc[TM][TN] = {};
    const int num_k = (K + BK - 1) / BK;

    {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BM * BK) {
                int r = idx / BK, c = idx % BK;
                int gr = brow + r, gc = c;
                sA[0][c][r] = (gr < M && gc < K) ? Ab[gr * K + gc] : 0.0f;
            }
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
            int idx = tid + l * NTHREADS;
            if (idx < BK * BN) {
                int k = idx % BK, n = idx / BK;
                int gk = k, gn = bcol + n;
                sB[0][k][n] = (gk < K && gn < N) ? Bb[gn * K + gk] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1;
        int nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK;
            #pragma unroll
            for (int l = 0; l < (BM * BK) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BM * BK) {
                    int r = idx / BK, c = idx % BK;
                    int gr = brow + r, gc = k_off + c;
                    sA[nxt][c][r] = (gr < M && gc < K) ? Ab[gr * K + gc] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < (BK * BN) / NTHREADS; ++l) {
                int idx = tid + l * NTHREADS;
                if (idx < BK * BN) {
                    int k = idx % BK, n = idx / BK;
                    int gk = k_off + k, gn = bcol + n;
                    sB[nxt][k][n] = (gk < K && gn < N) ? Bb[gn * K + gk] : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[cur][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[cur][k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int grow = brow + ty * TM + i;
        if (grow >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gcol = bcol + tx * TN + j;
            if (gcol >= N) continue;
            Cb[grow * N + gcol] = rc[i][j];
        }
    }
}

__global__ __launch_bounds__(NTHREADS2)
void gemm_v2_batched_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K,
                             int stride_a, int stride_b, int stride_c) {
    __shared__ float sA[2][BK2][BM2 + 1];
    __shared__ float sB[2][BK2][BN2 + 1];

    const int batch = blockIdx.z;
    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float* Cb       = C + batch * stride_c;

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN2 / TN2);
    const int ty   = tid / (BN2 / TN2);
    const int brow = blockIdx.y * BM2;
    const int bcol = blockIdx.x * BN2;

    float rc[TM2][TN2] = {};
    const int num_k = (K + BK2 - 1) / BK2;

    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BK2, c = elem % BK2;
        int gr = brow + r, gc = c;
        float4 v = (gr < M && gc + 3 < K)
            ? *reinterpret_cast<const float4*>(&Ab[gr * K + gc])
            : make_float4(
                (gr < M && gc   < K) ? Ab[gr * K + gc]   : 0.0f,
                (gr < M && gc+1 < K) ? Ab[gr * K + gc+1] : 0.0f,
                (gr < M && gc+2 < K) ? Ab[gr * K + gc+2] : 0.0f,
                (gr < M && gc+3 < K) ? Ab[gr * K + gc+3] : 0.0f);
        sA[0][c][r] = v.x; sA[0][c+1][r] = v.y;
        sA[0][c+2][r] = v.z; sA[0][c+3][r] = v.w;
    }
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BN2, c = elem % BN2;
        int gr = r, gc = bcol + c;
        float4 v = (gr < K && gc + 3 < N)
            ? *reinterpret_cast<const float4*>(&Bb[gr * N + gc])
            : make_float4(
                (gr < K && gc   < N) ? Bb[gr * N + gc]   : 0.0f,
                (gr < K && gc+1 < N) ? Bb[gr * N + gc+1] : 0.0f,
                (gr < K && gc+2 < N) ? Bb[gr * N + gc+2] : 0.0f,
                (gr < K && gc+3 < N) ? Bb[gr * N + gc+3] : 0.0f);
        sB[0][r][c] = v.x; sB[0][r][c+1] = v.y;
        sB[0][r][c+2] = v.z; sB[0][r][c+3] = v.w;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK2;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BK2, c = elem % BK2;
                int gr = brow + r, gc = k_off + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *reinterpret_cast<const float4*>(&Ab[gr * K + gc])
                    : make_float4(
                        (gr < M && gc   < K) ? Ab[gr * K + gc]   : 0.0f,
                        (gr < M && gc+1 < K) ? Ab[gr * K + gc+1] : 0.0f,
                        (gr < M && gc+2 < K) ? Ab[gr * K + gc+2] : 0.0f,
                        (gr < M && gc+3 < K) ? Ab[gr * K + gc+3] : 0.0f);
                sA[nxt][c][r] = v.x; sA[nxt][c+1][r] = v.y;
                sA[nxt][c+2][r] = v.z; sA[nxt][c+3][r] = v.w;
            }
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BN2, c = elem % BN2;
                int gr = k_off + r, gc = bcol + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *reinterpret_cast<const float4*>(&Bb[gr * N + gc])
                    : make_float4(
                        (gr < K && gc   < N) ? Bb[gr * N + gc]   : 0.0f,
                        (gr < K && gc+1 < N) ? Bb[gr * N + gc+1] : 0.0f,
                        (gr < K && gc+2 < N) ? Bb[gr * N + gc+2] : 0.0f,
                        (gr < K && gc+3 < N) ? Bb[gr * N + gc+3] : 0.0f);
                sB[nxt][r][c] = v.x; sB[nxt][r][c+1] = v.y;
                sB[nxt][r][c+2] = v.z; sB[nxt][r][c+3] = v.w;
            }
        }
        #pragma unroll
        for (int k = 0; k < BK2; ++k) {
            float rA[TM2], rB[TN2];
            #pragma unroll
            for (int i = 0; i < TM2; ++i) rA[i] = sA[cur][k][ty * TM2 + i];
            #pragma unroll
            for (int j = 0; j < TN2; ++j) rB[j] = sB[cur][k][tx * TN2 + j];
            #pragma unroll
            for (int i = 0; i < TM2; ++i)
                #pragma unroll
                for (int j = 0; j < TN2; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM2; ++i) {
        int grow = brow + ty * TM2 + i;
        if (grow >= M) continue;
        int gcol = bcol + tx * TN2;
        if (gcol + TN2 - 1 < N) {
            *reinterpret_cast<float4*>(&Cb[grow * N + gcol]) =
                make_float4(rc[i][0], rc[i][1], rc[i][2], rc[i][3]);
        } else {
            for (int j = 0; j < TN2; ++j) {
                int gc = gcol + j;
                if (gc < N) Cb[grow * N + gc] = rc[i][j];
            }
        }
    }
}

__global__ __launch_bounds__(NTHREADS2)
void gemm_v2_batched_BT_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                int stride_a, int stride_b, int stride_c) {
    __shared__ float sA[2][BK2][BM2 + 1];
    __shared__ float sB[2][BK2][BN2 + 1];

    const int batch = blockIdx.z;
    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float* Cb       = C + batch * stride_c;

    const int tid  = threadIdx.x;
    const int tx   = tid % (BN2 / TN2);
    const int ty   = tid / (BN2 / TN2);
    const int brow = blockIdx.y * BM2;
    const int bcol = blockIdx.x * BN2;

    float rc[TM2][TN2] = {};
    const int num_k = (K + BK2 - 1) / BK2;

    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int r = elem / BK2, c = elem % BK2;
        int gr = brow + r, gc = c;
        float4 v = (gr < M && gc + 3 < K)
            ? *reinterpret_cast<const float4*>(&Ab[gr * K + gc])
            : make_float4(
                (gr < M && gc   < K) ? Ab[gr * K + gc]   : 0.0f,
                (gr < M && gc+1 < K) ? Ab[gr * K + gc+1] : 0.0f,
                (gr < M && gc+2 < K) ? Ab[gr * K + gc+2] : 0.0f,
                (gr < M && gc+3 < K) ? Ab[gr * K + gc+3] : 0.0f);
        sA[0][c][r] = v.x; sA[0][c+1][r] = v.y;
        sA[0][c+2][r] = v.z; sA[0][c+3][r] = v.w;
    }
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int vid = tid + l * NTHREADS2;
        int elem = vid * 4;
        int k = elem % BK2, n = elem / BK2;
        int gk = k, gn = bcol + n;
        float4 v = (gn < N && gk + 3 < K)
            ? *reinterpret_cast<const float4*>(&Bb[gn * K + gk])
            : make_float4(
                (gn < N && gk   < K) ? Bb[gn * K + gk]   : 0.0f,
                (gn < N && gk+1 < K) ? Bb[gn * K + gk+1] : 0.0f,
                (gn < N && gk+2 < K) ? Bb[gn * K + gk+2] : 0.0f,
                (gn < N && gk+3 < K) ? Bb[gn * K + gk+3] : 0.0f);
        sB[0][k][n] = v.x; sB[0][k+1][n] = v.y;
        sB[0][k+2][n] = v.z; sB[0][k+3][n] = v.w;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt & 1, nxt = 1 - cur;
        if (kt + 1 < num_k) {
            int k_off = (kt + 1) * BK2;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int r = elem / BK2, c = elem % BK2;
                int gr = brow + r, gc = k_off + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *reinterpret_cast<const float4*>(&Ab[gr * K + gc])
                    : make_float4(
                        (gr < M && gc   < K) ? Ab[gr * K + gc]   : 0.0f,
                        (gr < M && gc+1 < K) ? Ab[gr * K + gc+1] : 0.0f,
                        (gr < M && gc+2 < K) ? Ab[gr * K + gc+2] : 0.0f,
                        (gr < M && gc+3 < K) ? Ab[gr * K + gc+3] : 0.0f);
                sA[nxt][c][r] = v.x; sA[nxt][c+1][r] = v.y;
                sA[nxt][c+2][r] = v.z; sA[nxt][c+3][r] = v.w;
            }
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int vid = tid + l * NTHREADS2;
                int elem = vid * 4;
                int k = elem % BK2, n = elem / BK2;
                int gk = k_off + k, gn = bcol + n;
                float4 v = (gn < N && gk + 3 < K)
                    ? *reinterpret_cast<const float4*>(&Bb[gn * K + gk])
                    : make_float4(
                        (gn < N && gk   < K) ? Bb[gn * K + gk]   : 0.0f,
                        (gn < N && gk+1 < K) ? Bb[gn * K + gk+1] : 0.0f,
                        (gn < N && gk+2 < K) ? Bb[gn * K + gk+2] : 0.0f,
                        (gn < N && gk+3 < K) ? Bb[gn * K + gk+3] : 0.0f);
                sB[nxt][k][n] = v.x; sB[nxt][k+1][n] = v.y;
                sB[nxt][k+2][n] = v.z; sB[nxt][k+3][n] = v.w;
            }
        }
        #pragma unroll
        for (int k = 0; k < BK2; ++k) {
            float rA[TM2], rB[TN2];
            #pragma unroll
            for (int i = 0; i < TM2; ++i) rA[i] = sA[cur][k][ty * TM2 + i];
            #pragma unroll
            for (int j = 0; j < TN2; ++j) rB[j] = sB[cur][k][tx * TN2 + j];
            #pragma unroll
            for (int i = 0; i < TM2; ++i)
                #pragma unroll
                for (int j = 0; j < TN2; ++j)
                    rc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM2; ++i) {
        int grow = brow + ty * TM2 + i;
        if (grow >= M) continue;
        int gcol = bcol + tx * TN2;
        if (gcol + TN2 - 1 < N) {
            *reinterpret_cast<float4*>(&Cb[grow * N + gcol]) =
                make_float4(rc[i][0], rc[i][1], rc[i][2], rc[i][3]);
        } else {
            for (int j = 0; j < TN2; ++j) {
                int gc = gcol + j;
                if (gc < N) Cb[grow * N + gc] = rc[i][j];
            }
        }
    }
}

void gemm_batched(const float* A, const float* B, float* C,
                  int M, int N, int K, int batch,
                  int stride_a, int stride_b, int stride_c) {
    if (K % 4 == 0 && N % 4 == 0) {
        dim3 block(NTHREADS2);
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2, batch);
        gemm_v2_batched_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    } else {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
        gemm_batched_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    }
    CUDA_CHECK_LAST();
}

void gemm_batched_AT(const float* A, const float* B, float* C,
                     int M, int N, int K, int batch,
                     int stride_a, int stride_b, int stride_c) {
    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
    gemm_batched_AT_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    CUDA_CHECK_LAST();
}

void gemm_batched_BT(const float* A, const float* B, float* C,
                     int M, int N, int K, int batch,
                     int stride_a, int stride_b, int stride_c) {
    if (K % 4 == 0 && N % 4 == 0) {
        dim3 block(NTHREADS2);
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2, batch);
        gemm_v2_batched_BT_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    } else {
        dim3 block(NTHREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
        gemm_batched_BT_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    }
    CUDA_CHECK_LAST();
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN
#undef NTHREADS
#undef BM2
#undef BN2
#undef BK2
#undef TM2
#undef TN2
#undef NTHREADS2
