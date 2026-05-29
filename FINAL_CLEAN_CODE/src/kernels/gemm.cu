// gemm backends for custom cuda and cublas paths.
#include "kernels.h"
#include "common.h"
#include <cstdint>

#ifndef SPLITK_TARGET_BLOCKS
#define SPLITK_TARGET_BLOCKS 216
#endif

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstring>

namespace {

enum GemmBackend { kAuto = -1, kCustom = 0, kCublas = 1, kCublasTc = 2 };
bool g_cublas_initialized = false;
GemmBackend g_requested_backend = kAuto;
cublasHandle_t g_cublas_handle = nullptr;
cublasLtHandle_t g_cublas_lt_handle = nullptr;
void* g_cublas_lt_workspace = nullptr;
constexpr size_t kCublasLtWorkspaceBytes = 4 * 1024 * 1024;

#define CUBLAS_CHECK(call) do {                                                \
    cublasStatus_t _s = (call);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,     \
                (int)_s);                                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

const char* backend_name(GemmBackend backend) {
    switch (backend) {
        case kAuto: return "auto";
        case kCustom: return "custom";
        case kCublas: return "cublas";
        case kCublasTc: return "cublas_tc";
    }
    return "unknown";
}

GemmBackend requested_backend_id() {
    if (!g_cublas_initialized) {
        const char* env = std::getenv("CME213_GEMM_BACKEND");
        if (!env || std::strcmp(env, "") == 0 || std::strcmp(env, "auto") == 0) {
            g_requested_backend = kAuto;
        } else if (std::strcmp(env, "custom") == 0) {
            g_requested_backend = kCustom;
        } else if (std::strcmp(env, "cublas") == 0) {
            g_requested_backend = kCublas;
        } else if (std::strcmp(env, "cublas_tc") == 0) {
            g_requested_backend = kCublasTc;
        } else {
            fprintf(stderr,
                    "Unknown CME213_GEMM_BACKEND='%s'; falling back to auto\n",
                    env);
            g_requested_backend = kAuto;
        }
        g_cublas_initialized = true;
    }
    return g_requested_backend;
}

GemmBackend auto_backend_for_shape(int M, int N, int K, int batch) {
    (void)M; (void)N; (void)K; (void)batch;

    return kCublasTc;
}

GemmBackend effective_backend_for_shape(int M, int N, int K, int batch = 1) {
    GemmBackend requested = requested_backend_id();
    if (requested != kAuto)
        return requested;
    return auto_backend_for_shape(M, N, K, batch);
}

bool backend_uses_cublas(GemmBackend backend) {
    return backend == kCublas || backend == kCublasTc;
}

bool backend_uses_cublas_tc(GemmBackend backend) {
    return backend == kCublasTc;
}

bool auto_policy_can_use_cublas_tc() {
    GemmBackend requested = requested_backend_id();
    if (requested == kCublasTc) return true;
    if (requested != kAuto) return false;
    return auto_backend_for_shape(0, 0, 0, 1) == kCublasTc;
}

bool g_lt_fusion_set = false;
bool g_lt_fusion = false;
bool use_cublas_lt_fusion() {
    if (!g_lt_fusion_set) {
        if (!auto_policy_can_use_cublas_tc()) {
            g_lt_fusion = false;
        } else {
            const char* env = std::getenv("CME213_LT_FUSION");
            g_lt_fusion = (env && std::strcmp(env, "1") == 0);
        }
        g_lt_fusion_set = true;
    }
    return g_lt_fusion;
}

constexpr size_t kCublasWorkspaceBytes = 4 * 1024 * 1024;
void* g_cublas_workspace = nullptr;

cublasHandle_t cublas_handle() {
    if (!g_cublas_handle) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

        CUBLAS_CHECK(cublasSetStream(g_cublas_handle, cudaStreamPerThread));

        CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle,
                                        CUBLAS_TF32_TENSOR_OP_MATH));
        if (!g_cublas_workspace) {
            cudaMalloc(&g_cublas_workspace, kCublasWorkspaceBytes);
        }
        if (g_cublas_workspace) {

            cublasSetWorkspace(g_cublas_handle, g_cublas_workspace,
                                kCublasWorkspaceBytes);
        }
    }
    return g_cublas_handle;
}

cublasLtHandle_t cublas_lt_handle() {
    if (!g_cublas_lt_handle) {
        CUBLAS_CHECK(cublasLtCreate(&g_cublas_lt_handle));
        if (!g_cublas_lt_workspace) {
            cudaMalloc(&g_cublas_lt_workspace, kCublasLtWorkspaceBytes);
        }
    }
    return g_cublas_lt_handle;
}

struct LtCache {
    int M = -1, N = -1, K = -1;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t A_lt = nullptr;
    cublasLtMatrixLayout_t B_lt = nullptr;
    cublasLtMatrixLayout_t C_lt = nullptr;
    cublasLtMatmulHeuristicResult_t heur;
    bool heur_valid = false;
};
LtCache g_fwd_cache;
LtCache g_bwd_cache;

void lt_cache_reset(LtCache& c) {
    if (c.desc) { cublasLtMatmulDescDestroy(c.desc); c.desc = nullptr; }
    if (c.A_lt) { cublasLtMatrixLayoutDestroy(c.A_lt); c.A_lt = nullptr; }
    if (c.B_lt) { cublasLtMatrixLayoutDestroy(c.B_lt); c.B_lt = nullptr; }
    if (c.C_lt) { cublasLtMatrixLayoutDestroy(c.C_lt); c.C_lt = nullptr; }
    c.M = c.N = c.K = -1;
    c.heur_valid = false;
}

void lt_find_heuristic(LtCache& c) {
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = kCublasLtWorkspaceBytes;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(cublas_lt_handle(), c.desc,
                                    c.A_lt, c.B_lt, c.C_lt, c.C_lt,
                                    pref, 1, &c.heur, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (returned < 1) {
        fprintf(stderr,
                "cuBLASLt: no algo found for M=%d N=%d K=%d\n", c.M, c.N, c.K);
        exit(EXIT_FAILURE);
    }
    c.heur_valid = true;
}

void cublas_lt_NN_relu_aux_bias(const float* A, const float* B,
                                  const float* bias, float* C, void* mask,
                                  int M, int N, int K) {
    LtCache& c = g_fwd_cache;
    if (c.M != M || c.N != N || c.K != K) {
        lt_cache_reset(c);
        c.M = M; c.N = N; c.K = K;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &c.desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));
        cublasOperation_t op_n = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

        int64_t aux_ld = N;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.A_lt, CUDA_R_32F, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.B_lt, CUDA_R_32F, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.C_lt, CUDA_R_32F, N, M, N));
    }
    cublasLtMatmulDescSetAttribute(c.desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cublasLtMatmulDescSetAttribute(c.desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &mask, sizeof(mask));

    if (!c.heur_valid) lt_find_heuristic(c);

    static const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(cublas_lt_handle(), c.desc, &alpha,
                                  B, c.A_lt,
                                  A, c.B_lt,
                                  &beta,
                                  C, c.C_lt,
                                  C, c.C_lt,
                                  &c.heur.algo,
                                  g_cublas_lt_workspace,
                                  kCublasLtWorkspaceBytes,
                                  cudaStreamPerThread));
}

void cublas_lt_NT_drelu_bgrad(const float* A, const float* B,
                                const void* mask, float* C, float* dbias,
                                int M, int N, int K) {
    LtCache& c = g_bwd_cache;
    if (c.M != M || c.N != N || c.K != K) {
        lt_cache_reset(c);
        c.M = M; c.N = N; c.K = K;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &c.desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));
        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DRELU_BGRAD;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        int64_t aux_ld = N;
        cublasLtMatmulDescSetAttribute(c.desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.A_lt, CUDA_R_32F, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.B_lt, CUDA_R_32F, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c.C_lt, CUDA_R_32F, N, M, N));
    }
    cublasLtMatmulDescSetAttribute(c.desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias));
    cublasLtMatmulDescSetAttribute(c.desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &mask, sizeof(mask));

    if (!c.heur_valid) lt_find_heuristic(c);

    static const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(cublas_lt_handle(), c.desc, &alpha,
                                  B, c.A_lt,
                                  A, c.B_lt,
                                  &beta,
                                  C, c.C_lt,
                                  C, c.C_lt,
                                  &c.heur.algo,
                                  g_cublas_lt_workspace,
                                  kCublasLtWorkspaceBytes,
                                  cudaStreamPerThread));
}

inline cublasComputeType_t compute_type(GemmBackend backend) {

    const char* strict = getenv("CME213_STRICT_FP32");

    if (strict && atoi(strict) == 1) {
        return CUBLAS_COMPUTE_32F;
    }

    return backend_uses_cublas_tc(backend)
        ? CUBLAS_COMPUTE_32F_FAST_16F
        : CUBLAS_COMPUTE_32F;
}
inline cublasGemmAlgo_t gemm_algo(GemmBackend backend) {
    return backend_uses_cublas_tc(backend) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                           : CUBLAS_GEMM_DEFAULT;
}

void cublas_gemm_NN(const float* A, const float* B, float* C,
                     int M, int N, int K, float beta, GemmBackend backend) {

    static const float alpha = 1.0f;

    auto handle = cublas_handle();

    CUBLAS_CHECK(
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)
    );

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N,
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, N,
        compute_type(backend),
        gemm_algo(backend)
    ));
}

void cublas_gemm_TN(const float* A, const float* B, float* C,
                     int M, int N, int K, float beta, GemmBackend backend) {
    static const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                               CUBLAS_OP_N, CUBLAS_OP_T,
                               N, M, K,
                               &alpha,
                               B, CUDA_R_32F, N,
                               A, CUDA_R_32F, M,
                               &beta,
                               C, CUDA_R_32F, N,
                               compute_type(backend), gemm_algo(backend)));
}

void cublas_gemm_NT(const float* A, const float* B, float* C,
                     int M, int N, int K, float beta, GemmBackend backend) {
    static const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               B, CUDA_R_32F, K,
                               A, CUDA_R_32F, K,
                               &beta,
                               C, CUDA_R_32F, N,
                               compute_type(backend), gemm_algo(backend)));
}

void cublas_gemm_half_NN(const __half* A, const __half* B, float* C,
                         int M, int N, int K, float beta) {
    static const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               B, CUDA_R_16F, N,
                               A, CUDA_R_16F, K,
                               &beta,
                               C, CUDA_R_32F, N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cublas_gemm_half_TN(const __half* A, const __half* B, float* C,
                         int M, int N, int K, float beta) {
    static const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                               CUBLAS_OP_N, CUBLAS_OP_T,
                               N, M, K,
                               &alpha,
                               B, CUDA_R_16F, N,
                               A, CUDA_R_16F, M,
                               &beta,
                               C, CUDA_R_32F, N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cublas_gemm_half_NT(const __half* A, const __half* B, float* C,
                         int M, int N, int K, float beta) {
    static const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               B, CUDA_R_16F, K,
                               A, CUDA_R_16F, K,
                               &beta,
                               C, CUDA_R_32F, N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cublas_gemm_batched_NN(const float* A, const float* B, float* C,
                              int M, int N, int K, int batch,
                              int stride_a, int stride_b, int stride_c,
                              GemmBackend backend) {
    static const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_handle(),
                                              CUBLAS_OP_N, CUBLAS_OP_N,
                                              N, M, K,
                                              &alpha,
                                              B, CUDA_R_32F, N, stride_b,
                                              A, CUDA_R_32F, K, stride_a,
                                              &beta,
                                              C, CUDA_R_32F, N, stride_c,
                                              batch,
                                              compute_type(backend), gemm_algo(backend)));
}

void cublas_gemm_batched_TN(const float* A, const float* B, float* C,
                              int M, int N, int K, int batch,
                              int stride_a, int stride_b, int stride_c,
                              GemmBackend backend) {
    static const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_handle(),
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              N, M, K,
                                              &alpha,
                                              B, CUDA_R_32F, N, stride_b,
                                              A, CUDA_R_32F, M, stride_a,
                                              &beta,
                                              C, CUDA_R_32F, N, stride_c,
                                              batch,
                                              compute_type(backend), gemm_algo(backend)));
}

void cublas_gemm_batched_NT(const float* A, const float* B, float* C,
                              int M, int N, int K, int batch,
                              int stride_a, int stride_b, int stride_c,
                              GemmBackend backend) {
    static const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_handle(),
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              N, M, K,
                                              &alpha,
                                              B, CUDA_R_32F, K, stride_b,
                                              A, CUDA_R_32F, K, stride_a,
                                              &beta,
                                              C, CUDA_R_32F, N, stride_c,
                                              batch,
                                              compute_type(backend), gemm_algo(backend)));
}

}

bool gemm_uses_cublas_lt_fusion() { return use_cublas_lt_fusion(); }

const char* gemm_requested_backend_name() {
    return backend_name(requested_backend_id());
}

const char* gemm_auto_policy_name() {
    return "auto:cublas_tc_current_clean_benchmark";
}

void gemm_lt_relu_aux_bias(const float* A, const float* B, const float* bias,
                            float* C, void* mask, int M, int N, int K) {
    cublas_lt_NN_relu_aux_bias(A, B, bias, C, mask, M, N, K);
}

void gemm_lt_drelu_bgrad(const float* A, const float* B, const void* mask,
                          float* C, float* dbias, int M, int N, int K) {
    cublas_lt_NT_drelu_bgrad(A, B, mask, C, dbias, M, N, K);
}

void gemm_half_NN(const __half* A, const __half* B, float* C,
                  int M, int N, int K) {
    cublas_gemm_half_NN(A, B, C, M, N, K, 0.0f);
}

void gemm_half_NT(const __half* A, const __half* B, float* C,
                  int M, int N, int K) {
    cublas_gemm_half_NT(A, B, C, M, N, K, 0.0f);
}

void gemm_half_TN_acc(const __half* A, const __half* B, float* C,
                      int M, int N, int K) {
    cublas_gemm_half_TN(A, B, C, M, N, K, 1.0f);
}

bool gemm_lt_ffn_relu_bias_shape_ok(int batch_seq, int ff_dim, int hidden_dim) {
    (void)batch_seq;
    (void)hidden_dim;
    return ff_dim >= 128 && (ff_dim % 128) == 0;
}

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
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_NN(A, B, C, M, N, K, 0.0f, backend);
        return;
    }
    launch_gemm(A, B, C, M, N, K, false);
}

void gemm_tiled_acc(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_NN(A, B, C, M, N, K, 1.0f, backend);
        return;
    }
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
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_TN(A, B, C, M, N, K, 0.0f, backend);
        return;
    }
    launch_gemm_AT(A, B, C, M, N, K, false);
}

void gemm_tiled_AT_acc(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_TN(A, B, C, M, N, K, 1.0f, backend);
        return;
    }
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
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_TN(A, B, C, M, N, K, 1.0f, backend);
        return;
    }
    int grid_mn = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);

    int target_blocks = SPLITK_TARGET_BLOCKS;
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
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_NT(A, B, C, M, N, K, 0.0f, backend);
        return;
    }
    launch_gemm_BT(A, B, C, M, N, K, false);
}

void gemm_tiled_BT_acc(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    GemmBackend backend = effective_backend_for_shape(M, N, K);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_NT(A, B, C, M, N, K, 1.0f, backend);
        return;
    }
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
    GemmBackend backend = effective_backend_for_shape(M, N, K, batch);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_batched_NN(A, B, C, M, N, K, batch,
                               stride_a, stride_b, stride_c, backend);
        return;
    }
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
    GemmBackend backend = effective_backend_for_shape(M, N, K, batch);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_batched_TN(A, B, C, M, N, K, batch,
                               stride_a, stride_b, stride_c, backend);
        return;
    }
    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
    gemm_batched_AT_kernel<<<grid, block>>>(A, B, C, M, N, K, stride_a, stride_b, stride_c);
    CUDA_CHECK_LAST();
}

void gemm_batched_BT(const float* A, const float* B, float* C,
                     int M, int N, int K, int batch,
                     int stride_a, int stride_b, int stride_c) {
    GemmBackend backend = effective_backend_for_shape(M, N, K, batch);
    if (backend_uses_cublas(backend)) {
        cublas_gemm_batched_NT(A, B, C, M, N, K, batch,
                               stride_a, stride_b, stride_c, backend);
        return;
    }
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
