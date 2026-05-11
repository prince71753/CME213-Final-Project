// Single-GPU CUDA code for the mini-Transformer.
#include "kernels.h"
#include <cfloat>

__global__ void attention_naive_kernel(const float* Q, const float* K,
                                       const float* V, float* out,
                                       int seq, int hd) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq) return;

    float scale = 1.0f / sqrtf((float)hd);
    int base = bh * seq * hd;

    extern __shared__ float smem[];
    float* scores = smem;

    float max_val = -FLT_MAX;
    for (int j = 0; j < seq; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d)
            dot += Q[base + row * hd + d] * K[base + j * hd + d];
        dot *= scale;
        scores[j] = dot;
        if (dot > max_val) max_val = dot;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq; ++j) {
        scores[j] = expf(scores[j] - max_val);
        sum_exp += scores[j];
    }
    for (int j = 0; j < seq; ++j)
        scores[j] /= sum_exp;

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < hd) {
        float val = 0.0f;
        for (int j = 0; j < seq; ++j)
            val += scores[j] * V[base + j * hd + d];
        out[base + row * hd + d] = val;
    }
}

void attention_naive(const float* Q, const float* K, const float* V,
                     float* out, int BH, int seq, int head_dim) {
    dim3 block(head_dim, 1);
    dim3 grid(1, seq, BH);
    size_t smem_bytes = seq * sizeof(float);
    attention_naive_kernel<<<grid, block, smem_bytes>>>(Q, K, V, out, seq, head_dim);
    CUDA_CHECK_LAST();
}

#define ATTN_TILE 32

__global__ void attention_tiled_kernel(const float* Q, const float* K,
                                       const float* V, float* out,
                                       int seq, int hd) {
    int bh = blockIdx.z;
    int tile_row = blockIdx.y;
    int qi = tile_row * ATTN_TILE + threadIdx.y;

    float scale = 1.0f / sqrtf((float)hd);

    extern __shared__ float shared[];
    float* sK = shared;
    float* sV = sK + ATTN_TILE * hd;

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_acc[128];
    for (int d = 0; d < hd; ++d) o_acc[d] = 0.0f;

    int num_kv_tiles = (seq + ATTN_TILE - 1) / ATTN_TILE;
    int base_qkv = bh * seq * hd;

    for (int tile_k = 0; tile_k < num_kv_tiles; ++tile_k) {
        int kj = tile_k * ATTN_TILE + threadIdx.y;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            sK[threadIdx.y * hd + d] =
                (kj < seq) ? K[base_qkv + kj * hd + d] : 0.0f;
            sV[threadIdx.y * hd + d] =
                (kj < seq) ? V[base_qkv + kj * hd + d] : 0.0f;
        }
        __syncthreads();

        if (qi < seq) {
            float m_new = m_prev;
            int tile_len = min(ATTN_TILE, seq - tile_k * ATTN_TILE);

            for (int j = 0; j < tile_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += Q[base_qkv + qi * hd + d] * sK[j * hd + d];
                dot *= scale;
                if (dot > m_new) m_new = dot;
            }

            float correction = expf(m_prev - m_new);
            float l_new = l_prev * correction;
            for (int d = 0; d < hd; ++d)
                o_acc[d] *= correction;

            for (int j = 0; j < tile_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += Q[base_qkv + qi * hd + d] * sK[j * hd + d];
                dot *= scale;
                float w = expf(dot - m_new);
                l_new += w;
                for (int d = 0; d < hd; ++d)
                    o_acc[d] += w * sV[j * hd + d];
            }

            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();
    }

    if (qi < seq) {
        float inv_l = 1.0f / l_prev;
        for (int d = 0; d < hd; ++d)
            out[base_qkv + qi * hd + d] = o_acc[d] * inv_l;
    }
}

void attention_tiled(const float* Q, const float* K, const float* V,
                     float* out, int BH, int seq, int head_dim) {
    int num_q_tiles = (seq + ATTN_TILE - 1) / ATTN_TILE;
    dim3 block(1, ATTN_TILE);
    dim3 grid(1, num_q_tiles, BH);
    size_t smem = 2 * ATTN_TILE * head_dim * sizeof(float);
    attention_tiled_kernel<<<grid, block, smem>>>(Q, K, V, out, seq, head_dim);
    CUDA_CHECK_LAST();
}

__global__ void scale_softmax_kernel(float* S, int seq, float scale) {
    int bh  = blockIdx.z;
    int row = blockIdx.y;
    if (row >= seq) return;
    int base = bh * seq * seq + row * seq;

    extern __shared__ float smem[];

    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float v = S[base + j] * scale;
        S[base + j] = v;
        if (v > max_val) max_val = v;
    }
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float e = expf(S[base + j] - row_max);
        S[base + j] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float sum = smem[0];

    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int j = threadIdx.x; j < seq; j += blockDim.x)
        S[base + j] *= inv_sum;
}

void attention_gemm(const float* Q, const float* K, const float* V,
                    float* out, float* scratch, int BH, int seq, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int s_qk = seq * head_dim;
    int s_p  = seq * seq;

    gemm_batched_BT(Q, K, scratch, seq, seq, head_dim, BH, s_qk, s_qk, s_p);

    {
        int threads = 64;
        while (threads < seq && threads < 256) threads *= 2;
        dim3 block(threads);
        dim3 grid(1, seq, BH);
        size_t smem_bytes = threads * sizeof(float);
        scale_softmax_kernel<<<grid, block, smem_bytes>>>(scratch, seq, scale);
        CUDA_CHECK_LAST();
    }

    gemm_batched(scratch, V, out, seq, head_dim, seq, BH, s_p, s_qk, s_qk);
}

__global__ void compute_attn_probs_kernel(const float* Q, const float* K,
                                           float* P, int seq, int hd) {
    int bh = blockIdx.z;
    int row = blockIdx.y;
    if (row >= seq) return;

    float scale = 1.0f / sqrtf((float)hd);
    int base = bh * seq * hd;
    int p_base = bh * seq * seq;

    const float* q_row = Q + base + row * hd;

    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d)
            dot += q_row[d] * K[base + j * hd + d];
        dot *= scale;
        P[p_base + row * seq + j] = dot;
        if (dot > max_val) max_val = dot;
    }

    extern __shared__ float smem[];
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float e = expf(P[p_base + row * seq + j] - row_max);
        P[p_base + row * seq + j] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float sum = smem[0];

    for (int j = threadIdx.x; j < seq; j += blockDim.x)
        P[p_base + row * seq + j] /= sum;
}

__global__ void compute_ds_kernel(const float* P, const float* dP,
                                  const float* dO, const float* O,
                                  float* dS, int seq, int hd) {
    int bh = blockIdx.z;
    int row = blockIdx.y;
    if (row >= seq) return;

    float scale = 1.0f / sqrtf((float)hd);
    int base = bh * seq * hd;
    int s_base = bh * seq * seq;

    float dot_oo = 0.0f;
    for (int d = 0; d < hd; ++d)
        dot_oo += dO[base + row * hd + d] * O[base + row * hd + d];

    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float p = P[s_base + row * seq + j];
        float dp = dP[s_base + row * seq + j];
        dS[s_base + row * seq + j] = scale * p * (dp - dot_oo);
    }
}

void attention_backward(const float* Q, const float* K, const float* V,
                        const float* out, const float* grad_out,
                        float* grad_Q, float* grad_K, float* grad_V,
                        float* d_P, float* d_dP, float* d_dS,
                        int BH, int seq, int head_dim) {

    gemm_batched_AT(d_P, grad_out, grad_V,
                    seq, head_dim, seq, BH,
                    seq * seq, seq * head_dim, seq * head_dim);

    gemm_batched_BT(grad_out, V, d_dP,
                    seq, seq, head_dim, BH,
                    seq * head_dim, seq * head_dim, seq * seq);

    {
        int threads = min(64, seq);
        dim3 block(threads);
        dim3 grid(1, seq, BH);
        compute_ds_kernel<<<grid, block>>>(d_P, d_dP, grad_out, out,
                                           d_dS, seq, head_dim);
        CUDA_CHECK_LAST();
    }

    gemm_batched(d_dS, K, grad_Q,
                 seq, head_dim, seq, BH,
                 seq * seq, seq * head_dim, seq * head_dim);

    gemm_batched_AT(d_dS, Q, grad_K,
                    seq, head_dim, seq, BH,
                    seq * seq, seq * head_dim, seq * head_dim);
}
