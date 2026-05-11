// Single-GPU CUDA code for the mini-Transformer.
#include "kernels.h"

__global__ void embedding_forward_kernel(const float* table, const int* tokens,
                                         float* out, int batch_seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq * dim) return;
    int token_idx = idx / dim;
    int d = idx % dim;
    out[idx] = table[tokens[token_idx] * dim + d];
}

void embedding_forward(const float* table, const int* tokens,
                       float* out, int batch_seq, int dim) {
    int n = batch_seq * dim;
    int block = 256;
    embedding_forward_kernel<<<(n + block - 1) / block, block>>>(
        table, tokens, out, batch_seq, dim);
    CUDA_CHECK_LAST();
}

__global__ void embedding_backward_kernel(const float* grad_out,
                                          const int* tokens,
                                          float* grad_table,
                                          int batch_seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq * dim) return;
    int token_idx = idx / dim;
    int d = idx % dim;
    atomicAdd(&grad_table[tokens[token_idx] * dim + d], grad_out[idx]);
}

void embedding_backward(const float* grad_out, const int* tokens,
                        float* grad_table, int batch_seq, int dim) {
    int n = batch_seq * dim;
    int block = 256;
    embedding_backward_kernel<<<(n + block - 1) / block, block>>>(
        grad_out, tokens, grad_table, batch_seq, dim);
    CUDA_CHECK_LAST();
}

__global__ void pos_embedding_kernel(const float* pos_table, float* embeddings,
                                     int batch, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    if (idx >= total) return;
    int t = (idx / dim) % seq;
    int d = idx % dim;
    embeddings[idx] += pos_table[t * dim + d];
}

void pos_embedding_forward(const float* pos_table, float* embeddings,
                           int batch, int seq, int dim) {
    int n = batch * seq * dim;
    int block = 256;
    pos_embedding_kernel<<<(n + block - 1) / block, block>>>(
        pos_table, embeddings, batch, seq, dim);
    CUDA_CHECK_LAST();
}

__global__ void pos_embedding_backward_kernel(const float* grad_out,
                                              float* grad_pos_table,
                                              int batch, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    if (idx >= total) return;
    int t = (idx / dim) % seq;
    int d = idx % dim;
    atomicAdd(&grad_pos_table[t * dim + d], grad_out[idx]);
}

void pos_embedding_backward(const float* grad_out, float* grad_pos_table,
                            int batch, int seq, int dim) {
    int n = batch * seq * dim;
    int block = 256;
    pos_embedding_backward_kernel<<<(n + block - 1) / block, block>>>(
        grad_out, grad_pos_table, batch, seq, dim);
    CUDA_CHECK_LAST();
}
