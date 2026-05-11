// Single-GPU CUDA code for the mini-Transformer.
#pragma once

#include "common.h"

void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_tiled_acc(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_tiled_AT(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_tiled_BT(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_tiled_BT_acc(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_tiled_AT_acc(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_splitk_AT_acc(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_batched(const float* A, const float* B, float* C,
                  int M, int N, int K, int batch,
                  int stride_a, int stride_b, int stride_c);
void gemm_batched_AT(const float* A, const float* B, float* C,
                     int M, int N, int K, int batch,
                     int stride_a, int stride_b, int stride_c);
void gemm_batched_BT(const float* A, const float* B, float* C,
                     int M, int N, int K, int batch,
                     int stride_a, int stride_b, int stride_c);

void attention_naive(const float* Q, const float* K, const float* V,
                     float* out, int BH, int seq, int head_dim);

void attention_tiled(const float* Q, const float* K, const float* V,
                     float* out, int BH, int seq, int head_dim);

void attention_gemm(const float* Q, const float* K, const float* V,
                    float* out, float* scratch, int BH, int seq, int head_dim);

void attention_backward(const float* Q, const float* K, const float* V,
                        const float* out, const float* grad_out,
                        float* grad_Q, float* grad_K, float* grad_V,
                        float* P_buf, float* dP_buf, float* dS_buf,
                        int BH, int seq, int head_dim);

void layernorm_forward(const float* x, const float* gamma, const float* beta,
                       float* out, int rows, int cols, float eps = 1e-5f);

void layernorm_forward_save(const float* x, const float* gamma, const float* beta,
                            float* out, float* mean, float* inv_std,
                            int rows, int cols, float eps = 1e-5f);

void residual_layernorm_forward_save(const float* a, const float* b,
                                     const float* gamma, const float* beta,
                                     float* residual_out, float* out,
                                     float* mean, float* inv_std,
                                     int rows, int cols, float eps = 1e-5f);

void layernorm_backward(const float* grad_out, const float* x,
                        const float* mean, const float* inv_std,
                        const float* gamma,
                        float* grad_x, float* dgamma, float* dbeta,
                        int rows, int cols);

void layernorm_backward_residual(const float* grad_out, const float* x,
                                 const float* mean, const float* inv_std,
                                 const float* gamma,
                                 float* grad_residual, float* dgamma, float* dbeta,
                                 int rows, int cols);

void relu_forward(const float* in, float* out, int n);
void relu_backward(const float* in, const float* grad_out, float* grad_in, int n);
void residual_add(const float* a, const float* b, float* out, int n);

void bias_add(float* out, const float* bias, int rows, int cols);

void bias_relu_forward(const float* in, const float* bias, float* out,
                       int rows, int cols);

void bias_backward(const float* grad, float* dbias, int rows, int cols);

void scale_buffer(float* buf, float scale, int n);

void buffer_norm_sq_acc(const float* buf, int n, float* d_result);

void compute_clip_scale(const float* d_norm_sq, float* d_clip_scale, float max_norm);

void softmax_forward(const float* x, float* out, int rows, int cols);

float cross_entropy_forward(const float* logits, const int* targets,
                            float* grad, int batch, int vocab);

void cross_entropy_forward_v2(const float* logits, const int* targets,
                               float* grad, float* d_losses, float* d_loss_out,
                               int batch, int vocab);

void embedding_forward(const float* table, const int* tokens,
                       float* out, int batch_seq, int dim);
void embedding_backward(const float* grad_out, const int* tokens,
                        float* grad_table, int batch_seq, int dim);
void pos_embedding_forward(const float* pos_table, float* embeddings,
                           int batch, int seq, int dim);
void pos_embedding_backward(const float* grad_out, float* grad_pos_table,
                            int batch, int seq, int dim);
