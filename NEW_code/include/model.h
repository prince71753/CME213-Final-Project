// Single-GPU CUDA code for the mini-Transformer.
#pragma once

#include "common.h"
#include "tensor.h"
#include "kernels.h"

struct DistributedContext;

struct TransformerConfig {
    int vocab_size  = 65;
    int seq_len     = SEQ_LEN;
    int hidden_dim  = HIDDEN_DIM;
    int num_heads   = NUM_HEADS;
    int head_dim    = HEAD_DIM;
    int ff_dim      = FF_DIM;
    int batch_size  = BATCH_SIZE;
    float lr        = 3e-4f;
    float eps       = 1e-5f;
};

struct TransformerBlockParams {
    Tensor Wq, Wk, Wv, Wo;
    Tensor dWq, dWk, dWv, dWo;

    Tensor ln1_gamma, ln1_beta;
    Tensor dln1_gamma, dln1_beta;

    Tensor W1, b1, W2, b2;
    Tensor dW1, db1, dW2, db2;

    Tensor ln2_gamma, ln2_beta;
    Tensor dln2_gamma, dln2_beta;

    void allocate(const TransformerConfig& cfg);
    void free_all();
    void zero_grad();
};

struct Activations {

    Tensor embedded;
    Tensor ln1_out;
    Tensor QKV_buf;
    Tensor Q, K, Val;
    Tensor attn_out;
    Tensor attn_proj;
    Tensor residual1;
    Tensor ln2_out;
    Tensor ff_hidden;
    Tensor ff_relu;
    Tensor ff_out;
    Tensor residual2;
    Tensor logits;
    Tensor loss_grad;

    Tensor ln1_mean, ln1_inv_std;
    Tensor ln2_mean, ln2_inv_std;

    Tensor grad_residual2;
    Tensor grad_ff_relu;
    Tensor grad_ff_hidden;
    Tensor grad_ln2_out;
    Tensor grad_attn_out;
    Tensor grad_QKV_buf;
    Tensor grad_Q;
    Tensor grad_K;
    Tensor grad_V;
    Tensor grad_ln1_out;

    Tensor attn_P;
    Tensor attn_dP;
    Tensor attn_dS;

    Tensor losses;
    Tensor loss_scalar;

    void allocate(const TransformerConfig& cfg);
    void free_all();
};

struct AdamState {
    Tensor m;
    Tensor v;
    void alloc_like(const Tensor& param);
    void free() { m.free(); v.free(); }
};

struct TransformerModel {
    TransformerConfig cfg;

    Tensor tok_embed, pos_embed;
    Tensor dtok_embed, dpos_embed;

    TransformerBlockParams block;

    Tensor Wout, dWout;

    Activations act;

    int* d_tokens  = nullptr;
    int* d_targets = nullptr;

    static constexpr int NUM_PARAMS = 15;
    AdamState adam_states[NUM_PARAMS];
    int adam_step_count = 0;

    float* d_all_params = nullptr;
    float* d_all_grads = nullptr;
    float* d_all_adam_m = nullptr;
    float* d_all_adam_v = nullptr;
    int total_param_count = 0;

    float* d_norm = nullptr;
    float* d_clip_scale = nullptr;

    void build(const TransformerConfig& cfg);
    void init_weights(unsigned seed = 42);

    float forward(const int* d_tokens, const int* d_targets);
    void forward_no_sync(const int* d_tokens, const int* d_targets);
    float read_loss();
    void backward();
    void backward_bucketed(DistributedContext* dist, int min_bucket_floats = 0);
    void zero_grad();
    void update_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                     float eps = 1e-8f);
    void free_all();
    float* get_grad_ptr() { return d_all_grads; }
    int get_grad_size() { return total_param_count; }
    float* get_param_ptr() { return d_all_params; }
};
