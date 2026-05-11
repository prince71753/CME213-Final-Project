// Single-GPU CUDA code for the mini-Transformer.
#include "model.h"
#include "distributed.h"

__global__ void init_weights_kernel(float* data, int n, float scale,
                                    unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned s = seed + i;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float u = (float)(s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    data[i] = (u * 2.0f - 1.0f) * scale;
}

static void init_tensor(Tensor& t, float scale, unsigned seed) {
    int n = t.size();
    int block = 256;
    init_weights_kernel<<<(n + block - 1) / block, block>>>(
        t.data, n, scale, seed);
    CUDA_CHECK_LAST();
}

static void fill_ones(Tensor& t) {
    std::vector<float> h(t.size(), 1.0f);
    t.copy_from_host(h.data(), t.size());
}

static void fill_zeros(Tensor& t) {
    CUDA_CHECK(cudaMemsetAsync(t.data, 0, t.size() * sizeof(float)));
}

void TransformerBlockParams::allocate(const TransformerConfig& cfg) {
    (void)cfg;
}

void TransformerBlockParams::free_all() {
    Wq.free(); Wk.free(); Wv.free(); Wo.free();
    dWq.free(); dWk.free(); dWv.free(); dWo.free();
    ln1_gamma.free(); ln1_beta.free(); dln1_gamma.free(); dln1_beta.free();
    W1.free(); b1.free(); W2.free(); b2.free();
    dW1.free(); db1.free(); dW2.free(); db2.free();
    ln2_gamma.free(); ln2_beta.free(); dln2_gamma.free(); dln2_beta.free();
}

void TransformerBlockParams::zero_grad() {
}

void Activations::allocate(const TransformerConfig& cfg) {
    int B = cfg.batch_size, T = cfg.seq_len, D = cfg.hidden_dim;
    int H = cfg.num_heads, HD = cfg.head_dim, FF = cfg.ff_dim;
    int V = cfg.vocab_size;
    int BT = B * T;

    embedded.alloc({BT, D});
    ln1_out.alloc({BT, D});
    int qkv_size = B * H * T * HD;
    QKV_buf.alloc({3 * qkv_size});
    Q.set_shape({B * H, T, HD}); Q.data = QKV_buf.data; Q.owned = false;
    K.set_shape({B * H, T, HD}); K.data = QKV_buf.data + qkv_size; K.owned = false;
    Val.set_shape({B * H, T, HD}); Val.data = QKV_buf.data + 2 * qkv_size; Val.owned = false;
    attn_out.alloc({B * H, T, HD});
    attn_proj.alloc({BT, D});
    residual1.alloc({BT, D});
    ln2_out.alloc({BT, D});
    ff_hidden.alloc({BT, FF});
    ff_relu.alloc({BT, FF});
    ff_out.alloc({BT, D});
    residual2.alloc({BT, D});
    logits.alloc({BT, V});
    loss_grad.alloc({BT, V});

    ln1_mean.alloc({BT}); ln1_inv_std.alloc({BT});
    ln2_mean.alloc({BT}); ln2_inv_std.alloc({BT});

    grad_residual2.alloc({BT, D});
    grad_ff_relu.alloc({BT, FF});
    grad_ff_hidden.alloc({BT, FF});
    grad_ln2_out.alloc({BT, D});
    grad_attn_out.alloc({B * H, T, HD});
    grad_QKV_buf.alloc({3 * qkv_size});
    grad_Q.set_shape({B * H, T, HD}); grad_Q.data = grad_QKV_buf.data; grad_Q.owned = false;
    grad_K.set_shape({B * H, T, HD}); grad_K.data = grad_QKV_buf.data + qkv_size; grad_K.owned = false;
    grad_V.set_shape({B * H, T, HD}); grad_V.data = grad_QKV_buf.data + 2 * qkv_size; grad_V.owned = false;
    grad_ln1_out.alloc({BT, D});

    attn_P.alloc({B * H, T, T});
    attn_dP.alloc({B * H, T, T});
    attn_dS.alloc({B * H, T, T});

    losses.alloc({BT});
    loss_scalar.alloc({1});
}

void Activations::free_all() {
    embedded.free(); ln1_out.free();
    QKV_buf.free(); Q.free(); K.free(); Val.free();
    attn_out.free(); attn_proj.free(); residual1.free();
    ln2_out.free(); ff_hidden.free(); ff_relu.free(); ff_out.free();
    residual2.free(); logits.free(); loss_grad.free();
    ln1_mean.free(); ln1_inv_std.free();
    ln2_mean.free(); ln2_inv_std.free();
    grad_residual2.free();
    grad_ff_relu.free(); grad_ff_hidden.free();
    grad_ln2_out.free();
    grad_attn_out.free();
    grad_QKV_buf.free(); grad_Q.free(); grad_K.free(); grad_V.free();
    grad_ln1_out.free();
    attn_P.free(); attn_dP.free(); attn_dS.free();
    losses.free(); loss_scalar.free();
}

static void set_tensor_shape(Tensor& t, std::initializer_list<int> shape) {
    t.set_shape(shape);
    t.owned = false;
}

void TransformerModel::build(const TransformerConfig& c) {
    cfg = c;
    int D = cfg.hidden_dim, FF = cfg.ff_dim, V = cfg.vocab_size, T = cfg.seq_len;

    struct ShapeDef { Tensor* param; Tensor* grad; std::initializer_list<int> shape; };
    ShapeDef defs[] = {

        {&Wout, &dWout, {D, V}},
        {&block.W2, &block.dW2, {FF, D}},
        {&block.b2, &block.db2, {D}},
        {&block.W1, &block.dW1, {D, FF}},
        {&block.b1, &block.db1, {FF}},
        {&block.ln2_gamma, &block.dln2_gamma, {D}},
        {&block.ln2_beta, &block.dln2_beta, {D}},
        {&block.Wo, &block.dWo, {D, D}},
        {&block.Wq, &block.dWq, {D, D}},
        {&block.Wk, &block.dWk, {D, D}},
        {&block.Wv, &block.dWv, {D, D}},
        {&block.ln1_gamma, &block.dln1_gamma, {D}},
        {&block.ln1_beta, &block.dln1_beta, {D}},
        {&tok_embed, &dtok_embed, {V, D}},
        {&pos_embed, &dpos_embed, {T, D}},
    };

    total_param_count = 0;
    for (int i = 0; i < NUM_PARAMS; ++i) {
        set_tensor_shape(*defs[i].param, defs[i].shape);
        set_tensor_shape(*defs[i].grad, defs[i].shape);
        total_param_count += defs[i].param->size();
    }

    CUDA_CHECK(cudaMalloc(&d_all_params, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_all_params, 0, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_grads, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_all_grads, 0, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_adam_m, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_all_adam_m, 0, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_adam_v, total_param_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_all_adam_v, 0, total_param_count * sizeof(float)));

    int offset = 0;
    for (int i = 0; i < NUM_PARAMS; ++i) {
        int sz = defs[i].param->size();

        defs[i].param->data = d_all_params + offset;
        defs[i].grad->data = d_all_grads + offset;

        adam_states[i].m.set_shape({sz});
        adam_states[i].m.data = d_all_adam_m + offset;
        adam_states[i].m.owned = false;
        adam_states[i].v.set_shape({sz});
        adam_states[i].v.data = d_all_adam_v + offset;
        adam_states[i].v.owned = false;

        offset += sz;
    }
    adam_step_count = 0;

    act.allocate(cfg);

    CUDA_CHECK(cudaMalloc(&d_norm, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clip_scale, sizeof(float)));
}

void TransformerModel::init_weights(unsigned seed) {
    float scale = 0.02f;
    init_tensor(tok_embed, scale, seed + 0);
    init_tensor(pos_embed, scale, seed + 1);
    init_tensor(Wout, scale, seed + 2);
    init_tensor(block.Wq, scale, seed + 10);
    init_tensor(block.Wk, scale, seed + 11);
    init_tensor(block.Wv, scale, seed + 12);
    init_tensor(block.Wo, scale, seed + 13);
    fill_ones(block.ln1_gamma); fill_zeros(block.ln1_beta);
    init_tensor(block.W1, scale, seed + 20);
    fill_zeros(block.b1);
    init_tensor(block.W2, scale, seed + 21);
    fill_zeros(block.b2);
    fill_ones(block.ln2_gamma); fill_zeros(block.ln2_beta);
}

void TransformerModel::zero_grad() {
    CUDA_CHECK(cudaMemsetAsync(d_all_grads, 0, total_param_count * sizeof(float)));
}

float TransformerModel::forward(const int* tokens, const int* targets) {
    int B = cfg.batch_size, T = cfg.seq_len, D = cfg.hidden_dim;
    int H = cfg.num_heads, HD = cfg.head_dim;
    int BT = B * T;

    d_tokens  = const_cast<int*>(tokens);
    d_targets = const_cast<int*>(targets);

    embedding_forward(tok_embed.data, tokens, act.embedded.data, BT, D);
    pos_embedding_forward(pos_embed.data, act.embedded.data, B, T, D);

    layernorm_forward_save(act.embedded.data, block.ln1_gamma.data,
                           block.ln1_beta.data, act.ln1_out.data,
                           act.ln1_mean.data, act.ln1_inv_std.data, BT, D);

    gemm_batched(act.ln1_out.data, block.Wq.data, act.Q.data,
                 BT, D, D, 3, 0, D * D, BT * D);

    attention_gemm(act.Q.data, act.K.data, act.Val.data,
                   act.attn_out.data, act.attn_P.data, B * H, T, HD);

    gemm_tiled(act.attn_out.data, block.Wo.data, act.attn_proj.data, BT, D, D);

    residual_layernorm_forward_save(act.embedded.data, act.attn_proj.data,
                                    block.ln2_gamma.data, block.ln2_beta.data,
                                    act.residual1.data, act.ln2_out.data,
                                    act.ln2_mean.data, act.ln2_inv_std.data,
                                    BT, D);

    gemm_tiled(act.ln2_out.data, block.W1.data, act.ff_hidden.data,
               BT, cfg.ff_dim, D);
    bias_relu_forward(act.ff_hidden.data, block.b1.data, act.ff_relu.data,
                      BT, cfg.ff_dim);

    gemm_tiled(act.ff_relu.data, block.W2.data, act.ff_out.data,
               BT, D, cfg.ff_dim);
    bias_add(act.ff_out.data, block.b2.data, BT, D);

    residual_add(act.residual1.data, act.ff_out.data,
                 act.residual2.data, BT * D);

    gemm_tiled(act.residual2.data, Wout.data, act.logits.data,
               BT, cfg.vocab_size, D);

    float loss = cross_entropy_forward(act.logits.data, targets,
                                       act.loss_grad.data, BT, cfg.vocab_size);
    return loss;
}

void TransformerModel::forward_no_sync(const int* tokens, const int* targets) {
    int B = cfg.batch_size, T = cfg.seq_len, D = cfg.hidden_dim;
    int H = cfg.num_heads, HD = cfg.head_dim;
    int BT = B * T;

    d_tokens  = const_cast<int*>(tokens);
    d_targets = const_cast<int*>(targets);

    embedding_forward(tok_embed.data, tokens, act.embedded.data, BT, D);
    pos_embedding_forward(pos_embed.data, act.embedded.data, B, T, D);

    layernorm_forward_save(act.embedded.data, block.ln1_gamma.data,
                           block.ln1_beta.data, act.ln1_out.data,
                           act.ln1_mean.data, act.ln1_inv_std.data, BT, D);

    gemm_batched(act.ln1_out.data, block.Wq.data, act.Q.data,
                 BT, D, D, 3, 0, D * D, BT * D);

    attention_gemm(act.Q.data, act.K.data, act.Val.data,
                   act.attn_out.data, act.attn_P.data, B * H, T, HD);

    gemm_tiled(act.attn_out.data, block.Wo.data, act.attn_proj.data, BT, D, D);

    residual_layernorm_forward_save(act.embedded.data, act.attn_proj.data,
                                    block.ln2_gamma.data, block.ln2_beta.data,
                                    act.residual1.data, act.ln2_out.data,
                                    act.ln2_mean.data, act.ln2_inv_std.data,
                                    BT, D);

    gemm_tiled(act.ln2_out.data, block.W1.data, act.ff_hidden.data,
               BT, cfg.ff_dim, D);
    bias_relu_forward(act.ff_hidden.data, block.b1.data, act.ff_relu.data,
                      BT, cfg.ff_dim);

    gemm_tiled(act.ff_relu.data, block.W2.data, act.ff_out.data,
               BT, D, cfg.ff_dim);
    bias_add(act.ff_out.data, block.b2.data, BT, D);

    residual_add(act.residual1.data, act.ff_out.data,
                 act.residual2.data, BT * D);

    gemm_tiled(act.residual2.data, Wout.data, act.logits.data,
               BT, cfg.vocab_size, D);

    cross_entropy_forward_v2(act.logits.data, targets,
                              act.loss_grad.data, act.losses.data,
                              act.loss_scalar.data, BT, cfg.vocab_size);
}

float TransformerModel::read_loss() {
    float loss;
    CUDA_CHECK(cudaMemcpy(&loss, act.loss_scalar.data, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return loss;
}

void TransformerModel::backward() {
    backward_bucketed(nullptr);
}

void TransformerModel::backward_bucketed(DistributedContext* dist,
                                         int min_bucket_floats) {
    int B = cfg.batch_size, T = cfg.seq_len, D = cfg.hidden_dim;
    int H = cfg.num_heads, HD = cfg.head_dim, FF = cfg.ff_dim;
    int V = cfg.vocab_size;
    int BT = B * T;

    float* pending_bucket = nullptr;
    int pending_count = 0;

    auto flush_bucket = [&]() {
        if (dist && dist->world_size > 1 && pending_bucket && pending_count > 0)
            dist->start_async_gradient_sync(pending_bucket, pending_count);
        pending_bucket = nullptr;
        pending_count = 0;
    };

    auto sync_bucket = [&](float* d_buf, int count) {
        if (!dist || dist->world_size == 1 || count <= 0)
            return;
        if (min_bucket_floats <= 0) {
            dist->start_async_gradient_sync(d_buf, count);
            return;
        }
        if (!pending_bucket) {
            pending_bucket = d_buf;
            pending_count = count;
        } else if (pending_bucket + pending_count == d_buf) {
            pending_count += count;
        } else {
            flush_bucket();
            pending_bucket = d_buf;
            pending_count = count;
        }
        if (pending_count >= min_bucket_floats)
            flush_bucket();
    };

    gemm_tiled_BT(act.loss_grad.data, Wout.data, act.grad_residual2.data,
                  BT, D, V);
    gemm_splitk_AT_acc(act.residual2.data, act.loss_grad.data, dWout.data,
                      D, V, BT);
    sync_bucket(dWout.data, dWout.size());

    gemm_tiled_BT(act.grad_residual2.data, block.W2.data, act.grad_ff_relu.data,
                  BT, FF, D);
    gemm_splitk_AT_acc(act.ff_relu.data, act.grad_residual2.data, block.dW2.data,
                      FF, D, BT);
    bias_backward(act.grad_residual2.data, block.db2.data, BT, D);
    sync_bucket(block.dW2.data, block.dW2.size() + block.db2.size());

    relu_backward(act.ff_hidden.data, act.grad_ff_relu.data,
                  act.grad_ff_hidden.data, BT * FF);

    gemm_tiled_BT(act.grad_ff_hidden.data, block.W1.data, act.grad_ln2_out.data,
                  BT, D, FF);
    gemm_splitk_AT_acc(act.ln2_out.data, act.grad_ff_hidden.data, block.dW1.data,
                      D, FF, BT);
    bias_backward(act.grad_ff_hidden.data, block.db1.data, BT, FF);
    sync_bucket(block.dW1.data, block.dW1.size() + block.db1.size());

    layernorm_backward_residual(act.grad_ln2_out.data, act.residual1.data,
                                act.ln2_mean.data, act.ln2_inv_std.data,
                                block.ln2_gamma.data,
                                act.grad_residual2.data, block.dln2_gamma.data,
                                block.dln2_beta.data, BT, D);
    sync_bucket(block.dln2_gamma.data,
                block.dln2_gamma.size() + block.dln2_beta.size());

    gemm_tiled_BT(act.grad_residual2.data, block.Wo.data,
                  act.grad_attn_out.data, BT, D, D);
    gemm_splitk_AT_acc(act.attn_out.data, act.grad_residual2.data,
                      block.dWo.data, D, D, BT);
    sync_bucket(block.dWo.data, block.dWo.size());

    CUDA_CHECK(cudaMemsetAsync(act.grad_QKV_buf.data, 0,
                               act.grad_QKV_buf.size() * sizeof(float)));
    attention_backward(act.Q.data, act.K.data, act.Val.data,
                       act.attn_out.data, act.grad_attn_out.data,
                       act.grad_Q.data, act.grad_K.data, act.grad_V.data,
                       act.attn_P.data, act.attn_dP.data, act.attn_dS.data,
                       B * H, T, HD);

    gemm_tiled_BT(act.grad_Q.data, block.Wq.data, act.grad_ln1_out.data,
                  BT, D, D);
    gemm_tiled_BT_acc(act.grad_K.data, block.Wk.data, act.grad_ln1_out.data,
                      BT, D, D);
    gemm_tiled_BT_acc(act.grad_V.data, block.Wv.data, act.grad_ln1_out.data,
                      BT, D, D);

    gemm_splitk_AT_acc(act.ln1_out.data, act.grad_Q.data, block.dWq.data,
                      D, D, BT);
    gemm_splitk_AT_acc(act.ln1_out.data, act.grad_K.data, block.dWk.data,
                      D, D, BT);
    gemm_splitk_AT_acc(act.ln1_out.data, act.grad_V.data, block.dWv.data,
                      D, D, BT);
    sync_bucket(block.dWq.data,
                block.dWq.size() + block.dWk.size() + block.dWv.size());

    layernorm_backward_residual(act.grad_ln1_out.data, act.embedded.data,
                                act.ln1_mean.data, act.ln1_inv_std.data,
                                block.ln1_gamma.data,
                                act.grad_residual2.data, block.dln1_gamma.data,
                                block.dln1_beta.data, BT, D);
    sync_bucket(block.dln1_gamma.data,
                block.dln1_gamma.size() + block.dln1_beta.size());

    embedding_backward(act.grad_residual2.data, d_tokens,
                       dtok_embed.data, BT, D);
    pos_embedding_backward(act.grad_residual2.data, dpos_embed.data,
                           B, T, D);
    sync_bucket(dtok_embed.data, dtok_embed.size() + dpos_embed.size());

    flush_bucket();
    if (dist && dist->world_size > 1)
        dist->finish_async_gradient_syncs();
}

void AdamState::alloc_like(const Tensor& param) {
    int n = param.size();
    m.alloc({n});
    v.alloc({n});
}

__global__ void adam_clipped_kernel(float* param, const float* grad,
                                    float* m, float* v,
                                    const float* d_clip_scale,
                                    float lr, float beta1, float beta2,
                                    float eps, float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i] * (*d_clip_scale);
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void TransformerModel::update_adam(float lr, float beta1, float beta2,
                                   float eps) {
    adam_step_count++;
    float bc1 = 1.0f - powf(beta1, (float)adam_step_count);
    float bc2 = 1.0f - powf(beta2, (float)adam_step_count);

    CUDA_CHECK(cudaMemsetAsync(d_norm, 0, sizeof(float)));
    buffer_norm_sq_acc(d_all_grads, total_param_count, d_norm);
    compute_clip_scale(d_norm, d_clip_scale, 1.0f);

    int block = 256;
    adam_clipped_kernel<<<(total_param_count + block - 1) / block, block>>>(
        d_all_params, d_all_grads, d_all_adam_m, d_all_adam_v,
        d_clip_scale, lr, beta1, beta2, eps, bc1, bc2, total_param_count);
    CUDA_CHECK_LAST();
}

void TransformerModel::free_all() {
    act.free_all();
    if (d_all_params) { cudaFree(d_all_params); d_all_params = nullptr; }
    if (d_all_grads) { cudaFree(d_all_grads); d_all_grads = nullptr; }
    if (d_all_adam_m) { cudaFree(d_all_adam_m); d_all_adam_m = nullptr; }
    if (d_all_adam_v) { cudaFree(d_all_adam_v); d_all_adam_v = nullptr; }
    if (d_norm) { cudaFree(d_norm); d_norm = nullptr; }
    if (d_clip_scale) { cudaFree(d_clip_scale); d_clip_scale = nullptr; }
}
