// end to end cpu reference test.
#include "common.h"
#include "model.h"
#include <cmath>
#include <cstdio>
#include <vector>

struct ParamOffsets {
    int Wout, W2, b2, W1, b1, ln2_gamma, ln2_beta, Wo;
    int Wq, Wk, Wv, ln1_gamma, ln1_beta, tok_embed, pos_embed, total;
};

static ParamOffsets make_offsets(const TransformerConfig& cfg) {
    int D = cfg.hidden_dim, FF = cfg.ff_dim, V = cfg.vocab_size, T = cfg.seq_len;
    ParamOffsets o{};
    int p = 0;
    o.Wout = p; p += D * V;
    o.W2 = p; p += FF * D;
    o.b2 = p; p += D;
    o.W1 = p; p += D * FF;
    o.b1 = p; p += FF;
    o.ln2_gamma = p; p += D;
    o.ln2_beta = p; p += D;
    o.Wo = p; p += D * D;
    o.Wq = p; p += D * D;
    o.Wk = p; p += D * D;
    o.Wv = p; p += D * D;
    o.ln1_gamma = p; p += D;
    o.ln1_beta = p; p += D;
    o.tok_embed = p; p += V * D;
    o.pos_embed = p; p += T * D;
    o.total = p;
    return o;
}

static void matmul(const double* A, const double* B, std::vector<double>& C,
                   int M, int N, int K) {
    C.assign(M * N, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

static void layernorm(const std::vector<double>& x, const double* gamma,
                      const double* beta, std::vector<double>& out,
                      int rows, int cols, double eps) {
    out.assign(rows * cols, 0.0);
    for (int r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int c = 0; c < cols; ++c) mean += x[r * cols + c];
        mean /= cols;

        double var = 0.0;
        for (int c = 0; c < cols; ++c) {
            double d = x[r * cols + c] - mean;
            var += d * d;
        }
        double inv_std = 1.0 / std::sqrt(var / cols + eps);
        for (int c = 0; c < cols; ++c)
            out[r * cols + c] =
                (x[r * cols + c] - mean) * inv_std * gamma[c] + beta[c];
    }
}

static void attention_cpu(const std::vector<double>& Q,
                          const std::vector<double>& K,
                          const std::vector<double>& V,
                          std::vector<double>& out,
                          int BH, int T, int HD) {
    out.assign(BH * T * HD, 0.0);
    double scale = 1.0 / std::sqrt((double)HD);
    for (int bh = 0; bh < BH; ++bh) {
        int base = bh * T * HD;
        for (int i = 0; i < T; ++i) {
            std::vector<double> score(T);
            double row_max = -1.0e300;
            for (int j = 0; j < T; ++j) {
                double dot = 0.0;
                for (int d = 0; d < HD; ++d)
                    dot += Q[base + i * HD + d] * K[base + j * HD + d];
                score[j] = dot * scale;
                if (score[j] > row_max) row_max = score[j];
            }
            double denom = 0.0;
            for (int j = 0; j < T; ++j) {
                score[j] = std::exp(score[j] - row_max);
                denom += score[j];
            }
            for (int d = 0; d < HD; ++d) {
                double v = 0.0;
                for (int j = 0; j < T; ++j)
                    v += (score[j] / denom) * V[base + j * HD + d];
                out[base + i * HD + d] = v;
            }
        }
    }
}

static double model_forward_cpu(const std::vector<double>& p,
                                const TransformerConfig& cfg,
                                const std::vector<int>& tokens,
                                const std::vector<int>& targets,
                                std::vector<double>* logits_out = nullptr) {
    ParamOffsets o = make_offsets(cfg);
    int B = cfg.batch_size, T = cfg.seq_len, D = cfg.hidden_dim;
    int H = cfg.num_heads, HD = cfg.head_dim, FF = cfg.ff_dim, V = cfg.vocab_size;
    int BT = B * T;

    const double* Wout = p.data() + o.Wout;
    const double* W2 = p.data() + o.W2;
    const double* b2 = p.data() + o.b2;
    const double* W1 = p.data() + o.W1;
    const double* b1 = p.data() + o.b1;
    const double* ln2g = p.data() + o.ln2_gamma;
    const double* ln2b = p.data() + o.ln2_beta;
    const double* Wo = p.data() + o.Wo;
    const double* Wq = p.data() + o.Wq;
    const double* Wk = p.data() + o.Wk;
    const double* Wv = p.data() + o.Wv;
    const double* ln1g = p.data() + o.ln1_gamma;
    const double* ln1b = p.data() + o.ln1_beta;
    const double* tok = p.data() + o.tok_embed;
    const double* pos = p.data() + o.pos_embed;

    std::vector<double> embedded(BT * D);
    for (int bt = 0; bt < BT; ++bt) {
        int token = tokens[bt];
        int t = bt % T;
        for (int d = 0; d < D; ++d)
            embedded[bt * D + d] = tok[token * D + d] + pos[t * D + d];
    }

    std::vector<double> ln1;
    layernorm(embedded, ln1g, ln1b, ln1, BT, D, cfg.eps);

    int qkv_size = BT * D;
    std::vector<double> qkv(3 * qkv_size), tmp;
    matmul(ln1.data(), Wq, tmp, BT, D, D);
    std::copy(tmp.begin(), tmp.end(), qkv.begin());
    matmul(ln1.data(), Wk, tmp, BT, D, D);
    std::copy(tmp.begin(), tmp.end(), qkv.begin() + qkv_size);
    matmul(ln1.data(), Wv, tmp, BT, D, D);
    std::copy(tmp.begin(), tmp.end(), qkv.begin() + 2 * qkv_size);

    std::vector<double> Q(qkv.begin(), qkv.begin() + qkv_size);
    std::vector<double> K(qkv.begin() + qkv_size, qkv.begin() + 2 * qkv_size);
    std::vector<double> Val(qkv.begin() + 2 * qkv_size, qkv.end());
    std::vector<double> attn_out;
    attention_cpu(Q, K, Val, attn_out, B * H, T, HD);

    std::vector<double> attn_proj;
    matmul(attn_out.data(), Wo, attn_proj, BT, D, D);

    std::vector<double> residual1(BT * D);
    for (int i = 0; i < BT * D; ++i) residual1[i] = embedded[i] + attn_proj[i];

    std::vector<double> ln2;
    layernorm(residual1, ln2g, ln2b, ln2, BT, D, cfg.eps);

    std::vector<double> ff_hidden;
    matmul(ln2.data(), W1, ff_hidden, BT, FF, D);
    std::vector<double> ff_relu(BT * FF);
    for (int r = 0; r < BT; ++r) {
        for (int c = 0; c < FF; ++c) {
            double v = ff_hidden[r * FF + c] + b1[c];
            ff_hidden[r * FF + c] = v;
            ff_relu[r * FF + c] = v > 0.0 ? v : 0.0;
        }
    }

    std::vector<double> ff_out;
    matmul(ff_relu.data(), W2, ff_out, BT, D, FF);
    for (int r = 0; r < BT; ++r)
        for (int d = 0; d < D; ++d)
            ff_out[r * D + d] += b2[d];

    std::vector<double> residual2(BT * D);
    for (int i = 0; i < BT * D; ++i) residual2[i] = residual1[i] + ff_out[i];

    std::vector<double> logits;
    matmul(residual2.data(), Wout, logits, BT, V, D);
    if (logits_out) *logits_out = logits;

    double loss = 0.0;
    for (int r = 0; r < BT; ++r) {
        double row_max = logits[r * V];
        for (int v = 1; v < V; ++v)
            if (logits[r * V + v] > row_max) row_max = logits[r * V + v];
        double denom = 0.0;
        for (int v = 0; v < V; ++v)
            denom += std::exp(logits[r * V + v] - row_max);
        int target = targets[r];
        double prob = std::exp(logits[r * V + target] - row_max) / denom;
        loss += -std::log(prob + 1e-9);
    }
    return loss / BT;
}

static double max_abs_diff(const std::vector<float>& a,
                           const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs((double)a[i] - b[i]));
    return d;
}

int main() {
    printf("=== End-to-End Model Reference Test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 7;
    cfg.seq_len = 4;
    cfg.hidden_dim = 8;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.ff_dim = 16;
    cfg.batch_size = 2;
    cfg.eps = 1e-5f;

    TransformerModel model;
    model.build(cfg);

    ParamOffsets offsets = make_offsets(cfg);
    if (offsets.total != model.total_param_count) {
        printf("  param layout mismatch: cpu=%d gpu=%d FAIL\n",
               offsets.total, model.total_param_count);
        return 1;
    }

    std::vector<float> h_params(model.total_param_count);
    for (int i = 0; i < model.total_param_count; ++i)
        h_params[i] = 0.05f * std::sin(0.17f * (float)i) +
                      0.02f * std::cos(0.11f * (float)(i + 3));
    CUDA_CHECK(cudaMemcpy(model.d_all_params, h_params.data(),
                          h_params.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    int BT = cfg.batch_size * cfg.seq_len;
    std::vector<int> h_tokens(BT), h_targets(BT);
    for (int i = 0; i < BT; ++i) {
        h_tokens[i] = (3 * i + 1) % cfg.vocab_size;
        h_targets[i] = (5 * i + 2) % cfg.vocab_size;
    }

    int *d_tokens, *d_targets;
    CUDA_CHECK(cudaMalloc(&d_tokens, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, BT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), BT * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), BT * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::vector<double> params_d(h_params.begin(), h_params.end());
    std::vector<double> cpu_logits;
    double cpu_loss = model_forward_cpu(params_d, cfg, h_tokens, h_targets,
                                        &cpu_logits);

    model.zero_grad();
    model.forward_no_sync(d_tokens, d_targets);
    float gpu_loss = model.read_loss();

    std::vector<float> gpu_logits(BT * cfg.vocab_size);
    CUDA_CHECK(cudaMemcpy(gpu_logits.data(), model.act.logits.data,
                          gpu_logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double loss_err = std::fabs((double)gpu_loss - cpu_loss);
    double logits_err = max_abs_diff(gpu_logits, cpu_logits);
    printf("  forward loss cpu=%.8f gpu=%.8f abs_err=%.3e %s\n",
           cpu_loss, gpu_loss, loss_err, loss_err < 2e-5 ? "PASS" : "FAIL");
    printf("  logits max_abs_err=%.3e %s\n",
           logits_err, logits_err < 2e-5 ? "PASS" : "FAIL");
    if (loss_err >= 2e-5 || logits_err >= 2e-5) return 1;

    model.backward();
    std::vector<float> h_grads(model.total_param_count);
    CUDA_CHECK(cudaMemcpy(h_grads.data(), model.d_all_grads,
                          h_grads.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<int> grad_checks = {
        offsets.Wout + 3,
        offsets.W2 + 7,
        offsets.W1 + 11,
        offsets.Wo + 5,
        offsets.Wq + 9,
        offsets.Wv + 13,
        offsets.tok_embed + h_tokens[0] * cfg.hidden_dim + 1,
        offsets.pos_embed + 2
    };

    double max_grad_abs = 0.0, max_grad_rel = 0.0;
    const double fd_eps = 1e-3;
    for (int idx : grad_checks) {
        std::vector<double> plus = params_d;
        std::vector<double> minus = params_d;
        plus[idx] += fd_eps;
        minus[idx] -= fd_eps;
        double lp = model_forward_cpu(plus, cfg, h_tokens, h_targets);
        double lm = model_forward_cpu(minus, cfg, h_tokens, h_targets);
        double numeric = (lp - lm) / (2.0 * fd_eps);
        double analytic = h_grads[idx];
        double abs_err = std::fabs(numeric - analytic);
        double rel_err = abs_err / (std::fabs(numeric) + std::fabs(analytic) + 1e-12);
        max_grad_abs = std::max(max_grad_abs, abs_err);
        max_grad_rel = std::max(max_grad_rel, rel_err);
        printf("  grad idx %4d numeric=% .6e analytic=% .6e abs=%.3e rel=%.3e\n",
               idx, numeric, analytic, abs_err, rel_err);
    }
    bool grad_ok = max_grad_abs < 3e-2 || max_grad_rel < 2e-1;
    printf("  gradient finite-difference max_abs=%.3e max_rel=%.3e %s\n",
           max_grad_abs, max_grad_rel, grad_ok ? "PASS" : "FAIL");
    if (!grad_ok) return 1;

    std::vector<float> before = h_params;
    float lr = 1e-3f;
    model.update_adam(lr);
    std::vector<float> after(model.total_param_count);
    CUDA_CHECK(cudaMemcpy(after.data(), model.d_all_params,
                          after.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float clip_scale = 1.0f;
    CUDA_CHECK(cudaMemcpy(&clip_scale, model.d_clip_scale, sizeof(float),
                          cudaMemcpyDeviceToHost));

    double max_update_err = 0.0;
    for (int idx : grad_checks) {
        double g = (double)h_grads[idx] * (double)clip_scale;
        double expected = before[idx];
        if (g != 0.0)
            expected -= (double)lr * g / (std::fabs(g) + 1e-8);
        double err = std::fabs((double)after[idx] - expected);
        max_update_err = std::max(max_update_err, err);
    }
    printf("  adam first-step clip_scale=%.6f max_update_err=%.3e %s\n",
           clip_scale, max_update_err, max_update_err < 5e-5 ? "PASS" : "FAIL");
    if (max_update_err >= 5e-5) return 1;

    cudaFree(d_tokens);
    cudaFree(d_targets);
    model.free_all();

    printf("\n=== MODEL REFERENCE TEST PASSED ===\n");
    return 0;
}
