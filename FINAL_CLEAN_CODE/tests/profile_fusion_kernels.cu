// fusion profiling harness.
#include "common.h"
#include "kernels.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_profiler_api.h>

static void fill_pattern(std::vector<float>& v, float scale, float shift = 0.0f) {
    for (int i = 0; i < (int)v.size(); ++i)
        v[i] = scale * sinf(0.013f * (float)i) +
               0.5f * scale * cosf(0.021f * (float)(i + 7)) + shift;
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float d = 0.0f;
    for (int i = 0; i < (int)a.size(); ++i)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static void copy_to_device(float* dst, const std::vector<float>& src) {
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

static void estimate_traffic(const std::string& name, int rows, int cols,
                             double* read_bytes, double* write_bytes) {
    double nbytes = (double)rows * (double)cols * sizeof(float);
    if (name == "bias_relu_unfused") {
        *read_bytes = 3.0 * nbytes;
        *write_bytes = 2.0 * nbytes;
    } else if (name == "bias_relu_fused") {
        *read_bytes = 2.0 * nbytes;
        *write_bytes = 2.0 * nbytes;
    } else if (name == "residual_ln_unfused") {
        *read_bytes = 7.0 * nbytes;
        *write_bytes = 2.0 * nbytes;
    } else if (name == "residual_ln_fused") {
        *read_bytes = 6.0 * nbytes;
        *write_bytes = 2.0 * nbytes;
    } else if (name == "ln_bwd_residual_unfused") {
        *read_bytes = 8.0 * nbytes;
        *write_bytes = 4.0 * nbytes;
    } else if (name == "ln_bwd_residual_fused") {
        *read_bytes = 7.0 * nbytes;
        *write_bytes = 3.0 * nbytes;
    } else {
        *read_bytes = 0.0;
        *write_bytes = 0.0;
    }
}

static void print_timing(const std::string& name, int rows, int cols,
                         float runtime_ms) {
    double read_bytes = 0.0, write_bytes = 0.0;
    estimate_traffic(name, rows, cols, &read_bytes, &write_bytes);
    double total_bytes = read_bytes + write_bytes;
    double bw_gbs = (runtime_ms > 0.0f) ? total_bytes / (runtime_ms * 1.0e6) : 0.0;
    printf("PROFILE_TIMING case=%s runtime_us=%.3f estimated_read_bytes=%.0f "
           "estimated_write_bytes=%.0f estimated_total_bytes=%.0f "
           "estimated_effective_bw_gbs=%.2f\n",
           name.c_str(), runtime_ms * 1000.0f, read_bytes, write_bytes,
           total_bytes, bw_gbs);
}

static bool check_bias_relu(int rows, int cols) {
    int n = rows * cols;
    std::vector<float> h_in(n), h_bias(cols);
    fill_pattern(h_in, 0.1f);
    fill_pattern(h_bias, 0.001f);

    float *d_u, *d_f, *d_bias, *d_out_u, *d_out_f;
    CUDA_CHECK(cudaMalloc(&d_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f, n * sizeof(float)));
    copy_to_device(d_u, h_in);
    copy_to_device(d_f, h_in);
    copy_to_device(d_bias, h_bias);

    bias_add(d_u, d_bias, rows, cols);
    relu_forward(d_u, d_out_u, n);
    bias_relu_forward(d_f, d_bias, d_out_f, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_u(n), h_f(n);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_out_u, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f.data(), d_out_f, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float err = max_abs_diff(h_u, h_f);
    printf("check bias_relu max_abs=%.3e %s\n", err,
           err < 1e-6f ? "PASS" : "FAIL");

    cudaFree(d_u); cudaFree(d_f); cudaFree(d_bias);
    cudaFree(d_out_u); cudaFree(d_out_f);
    return err < 1e-6f;
}

static bool check_residual_ln(int rows, int cols) {
    int n = rows * cols;
    std::vector<float> h_a(n), h_b(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_a, 0.1f);
    fill_pattern(h_b, 0.08f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.02f);

    float *d_a, *d_b, *d_gamma, *d_beta;
    float *d_res_u, *d_out_u, *d_mean_u, *d_inv_u;
    float *d_res_f, *d_out_f, *d_mean_f, *d_inv_f;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_u, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_u, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_f, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_f, rows * sizeof(float)));
    copy_to_device(d_a, h_a);
    copy_to_device(d_b, h_b);
    copy_to_device(d_gamma, h_gamma);
    copy_to_device(d_beta, h_beta);

    residual_add(d_a, d_b, d_res_u, n);
    layernorm_forward_save(d_res_u, d_gamma, d_beta, d_out_u,
                           d_mean_u, d_inv_u, rows, cols);
    residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta, d_res_f,
                                    d_out_f, d_mean_f, d_inv_f, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_res_u(n), h_res_f(n), h_out_u(n), h_out_f(n);
    CUDA_CHECK(cudaMemcpy(h_res_u.data(), d_res_u, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_f.data(), d_res_f, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_u.data(), d_out_u, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_f.data(), d_out_f, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float res_err = max_abs_diff(h_res_u, h_res_f);
    float out_err = max_abs_diff(h_out_u, h_out_f);
    printf("check residual_ln res_abs=%.3e out_abs=%.3e %s\n",
           res_err, out_err,
           (res_err < 1e-6f && out_err < 1e-5f) ? "PASS" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_res_u); cudaFree(d_out_u); cudaFree(d_mean_u); cudaFree(d_inv_u);
    cudaFree(d_res_f); cudaFree(d_out_f); cudaFree(d_mean_f); cudaFree(d_inv_f);
    return res_err < 1e-6f && out_err < 1e-5f;
}

static bool check_ln_bwd_residual(int rows, int cols) {
    int n = rows * cols;
    std::vector<float> h_x(n), h_grad(n), h_residual(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_x, 0.1f);
    fill_pattern(h_grad, 0.05f);
    fill_pattern(h_residual, 0.02f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.01f);

    float *d_x, *d_grad, *d_gamma, *d_beta, *d_out, *d_mean, *d_inv;
    float *d_dx, *d_res_u, *d_res_f, *d_dg_u, *d_db_u, *d_dg_f, *d_db_f;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dg_u, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_u, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dg_f, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_f, cols * sizeof(float)));
    copy_to_device(d_x, h_x);
    copy_to_device(d_grad, h_grad);
    copy_to_device(d_gamma, h_gamma);
    copy_to_device(d_beta, h_beta);

    layernorm_forward_save(d_x, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);
    copy_to_device(d_res_u, h_residual);
    copy_to_device(d_res_f, h_residual);
    CUDA_CHECK(cudaMemset(d_dg_u, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_u, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dg_f, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db_f, 0, cols * sizeof(float)));

    layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                       d_dx, d_dg_u, d_db_u, rows, cols);
    residual_add(d_res_u, d_dx, d_res_u, n);
    layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                d_res_f, d_dg_f, d_db_f, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_res_u(n), h_res_f(n), h_dg_u(cols), h_dg_f(cols);
    std::vector<float> h_db_u(cols), h_db_f(cols);
    CUDA_CHECK(cudaMemcpy(h_res_u.data(), d_res_u, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_f.data(), d_res_f, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dg_u.data(), d_dg_u, cols * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dg_f.data(), d_dg_f, cols * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db_u.data(), d_db_u, cols * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db_f.data(), d_db_f, cols * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float res_err = max_abs_diff(h_res_u, h_res_f);
    float dg_err = max_abs_diff(h_dg_u, h_dg_f);
    float db_err = max_abs_diff(h_db_u, h_db_f);
    printf("check ln_bwd_residual res_abs=%.3e dgamma_abs=%.3e dbeta_abs=%.3e %s\n",
           res_err, dg_err, db_err,
           (res_err < 1e-5f && dg_err < 1e-3f && db_err < 1e-3f) ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_grad); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_out); cudaFree(d_mean); cudaFree(d_inv); cudaFree(d_dx);
    cudaFree(d_res_u); cudaFree(d_res_f);
    cudaFree(d_dg_u); cudaFree(d_db_u); cudaFree(d_dg_f); cudaFree(d_db_f);
    return res_err < 1e-5f && dg_err < 1e-3f && db_err < 1e-3f;
}

static int run_bias_relu_case(const std::string& name) {
    const int rows = 2048, cols = 512, n = rows * cols;
    std::vector<float> h_in(n), h_bias(cols);
    fill_pattern(h_in, 0.1f);
    fill_pattern(h_bias, 0.001f);

    float *d_base, *d_work, *d_bias, *d_out;
    CUDA_CHECK(cudaMalloc(&d_base, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    copy_to_device(d_base, h_in);
    copy_to_device(d_bias, h_bias);

    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemcpy(d_work, d_base, n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        if (name == "bias_relu_unfused") {
            bias_add(d_work, d_bias, rows, cols);
            relu_forward(d_work, d_out, n);
        } else {
            bias_relu_forward(d_work, d_bias, d_out, rows, cols);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_work, d_base, n * sizeof(float), cudaMemcpyDeviceToDevice));
    printf("PROFILE_CASE %s rows=%d cols=%d\n", name.c_str(), rows, cols);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start));
    if (name == "bias_relu_unfused") {
        bias_add(d_work, d_bias, rows, cols);
        relu_forward(d_work, d_out, n);
    } else {
        bias_relu_forward(d_work, d_bias, d_out, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float runtime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, start, stop));
    CUDA_CHECK(cudaProfilerStop());
    print_timing(name, rows, cols, runtime_ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_base); cudaFree(d_work); cudaFree(d_bias); cudaFree(d_out);
    return 0;
}

static int run_residual_ln_case(const std::string& name) {
    const int rows = 2048, cols = 128, n = rows * cols;
    std::vector<float> h_a(n), h_b(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_a, 0.1f);
    fill_pattern(h_b, 0.08f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.02f);

    float *d_a, *d_b, *d_gamma, *d_beta, *d_res, *d_out, *d_mean, *d_inv;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));
    copy_to_device(d_a, h_a);
    copy_to_device(d_b, h_b);
    copy_to_device(d_gamma, h_gamma);
    copy_to_device(d_beta, h_beta);

    for (int i = 0; i < 5; ++i) {
        if (name == "residual_ln_unfused") {
            residual_add(d_a, d_b, d_res, n);
            layernorm_forward_save(d_res, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);
        } else {
            residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta,
                                            d_res, d_out, d_mean, d_inv, rows, cols);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("PROFILE_CASE %s rows=%d cols=%d\n", name.c_str(), rows, cols);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start));
    if (name == "residual_ln_unfused") {
        residual_add(d_a, d_b, d_res, n);
        layernorm_forward_save(d_res, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);
    } else {
        residual_layernorm_forward_save(d_a, d_b, d_gamma, d_beta,
                                        d_res, d_out, d_mean, d_inv, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float runtime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, start, stop));
    CUDA_CHECK(cudaProfilerStop());
    print_timing(name, rows, cols, runtime_ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_res); cudaFree(d_out); cudaFree(d_mean); cudaFree(d_inv);
    return 0;
}

static int run_ln_bwd_case(const std::string& name) {
    const int rows = 2048, cols = 128, n = rows * cols;
    std::vector<float> h_x(n), h_grad(n), h_residual(n), h_gamma(cols), h_beta(cols);
    fill_pattern(h_x, 0.1f);
    fill_pattern(h_grad, 0.05f);
    fill_pattern(h_residual, 0.02f);
    fill_pattern(h_gamma, 0.03f, 1.0f);
    fill_pattern(h_beta, 0.01f);

    float *d_x, *d_grad, *d_gamma, *d_beta, *d_out, *d_mean, *d_inv;
    float *d_dx, *d_res, *d_dgamma, *d_dbeta;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dgamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbeta, cols * sizeof(float)));
    copy_to_device(d_x, h_x);
    copy_to_device(d_grad, h_grad);
    copy_to_device(d_gamma, h_gamma);
    copy_to_device(d_beta, h_beta);
    layernorm_forward_save(d_x, d_gamma, d_beta, d_out, d_mean, d_inv, rows, cols);

    for (int i = 0; i < 5; ++i) {
        copy_to_device(d_res, h_residual);
        CUDA_CHECK(cudaMemset(d_dgamma, 0, cols * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dbeta, 0, cols * sizeof(float)));
        if (name == "ln_bwd_residual_unfused") {
            layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                               d_dx, d_dgamma, d_dbeta, rows, cols);
            residual_add(d_res, d_dx, d_res, n);
        } else {
            layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                        d_res, d_dgamma, d_dbeta, rows, cols);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    copy_to_device(d_res, h_residual);
    CUDA_CHECK(cudaMemset(d_dgamma, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbeta, 0, cols * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("PROFILE_CASE %s rows=%d cols=%d\n", name.c_str(), rows, cols);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start));
    if (name == "ln_bwd_residual_unfused") {
        layernorm_backward(d_grad, d_x, d_mean, d_inv, d_gamma,
                           d_dx, d_dgamma, d_dbeta, rows, cols);
        residual_add(d_res, d_dx, d_res, n);
    } else {
        layernorm_backward_residual(d_grad, d_x, d_mean, d_inv, d_gamma,
                                    d_res, d_dgamma, d_dbeta, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float runtime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, start, stop));
    CUDA_CHECK(cudaProfilerStop());
    print_timing(name, rows, cols, runtime_ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_x); cudaFree(d_grad); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_out); cudaFree(d_mean); cudaFree(d_inv);
    cudaFree(d_dx); cudaFree(d_res); cudaFree(d_dgamma); cudaFree(d_dbeta);
    return 0;
}

static void print_usage(const char* argv0) {
    printf("Usage: %s --case <name>\n", argv0);
    printf("Cases:\n");
    printf("  bias_relu_unfused\n");
    printf("  bias_relu_fused\n");
    printf("  residual_ln_unfused\n");
    printf("  residual_ln_fused\n");
    printf("  ln_bwd_residual_unfused\n");
    printf("  ln_bwd_residual_fused\n");
}

static bool is_one_of(const std::string& name, const char* a, const char* b) {
    return name == a || name == b;
}

int main(int argc, char** argv) {
    std::string name;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
            name = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (name.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const int ln_rows = 2048, ln_cols = 128;
    const int ff_rows = 2048, ff_cols = 512;

    bool ok = true;
    if (is_one_of(name, "bias_relu_unfused", "bias_relu_fused")) {
        ok = check_bias_relu(ff_rows, ff_cols);
        if (!ok) return 1;
        return run_bias_relu_case(name);
    }
    if (is_one_of(name, "residual_ln_unfused", "residual_ln_fused")) {
        ok = check_residual_ln(ln_rows, ln_cols);
        if (!ok) return 1;
        return run_residual_ln_case(name);
    }
    if (is_one_of(name, "ln_bwd_residual_unfused", "ln_bwd_residual_fused")) {
        ok = check_ln_bwd_residual(ln_rows, ln_cols);
        if (!ok) return 1;
        return run_ln_bwd_case(name);
    }

    print_usage(argv[0]);
    return 1;
}
