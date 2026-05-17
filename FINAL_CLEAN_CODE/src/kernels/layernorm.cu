// layer normalization forward and backward kernels.
#include "kernels.h"

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ float block_reduce_sum_device(float val, float* shared) {
    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int nwarps  = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : 0.0f;
    val = warp_reduce_sum(val);
    return val;
}

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out,
                                 int cols, float eps) {
    int row = blockIdx.x;
    const float* row_in = x + row * cols;
    float* row_out = out + row * cols;

    __shared__ float shared[32];
    __shared__ float s_mean, s_inv_std;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += row_in[i];

    float total = block_reduce_sum_device(local_sum, shared);
    if (threadIdx.x == 0) s_mean = total / cols;
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    float var_total = block_reduce_sum_device(local_var, shared);
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_total / cols + eps);
    __syncthreads();
    float inv_std = s_inv_std;

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
}

void layernorm_forward(const float* x, const float* gamma, const float* beta,
                       float* out, int rows, int cols, float eps) {
    int threads = min(256, cols);
    threads = max(threads, 32);
    layernorm_kernel<<<rows, threads>>>(x, gamma, beta, out, cols, eps);
    CUDA_CHECK_LAST();
}

__global__ void layernorm_save_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ out,
                                      float* __restrict__ save_mean,
                                      float* __restrict__ save_inv_std,
                                      int cols, float eps) {
    int row = blockIdx.x;
    const float* row_in = x + row * cols;
    float* row_out = out + row * cols;

    __shared__ float shared[32];
    __shared__ float s_mean, s_inv_std;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_sum += row_in[i];

    float total = block_reduce_sum_device(local_sum, shared);
    if (threadIdx.x == 0) s_mean = total / cols;
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    float var_total = block_reduce_sum_device(local_var, shared);
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_total / cols + eps);
    __syncthreads();
    float inv_std = s_inv_std;

    if (threadIdx.x == 0) {
        save_mean[row] = mean;
        save_inv_std[row] = inv_std;
    }

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
}

void layernorm_forward_save(const float* x, const float* gamma, const float* beta,
                            float* out, float* mean, float* inv_std,
                            int rows, int cols, float eps) {
    int threads = min(256, cols);
    threads = max(threads, 32);
    layernorm_save_kernel<<<rows, threads>>>(x, gamma, beta, out,
                                             mean, inv_std, cols, eps);
    CUDA_CHECK_LAST();
}

__global__ void residual_layernorm_save_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ residual_out,
    float* __restrict__ out,
    float* __restrict__ save_mean,
    float* __restrict__ save_inv_std,
    int cols, float eps)
{
    int row = blockIdx.x;
    const float* a_row = a + row * cols;
    const float* b_row = b + row * cols;
    float* res_row = residual_out + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared[32];
    __shared__ float s_mean, s_inv_std;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float r = a_row[i] + b_row[i];
        res_row[i] = r;
        local_sum += r;
    }

    float total = block_reduce_sum_device(local_sum, shared);
    if (threadIdx.x == 0) s_mean = total / cols;
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = res_row[i] - mean;
        local_var += diff * diff;
    }

    float var_total = block_reduce_sum_device(local_var, shared);
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_total / cols + eps);
    __syncthreads();
    float inv_std = s_inv_std;

    if (threadIdx.x == 0) {
        save_mean[row] = mean;
        save_inv_std[row] = inv_std;
    }

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] = (res_row[i] - mean) * inv_std * gamma[i] + beta[i];
}

void residual_layernorm_forward_save(const float* a, const float* b,
                                     const float* gamma, const float* beta,
                                     float* residual_out, float* out,
                                     float* mean, float* inv_std,
                                     int rows, int cols, float eps) {
    int threads = min(256, cols);
    threads = max(threads, 32);
    residual_layernorm_save_kernel<<<rows, threads>>>(
        a, b, gamma, beta, residual_out, out, mean, inv_std, cols, eps);
    CUDA_CHECK_LAST();
}

namespace {

template <bool ADD_TO_GRAD>
__global__ void layernorm_backward_dx_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gamma,
    float* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    const float* go = grad_out + row * cols;
    const float* xi = x + row * cols;
    float* gx = grad_x + row * cols;
    float mu = mean[row];
    float is = inv_std[row];

    __shared__ float shared[32];
    __shared__ float s_sum1, s_sum2;

    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float x_hat = (xi[i] - mu) * is;
        float dx_hat = go[i] * gamma[i];
        sum_dxhat += dx_hat;
        sum_dxhat_xhat += dx_hat * x_hat;
    }

    float total1 = block_reduce_sum_device(sum_dxhat, shared);
    if (threadIdx.x == 0) s_sum1 = total1 / cols;
    __syncthreads();
    float mean_dxhat = s_sum1;

    float total2 = block_reduce_sum_device(sum_dxhat_xhat, shared);
    if (threadIdx.x == 0) s_sum2 = total2 / cols;
    __syncthreads();
    float mean_dxhat_xhat = s_sum2;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float x_hat = (xi[i] - mu) * is;
        float dx_hat = go[i] * gamma[i];
        float dx = is * (dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat);
        if (ADD_TO_GRAD)
            gx[i] += dx;
        else
            gx[i]  = dx;
    }
}

__global__ void layernorm_backward_dweight_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows, int cols)
{
    int col = blockIdx.x;
    if (col >= cols) return;

    __shared__ float shared[32];

    float dg = 0.0f;
    float db = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        float go    = grad_out[r * cols + col];
        float x_hat = (x[r * cols + col] - mean[r]) * inv_std[r];
        dg += go * x_hat;
        db += go;
    }

    float dg_total = block_reduce_sum_device(dg, shared);
    __syncthreads();
    float db_total = block_reduce_sum_device(db, shared);

    if (threadIdx.x == 0) {

        dgamma[col] += dg_total;
        dbeta[col]  += db_total;
    }
}

}

void layernorm_backward(const float* grad_out, const float* x,
                        const float* mean, const float* inv_std,
                        const float* gamma,
                        float* grad_x, float* dgamma, float* dbeta,
                        int rows, int cols) {
    int threads = min(256, cols);
    threads = max(threads, 32);
    layernorm_backward_dx_kernel<false><<<rows, threads>>>(
        grad_out, x, mean, inv_std, gamma, grad_x, cols);
    CUDA_CHECK_LAST();

    int red_threads = min(256, rows);
    red_threads = max(red_threads, 32);
    layernorm_backward_dweight_kernel<<<cols, red_threads>>>(
        grad_out, x, mean, inv_std, dgamma, dbeta, rows, cols);
    CUDA_CHECK_LAST();
}

void layernorm_backward_residual(const float* grad_out, const float* x,
                                 const float* mean, const float* inv_std,
                                 const float* gamma,
                                 float* grad_residual, float* dgamma, float* dbeta,
                                 int rows, int cols) {
    int threads = min(256, cols);
    threads = max(threads, 32);

    layernorm_backward_dx_kernel<true><<<rows, threads>>>(
        grad_out, x, mean, inv_std, gamma, grad_residual, cols);
    CUDA_CHECK_LAST();

    int red_threads = min(256, rows);
    red_threads = max(red_threads, 32);
    layernorm_backward_dweight_kernel<<<cols, red_threads>>>(
        grad_out, x, mean, inv_std, dgamma, dbeta, rows, cols);
    CUDA_CHECK_LAST();
}
