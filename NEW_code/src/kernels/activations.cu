// Single-GPU CUDA code for the mini-Transformer.
#include "kernels.h"
#include <cfloat>

__global__ void relu_forward_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

void relu_forward(const float* in, float* out, int n) {
    int block = 256;
    relu_forward_kernel<<<(n + block - 1) / block, block>>>(in, out, n);
    CUDA_CHECK_LAST();
}

__global__ void relu_backward_kernel(const float* in, const float* grad_out,
                                     float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
}

void relu_backward(const float* in, const float* grad_out, float* grad_in, int n) {
    int block = 256;
    relu_backward_kernel<<<(n + block - 1) / block, block>>>(in, grad_out, grad_in, n);
    CUDA_CHECK_LAST();
}

__global__ void residual_add_kernel(const float* a, const float* b,
                                    float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

void residual_add(const float* a, const float* b, float* out, int n) {
    int block = 256;
    residual_add_kernel<<<(n + block - 1) / block, block>>>(a, b, out, n);
    CUDA_CHECK_LAST();
}

__global__ void bias_add_kernel(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols;
    out[idx] += bias[c];
}

void bias_add(float* out, const float* bias, int rows, int cols) {
    int n = rows * cols;
    int block = 256;
    bias_add_kernel<<<(n + block - 1) / block, block>>>(out, bias, rows, cols);
    CUDA_CHECK_LAST();
}

__global__ void bias_relu_kernel(float* __restrict__ in_save,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    float val = in_save[idx] + bias[idx % cols];
    in_save[idx] = val;
    out[idx] = (val > 0.0f) ? val : 0.0f;
}

void bias_relu_forward(const float* in, const float* bias, float* out,
                       int rows, int cols) {
    int n = rows * cols;
    int block = 256;
    bias_relu_kernel<<<(n + block - 1) / block, block>>>(
        const_cast<float*>(in), bias, out, rows, cols);
    CUDA_CHECK_LAST();
}

#define BIAS_BWD_ROWS_PER_BLOCK 128

__global__ void bias_backward_kernel(const float* grad, float* dbias,
                                     int rows, int cols, int rows_per_block) {
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, rows);

    float sum = 0.0f;
    for (int r = row_start; r < row_end; ++r)
        sum += grad[r * cols + c];
    atomicAdd(&dbias[c], sum);
}

void bias_backward(const float* grad, float* dbias, int rows, int cols) {
    int threads = min(256, cols);
    int rpb = BIAS_BWD_ROWS_PER_BLOCK;
    dim3 grid((rows + rpb - 1) / rpb, (cols + threads - 1) / threads);
    bias_backward_kernel<<<grid, threads>>>(grad, dbias, rows, cols, rpb);
    CUDA_CHECK_LAST();
}

__global__ void scale_kernel(float* buf, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= scale;
}

void scale_buffer(float* buf, float scale, int n) {
    int block = 256;
    scale_kernel<<<(n + block - 1) / block, block>>>(buf, scale, n);
    CUDA_CHECK_LAST();
}

__global__ void norm_sq_kernel(const float* buf, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if (i < n) sum += buf[i] * buf[i];
    if (i + blockDim.x < n) sum += buf[i + blockDim.x] * buf[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

void buffer_norm_sq_acc(const float* buf, int n, float* d_result) {
    int block = 256;
    int grid = (n + block * 2 - 1) / (block * 2);
    norm_sq_kernel<<<grid, block, block * sizeof(float)>>>(buf, d_result, n);
    CUDA_CHECK_LAST();
}

__global__ void clip_scale_kernel(const float* d_norm_sq, float* d_clip_scale,
                                   float max_norm) {
    float norm = sqrtf(*d_norm_sq);
    *d_clip_scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
}

void compute_clip_scale(const float* d_norm_sq, float* d_clip_scale, float max_norm) {
    clip_scale_kernel<<<1, 1>>>(d_norm_sq, d_clip_scale, max_norm);
    CUDA_CHECK_LAST();
}

__global__ void softmax_kernel(const float* x, float* out, int cols) {
    int row = blockIdx.x;
    const float* in_row  = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float sdata[];

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, in_row[i]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(in_row[i] - row_max);
        out_row[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] /= sum;
}

void softmax_forward(const float* x, float* out, int rows, int cols) {
    int threads = min(256, cols);
    threads = max(threads, 32);
    softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(x, out, cols);
    CUDA_CHECK_LAST();
}

__global__ void cross_entropy_kernel(const float* logits, const int* targets,
                                     float* grad, float* losses,
                                     int vocab, float grad_scale) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int warp_size = 32;
    const float* row = logits + b * vocab;
    float* g = grad + b * vocab;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab; i += blockDim.x)
        local_max = fmaxf(local_max, row[i]);
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    float row_max = local_max;

    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float e = expf(row[i] - row_max);
        g[i] = e;
        local_sum += e;
    }
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    float inv_sum = 1.0f / (local_sum + 1e-9f);
    for (int i = tid; i < vocab; i += blockDim.x)
        g[i] *= inv_sum;
    __syncwarp();

    if (tid == 0) {
        int tgt = targets[b];
        losses[b] = -logf(g[tgt] + 1e-9f);
        g[tgt] -= 1.0f;
    }
    __syncwarp();

    for (int i = tid; i < vocab; i += blockDim.x)
        g[i] *= grad_scale;
}

float cross_entropy_forward(const float* logits, const int* targets,
                            float* grad, int batch, int vocab) {
    float* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch * sizeof(float)));

    cross_entropy_kernel<<<batch, 32>>>(logits, targets, grad, d_losses, vocab, 1.0f);
    CUDA_CHECK_LAST();

    std::vector<float> h_losses(batch);
    CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, batch * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_losses);

    float total = 0.0f;
    for (int i = 0; i < batch; ++i) total += h_losses[i];
    return total / batch;
}

__global__ void reduce_sum_kernel(const float* data, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if (i < n) sum += data[i];
    if (i + blockDim.x < n) sum += data[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void scale_device_scalar(float* val, float divisor) {
    *val /= divisor;
}

void cross_entropy_forward_v2(const float* logits, const int* targets,
                               float* grad, float* d_losses, float* d_loss_out,
                               int batch, int vocab) {
    float grad_scale = 1.0f / (float)batch;
    cross_entropy_kernel<<<batch, 32>>>(logits, targets, grad, d_losses, vocab, grad_scale);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemsetAsync(d_loss_out, 0, sizeof(float)));
    int block = 256;
    int grid = (batch + block * 2 - 1) / (block * 2);
    reduce_sum_kernel<<<grid, block, block * sizeof(float)>>>(d_losses, d_loss_out, batch);
    CUDA_CHECK_LAST();
    scale_device_scalar<<<1, 1>>>(d_loss_out, (float)batch);
    CUDA_CHECK_LAST();
}
