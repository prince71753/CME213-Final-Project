// Single-GPU CUDA code for the mini-Transformer.
#pragma once

#include "common.h"

struct Tensor {
    float* data   = nullptr;
    int    dims[4] = {0,0,0,0};
    int    ndim    = 0;
    bool   owned   = true;

    Tensor() = default;
    Tensor(int d0)                         { alloc({d0});          }
    Tensor(int d0, int d1)                 { alloc({d0, d1});      }
    Tensor(int d0, int d1, int d2)         { alloc({d0, d1, d2});  }
    Tensor(int d0, int d1, int d2, int d3) { alloc({d0, d1, d2, d3}); }

    void alloc(std::initializer_list<int> shape) {
        ndim = (int)shape.size();
        int i = 0;
        for (int s : shape) dims[i++] = s;
        CUDA_CHECK(cudaMalloc(&data, size() * sizeof(float)));
        CUDA_CHECK(cudaMemset(data, 0, size() * sizeof(float)));
        owned = true;
    }

    void set_shape(std::initializer_list<int> shape) {
        ndim = (int)shape.size();
        int i = 0;
        for (int s : shape) dims[i++] = s;
    }

    int size() const {
        int n = 1;
        for (int i = 0; i < ndim; ++i) n *= dims[i];
        return n;
    }

    void free() {
        if (data && owned) { cudaFree(data); }
        data = nullptr;
    }

    void copy_from_host(const float* h, int count) {
        CUDA_CHECK(cudaMemcpy(data, h, count * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void copy_to_host(float* h, int count) const {
        CUDA_CHECK(cudaMemcpy(h, data, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};
