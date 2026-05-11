// Single-GPU CUDA code for the mini-Transformer.

#pragma once

struct DistributedContext {
    int world_size = 1;
    int rank = 0;
    double last_sync_ms = 0.0;
    double last_async_start_ms = 0.0;

    bool is_root() const { return true; }
    void init(int*, char***) {}
    void finalize() {}
    void sync_gradients(float*, int) {}
    void start_async_gradient_sync(float*, int) {}
    void finish_async_gradient_syncs() {}
    double max_double(double value) { return value; }
    double min_double(double value) { return value; }
    void barrier() {}
};
