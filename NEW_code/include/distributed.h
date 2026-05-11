#pragma once

#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>

// Safe CUDA macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

struct DistributedContext {
    int world_size = 1;
    int rank = 0;
    double last_sync_ms = 0.0;
    double last_async_start_ms = 0.0;

#ifdef USE_MPI
    struct GradientBucket {
        std::vector<float> host_buf;
        MPI_Request req;
        float* d_buf = nullptr;
        int count = 0;
    };

    std::vector<GradientBucket> pending_buckets;
    MPI_Comm comm = MPI_COMM_WORLD;
#endif

    bool is_root() const { return rank == 0; }

    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        MPI_Init(argc, argv);
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &rank);
#endif
    }

    void finalize() {
#ifdef USE_MPI
        finish_async_gradient_syncs();
        MPI_Finalize();
#endif
    }

    // ======================
    // Blocking gradient sync
    // ======================
    void sync_gradients(float* d_grads, int num_floats) {
#ifdef USE_MPI
        if (world_size == 1 || num_floats <= 0) return;

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_grads(num_floats);
        CUDA_CHECK(cudaMemcpy(h_grads.data(), d_grads,
                              num_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));

        MPI_Allreduce(MPI_IN_PLACE, h_grads.data(),
                      num_floats, MPI_FLOAT, MPI_SUM, comm);

        float scale = 1.0f / world_size;
        for (int i = 0; i < num_floats; ++i)
            h_grads[i] *= scale;

        CUDA_CHECK(cudaMemcpy(d_grads, h_grads.data(),
                              num_floats * sizeof(float),
                              cudaMemcpyHostToDevice));
#endif
    }

    // ======================
    // Async gradient sync
    // ======================
    void start_async_gradient_sync(float* d_buf, int count) {
#ifdef USE_MPI
        if (world_size == 1 || count <= 0) return;

        CUDA_CHECK(cudaDeviceSynchronize());

        pending_buckets.emplace_back();
        auto& bucket = pending_buckets.back();

        bucket.host_buf.resize(count);
        bucket.d_buf = d_buf;
        bucket.count = count;

        CUDA_CHECK(cudaMemcpy(bucket.host_buf.data(), d_buf,
                              count * sizeof(float),
                              cudaMemcpyDeviceToHost));

        MPI_Iallreduce(MPI_IN_PLACE, bucket.host_buf.data(),
                       count, MPI_FLOAT, MPI_SUM, comm, &bucket.req);
#endif
    }

    void finish_async_gradient_syncs() {
#ifdef USE_MPI
        if (pending_buckets.empty() || world_size == 1) return;

        std::vector<MPI_Request> requests;
        requests.reserve(pending_buckets.size());

        for (auto& bucket : pending_buckets)
            requests.push_back(bucket.req);

        std::vector<MPI_Status> statuses(requests.size());

        MPI_Waitall((int)requests.size(),
                    requests.data(),
                    statuses.data());

        float scale = 1.0f / world_size;

        for (auto& bucket : pending_buckets) {
            for (int i = 0; i < bucket.count; ++i)
                bucket.host_buf[i] *= scale;

            CUDA_CHECK(cudaMemcpy(bucket.d_buf,
                                  bucket.host_buf.data(),
                                  bucket.count * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        pending_buckets.clear();
#endif
    }

    // ======================
    // Parameter consistency
    // ======================
    uint64_t compute_parameter_checksum(const float* d_params, int num_params) {
        std::vector<float> h_params(num_params);

        CUDA_CHECK(cudaMemcpy(h_params.data(), d_params,
                              num_params * sizeof(float),
                              cudaMemcpyDeviceToHost));

        uint64_t checksum = 0;

        for (int i = 0; i < num_params; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &h_params[i], sizeof(float));
            checksum ^= (uint64_t)bits * 0x9e3779b97f4a7c15ULL;
        }

        return checksum;
    }

    bool verify_parameter_consistency(const float* d_params, int num_params) {
#ifdef USE_MPI
        if (world_size == 1) return true;

        uint64_t local_checksum = compute_parameter_checksum(d_params, num_params);
        uint64_t global_checksum = 0;

        MPI_Allreduce(&local_checksum, &global_checksum,
                      1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

        bool consistent = (local_checksum == global_checksum);
        int all_consistent = consistent ? 1 : 0;

        MPI_Allreduce(MPI_IN_PLACE, &all_consistent,
                      1, MPI_INT, MPI_MIN, comm);

        return all_consistent == 1;
#else
        return true;
#endif
    }

    // ======================
    // Helpers
    // ======================
    double max_double(double value) {
#ifdef USE_MPI
        double result;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, comm);
        return result;
#else
        return value;
#endif
    }

    double min_double(double value) {
#ifdef USE_MPI
        double result;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MIN, comm);
        return result;
#else
        return value;
#endif
    }

    void barrier() {
#ifdef USE_MPI
        MPI_Barrier(comm);
#endif
    }
};