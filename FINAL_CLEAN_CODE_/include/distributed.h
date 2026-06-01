// MPI, NCCL, and overlap communication interfaces for gradient synchronization.
#pragma once

#include "common.h"

#ifdef USE_MPI
#include <mpi.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#ifdef USE_NCCL
#include <nccl.h>
#endif
#endif

struct DistributedContext {
    int world_size = 1;
    int rank       = 0;
    bool use_cuda_aware_mpi = false;
    bool use_nccl = false;
    bool use_comm_thread = false;
    bool defer_gradient_average_to_adam = false;
    double last_allreduce_ms = 0.0;
    double last_sync_ms = 0.0;
    double last_async_start_ms = 0.0;

#ifdef USE_MPI
    struct PinnedHostBuffer {
        float* data = nullptr;
        int capacity = 0;
    };

    struct PendingGradientSync {
        MPI_Request request;
        float* d_buf = nullptr;
        float* h_buf = nullptr;
        int count = 0;
        bool host_staged = false;
        bool nccl = false;
#ifdef USE_NCCL
        cudaEvent_t ready_event = nullptr;
#endif
    };

    struct ThreadedGradientSync {
        float* d_buf = nullptr;
        int count = 0;
        int id = 0;
        cudaEvent_t ready_event = nullptr;
        std::atomic<int> done{0};
    };

    std::vector<PinnedHostBuffer> host_bucket_pool;
    std::vector<PendingGradientSync> pending_gradient_syncs;
    std::vector<ThreadedGradientSync*> pending_threaded_syncs;
    std::deque<ThreadedGradientSync*> threaded_sync_queue;
    std::mutex threaded_sync_mutex;
    std::condition_variable threaded_sync_cv;
    bool comm_worker_shutdown = false;
    int next_threaded_bucket_id = 0;
    int next_host_bucket = 0;
    int mpi_thread_provided = MPI_THREAD_SINGLE;
    int local_device = 0;
#ifdef USE_NCCL
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
#endif
#endif

    bool is_root() const { return rank == 0; }

    void init(int* argc, char*** argv);
    void finalize();

    void allreduce_sum(float* d_buf, int count);

    void sync_gradients(float* d_buf, int count);

    void start_async_gradient_sync(float* d_buf, int count);
    void finish_async_gradient_syncs();

    void request_comm_worker_shutdown();
    void comm_worker_loop();

    double max_double(double value);
    double min_double(double value);

    void barrier();
};
