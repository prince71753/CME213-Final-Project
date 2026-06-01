// MPI, NCCL, pinned-overlap, and OpenMP communication-thread implementation.
#include "distributed.h"
#include "kernels.h"
#include "profiling.h"
#include <vector>

#if defined(USE_MPI) && defined(USE_NCCL)
#define NCCL_CHECK(call) do {                                             \
    ncclResult_t _status = (call);                                        \
    if (_status != ncclSuccess) {                                         \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,    \
                ncclGetErrorString(_status));                             \
        MPI_Abort(MPI_COMM_WORLD, (int)_status);                          \
    }                                                                     \
} while (0)
#endif

#ifdef USE_MPI
static void free_host_bucket_pool(DistributedContext& dist) {
    for (auto& bucket : dist.host_bucket_pool) {
        if (bucket.data) {
            CUDA_CHECK(cudaFreeHost(bucket.data));
            bucket.data = nullptr;
            bucket.capacity = 0;
        }
    }
    dist.host_bucket_pool.clear();
    dist.next_host_bucket = 0;
}
#endif

void DistributedContext::init(int* argc, char*** argv) {
#ifdef USE_MPI
    const char* comm_thread_env = std::getenv("CME213_COMM_THREAD");
    bool comm_thread_requested =
        comm_thread_env && std::strcmp(comm_thread_env, "0") != 0;
    int required_thread_level = comm_thread_requested
        ? MPI_THREAD_MULTIPLE
        : MPI_THREAD_SINGLE;
    MPI_Init_thread(argc, argv, required_thread_level, &mpi_thread_provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    local_device = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(local_device));

    const char* cuda_aware_env = std::getenv("CME213_CUDA_AWARE_MPI");
    use_cuda_aware_mpi = cuda_aware_env && std::strcmp(cuda_aware_env, "0") != 0;
    const char* nccl_env = std::getenv("CME213_USE_NCCL");
    use_nccl = nccl_env && std::strcmp(nccl_env, "0") != 0;
    const char* allow_nccl_env = std::getenv("CME213_ALLOW_UNVALIDATED_NCCL");
    bool allow_unvalidated_nccl =
        allow_nccl_env && std::strcmp(allow_nccl_env, "0") != 0;
    if (use_nccl && !allow_unvalidated_nccl) {
        if (rank == 0) {
            printf("NCCL requested but disabled: validation job 84198 segfaulted; "
                   "falling back to MPI. Set CME213_ALLOW_UNVALIDATED_NCCL=1 "
                   "to force the experimental path.\n");
        }
        use_nccl = false;
    }
#ifndef USE_NCCL
    if (use_nccl) {
        if (rank == 0)
            fprintf(stderr, "CME213_USE_NCCL=1 requested, but this build was compiled without USE_NCCL\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
#endif
    const char* defer_avg_env = std::getenv("CME213_DEFER_GRAD_AVG_TO_ADAM");
    defer_gradient_average_to_adam =
        defer_avg_env && std::strcmp(defer_avg_env, "0") != 0;

#ifdef _OPENMP
    use_comm_thread = comm_thread_requested &&
                      mpi_thread_provided >= MPI_THREAD_MULTIPLE;
#else
    use_comm_thread = false;
#endif
    if (comm_thread_requested && mpi_thread_provided < MPI_THREAD_MULTIPLE &&
        rank == 0) {
        fprintf(stderr,
                "CME213_COMM_THREAD=1 requested, but MPI_THREAD_MULTIPLE "
                "was not provided (provided=%d); disabling comm thread\n",
                mpi_thread_provided);
    }
#ifndef _OPENMP
    if (comm_thread_requested && rank == 0) {
        fprintf(stderr,
                "CME213_COMM_THREAD=1 requested, but this MPI build was "
                "compiled without OpenMP; disabling comm thread\n");
    }
#endif
    if (use_comm_thread && !use_cuda_aware_mpi) {
        if (rank == 0) {
            fprintf(stderr,
                    "CME213_COMM_THREAD=1 requires CUDA-aware MPI device "
                    "buffers; disabling comm thread\n");
        }
        use_comm_thread = false;
    }
    if (use_comm_thread && use_nccl) {
        if (rank == 0) {
            fprintf(stderr,
                    "CME213_COMM_THREAD=1 is an MPI path and cannot be "
                    "combined with NCCL; disabling comm thread\n");
        }
        use_comm_thread = false;
    }
    if (world_size == 1)
        use_comm_thread = false;
    comm_worker_shutdown = false;
    next_threaded_bucket_id = 0;

#ifdef USE_NCCL
    if (use_nccl) {
        ncclUniqueId id;
        if (rank == 0)
            NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, id, rank));
        CUDA_CHECK(cudaStreamCreateWithFlags(&nccl_stream,
                                             cudaStreamNonBlocking));
    }
#endif

    if (rank == 0) {
        printf("MPI: %d processes, %d GPUs per node\n", world_size, num_devices);
        printf("MPI gradient backend: %s\n",
               use_nccl ? "NCCL device allreduce"
                        : (use_cuda_aware_mpi ? "CUDA-aware direct device buffers"
                                              : "portable host-staged buffers"));
        printf("MPI gradient averaging: %s\n",
               defer_gradient_average_to_adam
                   ? "deferred to Adam update"
                   : "post-allreduce scale kernels");
        printf("MPI thread level provided: %d%s\n",
               mpi_thread_provided,
               mpi_thread_provided >= MPI_THREAD_MULTIPLE
                   ? " (MPI_THREAD_MULTIPLE)"
                   : "");
        printf("MPI OpenMP comm thread: %s\n",
               use_comm_thread ? "enabled" : "disabled");
    }
#else
    (void)argc; (void)argv;
    world_size = 1;
    rank = 0;
#endif
}

void DistributedContext::finalize() {
#ifdef USE_MPI
    request_comm_worker_shutdown();
#ifdef USE_NCCL
    if (use_nccl) {
        if (nccl_stream)
            CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
        if (nccl_comm) {
            NCCL_CHECK(ncclCommDestroy(nccl_comm));
            nccl_comm = nullptr;
        }
        if (nccl_stream) {
            CUDA_CHECK(cudaStreamDestroy(nccl_stream));
            nccl_stream = nullptr;
        }
    }
#endif
    free_host_bucket_pool(*this);
    for (auto* pending : pending_threaded_syncs) {
        if (pending) {
            if (pending->ready_event)
                cudaEventDestroy(pending->ready_event);
            delete pending;
        }
    }
    pending_threaded_syncs.clear();
    MPI_Finalize();
#endif
}

void DistributedContext::allreduce_sum(float* d_buf, int count) {
#ifdef USE_MPI
    if (world_size == 1) return;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef USE_NCCL
    if (use_nccl) {
        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum,
                                 nccl_comm, nccl_stream));
        CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        last_allreduce_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }
#endif

    if (use_cuda_aware_mpi) {
        MPI_Allreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
    } else {
        std::vector<float> h_buf(count);
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, count * sizeof(float),
                              cudaMemcpyDeviceToHost));

        MPI_Allreduce(MPI_IN_PLACE, h_buf.data(), count, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), count * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    last_allreduce_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
#else
    (void)d_buf; (void)count;
#endif
}

void DistributedContext::sync_gradients(float* d_buf, int count) {
#ifdef USE_MPI
    if (world_size == 1) return;
    last_async_start_ms = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    allreduce_sum(d_buf, count);
    if (!defer_gradient_average_to_adam) {
        scale_buffer(d_buf, 1.0f / (float)world_size, count);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    last_sync_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
#else
    (void)d_buf; (void)count;
#endif
}

void DistributedContext::start_async_gradient_sync(float* d_buf, int count) {
#ifdef USE_MPI
    if (world_size == 1 || count <= 0) return;

    if (pending_gradient_syncs.empty() && pending_threaded_syncs.empty()) {
        last_async_start_ms = 0.0;
        next_host_bucket = 0;
        next_threaded_bucket_id = 0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    const char* async_device_env = std::getenv("CME213_ASYNC_DEVICE_MPI");
    bool use_device_async = use_cuda_aware_mpi && async_device_env &&
                            std::strcmp(async_device_env, "0") != 0;

    if (use_comm_thread) {
        ThreadedGradientSync* pending = new ThreadedGradientSync();
        pending->d_buf = d_buf;
        pending->count = count;
        pending->id = next_threaded_bucket_id++;
        CUDA_CHECK(cudaEventCreateWithFlags(&pending->ready_event,
                                            cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(pending->ready_event, cudaStreamPerThread));
        {
            std::lock_guard<std::mutex> lock(threaded_sync_mutex);
            pending_threaded_syncs.push_back(pending);
            threaded_sync_queue.push_back(pending);
        }
        threaded_sync_cv.notify_one();
        auto t1 = std::chrono::high_resolution_clock::now();
        last_async_start_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }

    pending_gradient_syncs.emplace_back();
    PendingGradientSync& pending = pending_gradient_syncs.back();
    pending.d_buf = d_buf;
    pending.count = count;

#ifdef USE_NCCL
    if (use_nccl) {
        pending.nccl = true;
        CUDA_CHECK(cudaEventCreateWithFlags(&pending.ready_event,
                                            cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(pending.ready_event, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, pending.ready_event, 0));
        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum,
                                 nccl_comm, nccl_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        last_async_start_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }
#endif

    if (use_device_async) {
        pending.host_staged = false;
        MPI_Iallreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM,
                       MPI_COMM_WORLD, &pending.request);
        auto t1 = std::chrono::high_resolution_clock::now();
        last_async_start_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }

    pending.host_staged = true;
    if (next_host_bucket >= (int)host_bucket_pool.size())
        host_bucket_pool.emplace_back();

    PinnedHostBuffer& host_bucket = host_bucket_pool[next_host_bucket++];
    if (host_bucket.capacity < count) {
        if (host_bucket.data)
            CUDA_CHECK(cudaFreeHost(host_bucket.data));
        CUDA_CHECK(cudaMallocHost(&host_bucket.data, count * sizeof(float)));
        host_bucket.capacity = count;
    }

    pending.h_buf = host_bucket.data;
    CUDA_CHECK(cudaMemcpyAsync(pending.h_buf, d_buf, count * sizeof(float),
                               cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    MPI_Iallreduce(MPI_IN_PLACE, pending.h_buf, count, MPI_FLOAT,
                   MPI_SUM, MPI_COMM_WORLD, &pending.request);
    auto t1 = std::chrono::high_resolution_clock::now();
    last_async_start_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();
#else
    (void)d_buf; (void)count;
#endif
}

void DistributedContext::finish_async_gradient_syncs() {
#ifdef USE_MPI
    if (world_size == 1) return;
    if (pending_gradient_syncs.empty() && pending_threaded_syncs.empty()) return;

    NvtxRange range("finish_async_gradient_syncs");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!pending_threaded_syncs.empty()) {
        for (auto* pending : pending_threaded_syncs) {
            std::unique_lock<std::mutex> lock(threaded_sync_mutex);
            threaded_sync_cv.wait(lock, [&]() {
                return pending->done.load(std::memory_order_acquire) != 0;
            });
        }
        for (auto* pending : pending_threaded_syncs) {
            if (!defer_gradient_average_to_adam) {
                scale_buffer(pending->d_buf, 1.0f / (float)world_size,
                             pending->count);
            }
            if (pending->ready_event) {
                CUDA_CHECK(cudaEventDestroy(pending->ready_event));
                pending->ready_event = nullptr;
            }
            delete pending;
        }
        pending_threaded_syncs.clear();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        last_sync_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }

    bool has_nccl = false;
    for (const auto& pending : pending_gradient_syncs)
        has_nccl = has_nccl || pending.nccl;
#ifdef USE_NCCL
    if (has_nccl)
        CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
#endif
    for (auto& pending : pending_gradient_syncs) {
        if (pending.nccl) {
#ifdef USE_NCCL
            if (pending.ready_event) {
                CUDA_CHECK(cudaEventDestroy(pending.ready_event));
                pending.ready_event = nullptr;
            }
#endif
        } else {
            MPI_Wait(&pending.request, MPI_STATUS_IGNORE);
            if (pending.host_staged) {
                CUDA_CHECK(cudaMemcpy(pending.d_buf, pending.h_buf,
                                      pending.count * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
        }
        if (!defer_gradient_average_to_adam) {
            scale_buffer(pending.d_buf, 1.0f / (float)world_size,
                         pending.count);
        }
    }
    pending_gradient_syncs.clear();
    next_host_bucket = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    last_sync_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
}

void DistributedContext::request_comm_worker_shutdown() {
#ifdef USE_MPI
    {
        std::lock_guard<std::mutex> lock(threaded_sync_mutex);
        comm_worker_shutdown = true;
    }
    threaded_sync_cv.notify_all();
#endif
}

void DistributedContext::comm_worker_loop() {
#ifdef USE_MPI
    if (!use_comm_thread)
        return;
    CUDA_CHECK(cudaSetDevice(local_device));

    const int kBatchThresholdBytes = 128 * 1024;

    while (true) {
        std::vector<ThreadedGradientSync*> batch;
        int batch_size_bytes = 0;

        {
            std::unique_lock<std::mutex> lock(threaded_sync_mutex);
            threaded_sync_cv.wait(lock, [&]() {
                return comm_worker_shutdown || !threaded_sync_queue.empty();
            });
            if (threaded_sync_queue.empty()) {
                if (comm_worker_shutdown)
                    break;
                continue;
            }

            ThreadedGradientSync* first = threaded_sync_queue.front();
            threaded_sync_queue.pop_front();
            batch.push_back(first);
            batch_size_bytes = first->count * sizeof(float);

            if (batch_size_bytes >= kBatchThresholdBytes) {
                lock.unlock();
            } else {
                while (!threaded_sync_queue.empty()) {
                    ThreadedGradientSync* next = threaded_sync_queue.front();
                    int next_size = next->count * sizeof(float);
                    if (batch_size_bytes + next_size > kBatchThresholdBytes * 2)
                        break;
                    threaded_sync_queue.pop_front();
                    batch.push_back(next);
                    batch_size_bytes += next_size;
                }
                lock.unlock();
            }
        }

        for (auto* pending : batch) {
            {
                NvtxRange range("openmp_comm_event_wait");
                CUDA_CHECK(cudaEventSynchronize(pending->ready_event));
            }
            {
                NvtxRange range("openmp_comm_mpi_allreduce");
                int status = MPI_Allreduce(MPI_IN_PLACE, pending->d_buf,
                                           pending->count, MPI_FLOAT, MPI_SUM,
                                           MPI_COMM_WORLD);
                if (status != MPI_SUCCESS)
                    MPI_Abort(MPI_COMM_WORLD, status);
            }
            pending->done.store(1, std::memory_order_release);
        }
        threaded_sync_cv.notify_all();
    }
#endif
}

double DistributedContext::max_double(double value) {
#ifdef USE_MPI
    double result = value;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return value;
#endif
}

double DistributedContext::min_double(double value) {
#ifdef USE_MPI
    double result = value;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return result;
#else
    return value;
#endif
}

void DistributedContext::barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
