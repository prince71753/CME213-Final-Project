// MPI/NCCL Allreduce message-size sweep for alpha/beta fitting.
#include <mpi.h>
#include <cuda_runtime.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void check_cuda(cudaError_t status, const char* expr, const char* file,
                       int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                     expr, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

#ifdef USE_NCCL
static void check_nccl(ncclResult_t status, const char* expr, const char* file,
                       int line) {
    if (status != ncclSuccess) {
        std::fprintf(stderr, "NCCL error at %s:%d for %s: %s\n", file, line,
                     expr, ncclGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

#define CHECK_NCCL(expr) check_nccl((expr), #expr, __FILE__, __LINE__)
#endif

__global__ void fill_float_kernel(float* buf, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = value;
}

static void fill_device(float* d_buf, int n, float value) {
    int block = 256;
    fill_float_kernel<<<(n + block - 1) / block, block>>>(d_buf, n, value);
    CHECK_CUDA(cudaGetLastError());
}

static bool check_device_first(float* d_buf, float expected) {
    float got = 0.0f;
    CHECK_CUDA(cudaMemcpy(&got, d_buf, sizeof(float), cudaMemcpyDeviceToHost));
    float tol = 1e-4f * std::max(1.0f, std::fabs(expected));
    return std::isfinite(got) && std::fabs(got - expected) <= tol;
}

static size_t parse_size(const char* value) {
    char* end = nullptr;
    double number = std::strtod(value, &end);
    if (end == value || number <= 0.0)
        return 0;
    if (*end == '\0')
        return static_cast<size_t>(number);
    if (std::strcmp(end, "K") == 0 || std::strcmp(end, "KB") == 0 ||
        std::strcmp(end, "k") == 0 || std::strcmp(end, "kb") == 0)
        return static_cast<size_t>(number * 1024.0);
    if (std::strcmp(end, "M") == 0 || std::strcmp(end, "MB") == 0 ||
        std::strcmp(end, "m") == 0 || std::strcmp(end, "mb") == 0)
        return static_cast<size_t>(number * 1024.0 * 1024.0);
    return static_cast<size_t>(number);
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--backend device|host_pinned|nccl] [--min-bytes N] "
                 "[--max-bytes N] [--iters N] [--warmup N]\n",
                 argv0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    std::string backend = "device";
    size_t min_bytes = 4 * 1024;
    size_t max_bytes = 32 * 1024 * 1024;
    int iters = 50;
    int warmup = 10;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (std::strcmp(argv[i], "--min-bytes") == 0 && i + 1 < argc) {
            min_bytes = parse_size(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) {
            max_bytes = parse_size(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            if (rank == 0)
                usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0)
                usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (min_bytes == 0 || max_bytes < min_bytes || iters <= 0 || warmup < 0) {
        if (rank == 0)
            usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0)
            std::fprintf(stderr, "No CUDA devices visible\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    CHECK_CUDA(cudaSetDevice(rank % device_count));

    const bool host_pinned =
        backend == "host" || backend == "host_pinned" || backend == "pinned";
    const bool use_nccl = backend == "nccl";
    if (!host_pinned && backend != "device" && !use_nccl) {
        if (rank == 0)
            std::fprintf(stderr, "unknown backend: %s\n", backend.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (host_pinned)
        backend = "host_pinned";
#ifndef USE_NCCL
    if (use_nccl) {
        if (rank == 0)
            std::fprintf(stderr, "backend=nccl requested, but this binary was built without USE_NCCL\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
#endif

#ifdef USE_NCCL
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
    if (use_nccl) {
        ncclUniqueId id;
        if (rank == 0)
            CHECK_NCCL(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        CHECK_NCCL(ncclCommInitRank(&nccl_comm, world, id, rank));
        CHECK_CUDA(cudaStreamCreateWithFlags(&nccl_stream,
                                             cudaStreamNonBlocking));
    }
#endif

    std::vector<size_t> sizes;
    for (size_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        sizes.push_back(bytes);
        if (bytes > max_bytes / 2)
            break;
    }
    if (sizes.empty() || sizes.back() != max_bytes)
        sizes.push_back(max_bytes);

    if (rank == 0) {
        std::printf("allreduce_alpha_beta,backend,ranks,bytes,count,iteration,time_ms\n");
        std::fflush(stdout);
    }

    for (size_t requested_bytes : sizes) {
        size_t count64 = std::max<size_t>(1, requested_bytes / sizeof(float));
        if (count64 > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0)
                std::fprintf(stderr, "message too large for MPI count: %zu\n",
                             count64);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int count = static_cast<int>(count64);
        size_t bytes = static_cast<size_t>(count) * sizeof(float);

        float* h_send = nullptr;
        float* h_recv = nullptr;
        float* d_send = nullptr;
        float* d_recv = nullptr;

        if (host_pinned) {
            CHECK_CUDA(cudaMallocHost(&h_send, bytes));
            CHECK_CUDA(cudaMallocHost(&h_recv, bytes));
            for (int i = 0; i < count; ++i)
                h_send[i] = static_cast<float>(rank + 1);
        } else {
            CHECK_CUDA(cudaMalloc(&d_send, bytes));
            CHECK_CUDA(cudaMalloc(&d_recv, bytes));
            fill_device(d_send, count, static_cast<float>(rank + 1));
            CHECK_CUDA(cudaMemset(d_recv, 0, bytes));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        const int total_iters = warmup + iters;
        bool valid = true;
        for (int iter = 0; iter < total_iters; ++iter) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (!host_pinned)
                CHECK_CUDA(cudaDeviceSynchronize());
            double t0 = MPI_Wtime();
#ifdef USE_NCCL
            if (use_nccl) {
                CHECK_NCCL(ncclAllReduce(d_send, d_recv, count, ncclFloat,
                                         ncclSum, nccl_comm, nccl_stream));
                CHECK_CUDA(cudaStreamSynchronize(nccl_stream));
            } else
#endif
            {
                int status = MPI_Allreduce(host_pinned ? h_send : d_send,
                                           host_pinned ? h_recv : d_recv,
                                           count, MPI_FLOAT, MPI_SUM,
                                           MPI_COMM_WORLD);
                if (status != MPI_SUCCESS)
                    MPI_Abort(MPI_COMM_WORLD, status);
            }
            if (!host_pinned)
                CHECK_CUDA(cudaDeviceSynchronize());
            double local_ms = (MPI_Wtime() - t0) * 1000.0;
            if (iter == total_iters - 1) {
                const float expected = 0.5f * world * (world + 1);
                if (host_pinned) {
                    float tol = 1e-4f * std::max(1.0f, std::fabs(expected));
                    valid = valid && std::isfinite(h_recv[0]) &&
                            std::fabs(h_recv[0] - expected) <= tol;
                } else {
                    valid = valid && check_device_first(d_recv, expected);
                }
            }
            double max_ms = 0.0;
            MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD);
            if (rank == 0 && iter >= warmup) {
                std::printf("allreduce_alpha_beta,%s,%d,%zu,%d,%d,%.6f\n",
                            backend.c_str(), world, bytes, count,
                            iter - warmup + 1, max_ms);
            }
        }
        int local_valid = valid ? 1 : 0;
        int all_valid = 0;
        MPI_Reduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("allreduce_alpha_beta_valid,%s,%d,%zu,%s\n",
                        backend.c_str(), world, bytes,
                        all_valid ? "yes" : "no");
        }

        if (host_pinned) {
            CHECK_CUDA(cudaFreeHost(h_send));
            CHECK_CUDA(cudaFreeHost(h_recv));
        } else {
            CHECK_CUDA(cudaFree(d_send));
            CHECK_CUDA(cudaFree(d_recv));
        }
    }

#ifdef USE_NCCL
    if (use_nccl) {
        CHECK_CUDA(cudaStreamSynchronize(nccl_stream));
        CHECK_NCCL(ncclCommDestroy(nccl_comm));
        CHECK_CUDA(cudaStreamDestroy(nccl_stream));
    }
#endif

    MPI_Finalize();
    return 0;
}
