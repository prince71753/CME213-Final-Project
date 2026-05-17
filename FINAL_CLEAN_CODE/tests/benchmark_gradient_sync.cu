// mpi gradient synchronization benchmark.
#include "common.h"
#include "distributed.h"
#include "kernels.h"
#include <cstdio>

__global__ void fill_rank_kernel(float* buf, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = value;
}

static void fill_rank(float* d_buf, int n, float value) {
    int block = 256;
    fill_rank_kernel<<<(n + block - 1) / block, block>>>(d_buf, n, value);
    CUDA_CHECK_LAST();
}

static bool check_first_value(float* d_buf, float expected) {
    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, d_buf, sizeof(float), cudaMemcpyDeviceToHost));
    return fabsf(h - expected) < 1e-4f;
}

int main(int argc, char** argv) {
    DistributedContext dist;
    dist.init(&argc, &argv);

    const int repeats = 20;
    const int counts[] = {222592, 838400, 3249664};
    const char* labels[] = {"H128_grad", "H256_grad", "H512_grad"};

    bool overlap = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--overlap") == 0)
            overlap = true;
    }

    if (dist.is_root()) {
        printf("Gradient sync microbenchmark: mode=%s repeats=%d world_size=%d\n",
               overlap ? "pinned_overlap" : "blocking", repeats, dist.world_size);
    }

    for (int s = 0; s < 3; ++s) {
        int count = counts[s];
        float* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, count * sizeof(float)));

        double wall_ms = 0.0;
        double start_ms = 0.0;
        double finish_ms = 0.0;
        bool valid = true;

        for (int r = 0; r < repeats + 2; ++r) {
            fill_rank(d_buf, count, (float)(dist.rank + 1));
            CUDA_CHECK(cudaDeviceSynchronize());
            dist.barrier();

            auto t0 = std::chrono::high_resolution_clock::now();
            if (overlap) {
                dist.start_async_gradient_sync(d_buf, count);
                start_ms += (r >= 2) ? dist.last_async_start_ms : 0.0;
                dist.finish_async_gradient_syncs();
                finish_ms += (r >= 2) ? dist.last_sync_ms : 0.0;
            } else {
                dist.sync_gradients(d_buf, count);
                finish_ms += (r >= 2) ? dist.last_sync_ms : 0.0;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= 2) {
                wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                float expected = 0.5f * (float)(dist.world_size + 1);
                valid = valid && check_first_value(d_buf, expected);
            }
        }

        double denom = (double)repeats;
        double max_wall = dist.max_double(wall_ms / denom);
        double max_start = dist.max_double(start_ms / denom);
        double max_finish = dist.max_double(finish_ms / denom);

        if (dist.is_root()) {
            double mb = count * sizeof(float) / (1024.0 * 1024.0);
            printf("SYNC_BENCH label=%s count=%d mb=%.3f mode=%s wall_ms=%.3f start_ms=%.3f finish_ms=%.3f valid=%s\n",
                   labels[s], count, mb, overlap ? "pinned_overlap" : "blocking",
                   max_wall, max_start, max_finish, valid ? "yes" : "no");
        }

        cudaFree(d_buf);
    }

    dist.finalize();
    return 0;
}
