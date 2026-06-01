// MPI thread-level capability probe for OpenMP communication-thread support.
#include <mpi.h>

#include <cstdio>

static const char* level_name(int level) {
    switch (level) {
        case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
        case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
        case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
        case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    int provided = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int min_provided = provided;
    MPI_Allreduce(&provided, &min_provided, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    printf("rank=%d/%d requested=%s(%d) provided=%s(%d)\n",
           rank, size, level_name(MPI_THREAD_MULTIPLE), MPI_THREAD_MULTIPLE,
           level_name(provided), provided);
    fflush(stdout);

    if (rank == 0) {
        printf("MPI_THREAD_MULTIPLE probe: min_provided=%s(%d) required=%s(%d) result=%s\n",
               level_name(min_provided), min_provided,
               level_name(MPI_THREAD_MULTIPLE), MPI_THREAD_MULTIPLE,
               min_provided >= MPI_THREAD_MULTIPLE ? "PASS" : "FAIL");
        fflush(stdout);
    }

    MPI_Finalize();
    return min_provided >= MPI_THREAD_MULTIPLE ? 0 : 2;
}
