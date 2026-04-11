#include <iostream>

__global__ void hello_kernel() {
    printf("TEST: GPU IS COMING, GPU IS HERE\n");
}

int main() {
    std::cout << "TEST: CPU INCOMING\n";

    hello_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}