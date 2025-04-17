#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024

// Kernel where threads perform different tasks
__global__ void computeSums(int *input, int *output) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Iterative sum
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += input[i];
        }
        output[0] = sum;
    }
    else if (tid == 1) {
        // Formula sum: n(n+1)/2
        int sum = N * (N + 1) / 2;
        output[1] = sum;
    }
}

int main() {
    int h_input[N], h_output[2];

    // Fill input array with 1 to N
    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;
    }

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(int) * N);
    cudaMalloc((void**)&d_output, sizeof(int) * 2);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Launch 2 threads for 2 tasks
    computeSums<<<1, 2>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(int) * 2, cudaMemcpyDeviceToHost);

    std::cout << "Sum using iterative approach (Thread 0): " << h_output[0] << std::endl;
    std::cout << "Sum using formula approach (Thread 1): " << h_output[1] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
