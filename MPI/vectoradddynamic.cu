#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024 * 1024
#define THREADS_PER_BLOCK 256

__global__ void vectorAddKernel(float* A, float* B, float* C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAddKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float memClockKHz = static_cast<float>(prop.memoryClockRate);
    float memBusWidth = static_cast<float>(prop.memoryBusWidth);

    float theoreticalBW = 2.0f * memClockKHz * 1000 * (memBusWidth / 8.0f) / (1 << 30);
    std::cout << "Theoretical Memory Bandwidth: " << theoreticalBW << " GB/s" << std::endl;

    float RBytes = size * 2;
    float WBytes = size;
    float timeInSeconds = milliseconds / 1000.0f;

    float measuredBW = (RBytes + WBytes) / (timeInSeconds * (1 << 30));
    std::cout << "Measured Memory Bandwidth: " << measuredBW << " GB/s" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
