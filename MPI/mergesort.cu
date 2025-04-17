#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define N 1024

__device__ void swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

// Bitonic sort kernel
__global__ void bitonicSortKernel(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                swap(arr[i], arr[ixj]);
            }
        }
        else {
            if (arr[i] < arr[ixj]) {
                swap(arr[i], arr[ixj]);
            }
        }
    }
}

void bitonicSort(int *arr) {
    int *d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / 512);
    dim3 threads(512);

    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<blocks, threads>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[N];
    for (int i = 0; i < N; ++i) arr[i] = rand() % 10000;

    auto start = std::chrono::high_resolution_clock::now();
    bitonicSort(arr);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "CUDA Bitonic Sort Time: " << duration.count() << " ms\n";

    return 0;
}
