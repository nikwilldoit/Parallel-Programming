#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

__global__ void kernel_shared(double a, double h, double *partial) {

    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    // 1 thread = 1 trapezoid
    if (i < N) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;

        value = (f(x1) + f(x2)) * h * 0.5;
    }

    cache[tid] = value;

    __syncthreads();

    // reduction μέσα στο block
    int step = blockDim.x / 2;

    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        __syncthreads();
        step /= 2;
    }

    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    double *h_partial = (double*)malloc(blocks * sizeof(double));
    double *d_partial;

    cudaMalloc(&d_partial, blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    kernel_shared<<<blocks, threads>>>(a, h, d_partial);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_partial,
               blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < blocks; i++)
        sum += h_partial[i];

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    cudaFree(d_partial);
    free(h_partial);
}