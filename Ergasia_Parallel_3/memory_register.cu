#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

__global__ void kernel_register(double a, double h, double *partial) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    if (i < N) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;

        value = (f(x1) + f(x2)) * h / 2.0;
    }


    partial[i] = value;

}

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    size_t size = N * sizeof(double);

    double *h_partial = (double*)malloc(size);
    double *d_partial;

    cudaMalloc(&d_partial, size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    kernel_register<<<blocks, threads>>>(a, h, d_partial);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_partial, d_partial, size, cudaMemcpyDeviceToHost);


    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += h_partial[i];

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    cudaFree(d_partial);
    free(h_partial);
}