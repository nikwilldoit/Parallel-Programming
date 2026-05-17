#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid
__global__ void trapezoid_1elem(double a, double h, double *partial){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;

        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

int main()
{
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    double *d_partial, *h_partial;

    size_t size = N * sizeof(double);

    h_partial = (double*)malloc(size);
    cudaMalloc(&d_partial, size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    trapezoid_1elem<<<blocks, threads>>>(a, h, d_partial);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(h_partial, d_partial, size, cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += h_partial[i];

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    free(h_partial);

    return 0;
}