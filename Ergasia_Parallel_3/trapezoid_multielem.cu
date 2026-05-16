#include <iostream>
#include <cuda.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread processes MANY trapezoids
__global__ void trapezoid_multielem(double a, double h, double *partial){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (int i = tid; i < N; i += step) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;

        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    partial[tid] = local_sum;
}

int main()
{
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threads = 256;
    int blocks = 256;

    int totalThreads = threads * blocks;

    double *d_partial, *h_partial;

    h_partial = (double*)malloc(totalThreads * sizeof(double));
    cudaMalloc(&d_partial, totalThreads * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    trapezoid_multielem<<<blocks, threads>>>(a, h, d_partial);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_partial,
               totalThreads * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < totalThreads; i++)
        sum += h_partial[i];

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    free(h_partial);

    return 0;
}