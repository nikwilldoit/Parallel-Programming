#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

__global__ void kernel_register(double a, double h, int n, double *partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        double value = (f(x1) + f(x2)) * h / 2.0;
        partial[i] = value;
    }
}

__global__ void reduce_sum(double *input, double *output, int n) {
    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        val = input[i];
    }

    cache[tid] = val;
    __syncthreads();

    int step = blockDim.x / 2;
    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        __syncthreads();
        step /= 2;
    }

    if (tid == 0) {
        output[blockIdx.x] = cache[0];
    }
}

int main() {
    int n = N;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    double *d_partial;
    double *d_reduce;
    double result = 0.0;

    cudaMalloc(&d_partial, n * sizeof(double));
    cudaMalloc(&d_reduce, blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    // 1ος kernel: κάθε thread υπολογίζει 1 trapezoid
    kernel_register<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    // επαναληπτικό reduction στη GPU
    int current_n = n;
    double *d_in = d_partial;
    double *d_out = d_reduce;

    while (current_n > 1) {
        int reduce_blocks = (current_n + threads - 1) / threads;

        reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        double *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    cudaFree(d_partial);
    cudaFree(d_reduce);

    return 0;
}