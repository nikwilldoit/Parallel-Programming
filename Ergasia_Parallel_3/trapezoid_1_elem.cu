#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid
__global__ void trapezoid_1elem(double a, double h, int n, double *partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// reduction kernel: αθροίζει πίνακα στη GPU
__global__ void reduce_sum(double *input, double *output, int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    if (i < n)
        sum = input[i];

    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s / 2){
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = N;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;

    double *d_partial;
    double result = 0.0;

    size_t size = n * sizeof(double);

    cudaMalloc(&d_partial, size);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    trapezoid_1elem<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    int current_n = n;
    double *d_in = d_partial;
    double *d_out;

    while (current_n > 1) {
        int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);
        cudaMalloc(&d_out, reduce_blocks * sizeof(double));

        reduce_sum<<<reduce_blocks, threads, threads * sizeof(double)>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        if (d_in != d_partial) {
            cudaFree(d_in);
        }

        d_in = d_out;
        current_n = reduce_blocks;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_in);
    if (d_partial != d_in) {
        cudaFree(d_partial);
    }

    return 0;
}