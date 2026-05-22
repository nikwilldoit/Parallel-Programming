#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

//1 thread = 1 trapezoid
__global__ void trapezoid_1elem(double a, double h, int n, double *partial) {
    //index of the thread in a 1D grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        //local trapezoid contribution in global memory
        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

//reduction kernel that sums an array on the GPU using shared memory
__global__ void reduce_sum(double *input, double *output, int n) {
    __shared__ double sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    //load first element if it is in range
    if (i < n)
        sum = input[i];

    //load second element if it is also in range
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    //write partial sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    //parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s = s / 2){
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //thread 0 of each block writes the blocks sum to global memory
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

    //allocate global memory on the device for threads partial results
    cudaMalloc(&d_partial, size);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    //starts total GPU timing (compute + all reductions)
    auto start = std::chrono::high_resolution_clock::now();

    //first kernel that compute one trapezoid per thread
    trapezoid_1elem<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    //iterative reduction on the GPU until only one value remains
    int current_n = n;
    double *d_in = d_partial;
    double *d_out;

    while (current_n > 1) {
        int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);
        cudaMalloc(&d_out, reduce_blocks * sizeof(double));

        //reduction kernel
        reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        if (d_in != d_partial) {
            cudaFree(d_in);
        }

        d_in = d_out;
        current_n = reduce_blocks;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //copy of the final result from device to host
    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_in);
    if (d_partial != d_in) {
        cudaFree(d_partial);
    }

    return 0;
}