#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

__global__ void kernel_global(double a, double h, int n, double *partial) {
    //global index of the thread in a 1D grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //each thread computes one trapezoid
    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

//block-level reduction using shared memory
__global__ void reduce_sum(double *input, double *output, int n) {
    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    // load element from global memory if in range
    if (i < n) {
        val = input[i];
    }

    //store to shared memory
    cache[tid] = val;
    __syncthreads();

    //parallel reduction in shared memory
    int step = blockDim.x / 2;
    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        __syncthreads();
        step /= 2;
    }

    //thread 0 writes blocks partial sum to global memory
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

    //each thread computes one trapezoid and stores it in global memory
    kernel_global<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    //iterative reduction on the GPU until only one value remains
    int current_n = n;
    double *d_in = d_partial;
    double *d_out = d_reduce;

    while (current_n > 1) {
        int reduce_blocks = (current_n + threads - 1) / threads;

        reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        //swap input/output buffers for the next reduction stage
        double *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    auto end = std::chrono::high_resolution_clock::now();

    //copy final result back to host
    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    cudaFree(d_partial);
    cudaFree(d_reduce);

    return 0;
}