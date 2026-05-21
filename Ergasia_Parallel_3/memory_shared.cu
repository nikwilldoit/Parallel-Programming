#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid, block-level reduction σε shared
__global__ void kernel_shared(double a, double h, int n, double *partial) {

    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h / 2.0;
    }

    cache[tid] = value;

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
        partial[blockIdx.x] = cache[0];
    }
}

// 2ος kernel: reduction στα block partial sums (πάλι με shared)
__global__ void final_reduce(double *input, double *output, int n) {

    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

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
    int blocks  = (n + threads - 1) / threads;

    double *d_partial;   // block-level sums από kernel_shared
    double *d_reduce;    // προσωρινά για final_reduce
    double result = 0.0;

    cudaMalloc(&d_partial, blocks * sizeof(double));
    cudaMalloc(&d_reduce,  blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    // 1ος kernel: trapezoids + block-level reduction σε shared
    kernel_shared<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    // 2ος kernel: reduction των block sums μέχρι να μείνει 1 τιμή
    int current_n = blocks;
    double *d_in  = d_partial;
    double *d_out = d_reduce;

    while (current_n > 1) {
        int reduce_blocks = (current_n + threads - 1) / threads;

        final_reduce<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        double *tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
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