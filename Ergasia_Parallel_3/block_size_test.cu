#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid
__global__ void trapezoid_kernel(double a, double h, int n, double *partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// reduction kernel με στατικό shared memory
__global__ void reduce_sum(double *input, double *output, int n) {
    // Μέγιστο block size που θα χρησιμοποιήσουμε: 512
    __shared__ double sdata[512];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    if (i < n)
        sum = input[i];
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // απλή block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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

    int block_sizes[] = {32, 64, 128, 256, 512};

    std::cout << "N = " << n << "\n\n";

    for (int threads : block_sizes) {

        // προσοχή: threads <= 512 για να χωράει στο sdata[512]
        int blocks = (n + threads - 1) / threads;

        double *d_partial;
        double *d_tmp;
        double result = 0.0;

        cudaMalloc(&d_partial, n * sizeof(double));
        cudaMalloc(&d_tmp,     n * sizeof(double));

        auto start = std::chrono::high_resolution_clock::now();

        // 1ος kernel: trapezoids
        trapezoid_kernel<<<blocks, threads>>>(a, h, n, d_partial);
        cudaDeviceSynchronize();

        // 2ος kernel: reduction στη GPU με fixed shared
        int current_n = n;
        double *d_in  = d_partial;
        double *d_out = d_tmp;

        while (current_n > 1) {
            int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);

            reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
            cudaDeviceSynchronize();

            current_n = reduce_blocks;

            double *tmp = d_in;
            d_in  = d_out;
            d_out = tmp;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

        std::cout << "Block size: " << threads
                  << " | Time: " << elapsed.count()
                  << " sec | Result: " << result << "\n";

        cudaFree(d_partial);
        cudaFree(d_tmp);
    }

    return 0;
}