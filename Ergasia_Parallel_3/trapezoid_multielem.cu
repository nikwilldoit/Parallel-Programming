#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000   // προς το παρόν, αλλά ο kernel δεν το εξαρτάται

__device__ double f(double x) {
    return x * x;
}

// 1 thread processes MANY trapezoids (grid-stride loop)
__global__ void trapezoid_multielem(double a, double h, int n, double *partial) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (int i = tid; i < n; i += step) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    if (tid < n) {
        partial[tid] = local_sum;
    }
}

// reduction kernel: άθροιση στη GPU (ίδιο pattern με πριν)
__global__ void reduce_sum(double *input, double *output, int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

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

    int threads = 256;
    int blocks  = 256;                // εδώ είναι “πειραματική” επιλογή για grid-stride
    int totalThreads = blocks * threads;

    double *d_partial = nullptr;
    double *d_tmp     = nullptr;
    double  result    = 0.0;

    // buffer για partial sums από τον πρώτο kernel
    cudaMalloc(&d_partial, totalThreads * sizeof(double));
    // δεύτερο buffer για reduction
    cudaMalloc(&d_tmp, totalThreads * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    // 1ος kernel: πολλά στοιχεία ανά thread, γράφει ένα partial ανά thread
    trapezoid_multielem<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    // 2ος kernel: reduction στη GPU
    int current_n = totalThreads;
    double *d_in  = d_partial;
    double *d_out = d_tmp;

    while (current_n > 1) {
        int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);
        int shmem_bytes   = threads * sizeof(double);

        reduce_sum<<<reduce_blocks, threads, shmem_bytes>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        // swap ρόλους χωρίς νέα cudaMalloc
        double *tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // μόνο 1 double αντιγράφεται στη CPU
    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    cudaFree(d_tmp);

    return 0;
}