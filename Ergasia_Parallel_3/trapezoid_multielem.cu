#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

//1 thread processes many trapezoids
__global__ void trapezoid_multielem(double a, double h, int n, double *partial) {
    //thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //number of threads in the grid
    int step = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (int i = tid; i < n; i += step) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    //each thread writes a single partial sum to global memory
    if (tid < n) {
        partial[tid] = local_sum;
    }
}

//reduction kernel that sums an array on the GPU using shared memory
__global__ void reduce_sum(double *input, double *output, int n) {
    __shared__ double sdata[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    //load first element if it is in range
    if (i < n)
        sum = input[i];

    //load second element if it is in range
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    //store value in shared memory
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

    int threads = 256;
    int blocks = 256;
    int totalThreads = blocks * threads;

    double *d_partial = nullptr; //buffer for per-thread partial sums (1 per thread)
    double *d_tmp = nullptr; //temporary buffer used during reduction
    double  result = 0.0;

    //allocate global memory for partial sums from the first kernel
    cudaMalloc(&d_partial, totalThreads * sizeof(double));
    //allocate second buffer for the reduction stages
    cudaMalloc(&d_tmp, totalThreads * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    //many elements per thread that writes one partial sum per thread
    trapezoid_multielem<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    //reduction on the GPU
    int current_n = totalThreads; //number of partials
    double *d_in = d_partial; //current input buffer
    double *d_out = d_tmp; //current output buffer

    while (current_n > 1) {
        int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);
        int shmem_bytes   = threads * sizeof(double);

        //perform one reduction step
        reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        //swap roles of input and output buffers without extra allocations
        double *tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //copy final result from device to host
    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    cudaFree(d_tmp);

    return 0;
}