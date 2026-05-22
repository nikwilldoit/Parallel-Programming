#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

//1 thread = 1 trapezoid, block-level reduction using __syncthreads()
__global__ void trapezoid_barrier(double a, double h, int n, double *partial) {

    //shared memory buffer for per-thread partial sums inside the block
    __shared__ double cache[256];  //blockDim.x = 256

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    //each thread computes one trapezoid if its index is in range
    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5;
    }

    //write local contribution to shared memory
    cache[tid] = value;

    //wait for all threads to finish writing into cache[]
    __syncthreads();

    //parallel reduction in shared memory
    int step = blockDim.x / 2;
    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        //ensure all partial updates before next step
        __syncthreads();
        step /= 2;
    }

    //thread 0 writes the blocks final sum to global memory
    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

//reduction over the block partial sums
__global__ void final_reduce(double *input, double *output, int n) {

    //shared memory buffer for this reduction stage
    __shared__ double cache[256];  //blockDim.x = 256

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

    int threads = 256; //threads per block
    int blocks = (n + threads - 1) / threads; //number of blocks

    double *d_partial;   //block-level sums from the first kernel
    double *d_reduce;    //temporary buffer for the second kernel
    double result = 0.0;

    //one partial sum per block produced by trapezoid_barrier
    cudaMalloc(&d_partial, blocks * sizeof(double));
    cudaMalloc(&d_reduce,  blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    //compute trapezoids + block-level reduction using shared memory
    trapezoid_barrier<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    //block partial sums until only one value remains
    int current_n = blocks;
    double *d_in  = d_partial;
    double *d_out = d_reduce;

    while (current_n > 1) {
        int reduce_blocks = (current_n + threads - 1) / threads;

        final_reduce<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        current_n = reduce_blocks;

        //swap input and output buffers for the next iteration
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
    cudaFree(d_reduce);

    return 0;
}