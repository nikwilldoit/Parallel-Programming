#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid
__global__ void trapezoid_kernel(double a, double h, int n, double *partial) {
    //index of the thread in a 1D grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        //stores local trapezoid contribution in global memory
        partial[i] = (f(x1) + f(x2)) * h / 2.0;
    }
}

//reduction kernel
__global__ void reduce_sum(double *input, double *output, int n) {
    //maximum block size used in this experiment is 512
    __shared__ double sdata[512];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    //load first element if it is in range
    if (i < n)
        sum = input[i];
    
    //load second element if it is in range
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    //write combined value to shared memory
    sdata[tid] = sum;
    __syncthreads();

    //simple block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //thread 0 writes the blocks partial sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = N;
    double a = 0.0, b = 10.0;
    double h = (b - a) / n;

    //set of block sizes for test
    int block_sizes[] = {32, 64, 128, 256, 512};

    std::cout << "N = " << n << "\n\n";

    for (int threads : block_sizes) {

        //constraint for threads <= 512 so that sdata[512] is sufficient
        int blocks = (n + threads - 1) / threads;

        double *d_partial;
        double *d_tmp;
        double result = 0.0;

        //allocate device memory for per-thread partial results
        cudaMalloc(&d_partial, n * sizeof(double));
        //temporary buffer for reductions
        cudaMalloc(&d_tmp, n * sizeof(double));

        auto start = std::chrono::high_resolution_clock::now();

        //1st kernel that compute one trapezoid per thread
        trapezoid_kernel<<<blocks, threads>>>(a, h, n, d_partial);
        cudaDeviceSynchronize();

        //reduction on the GPU with shared memory
        int current_n = n;
        double *d_in = d_partial;
        double *d_out = d_tmp;

        while (current_n > 1) {
            int reduce_blocks = (current_n + (threads * 2 - 1)) / (threads * 2);

            reduce_sum<<<reduce_blocks, threads>>>(d_in, d_out, current_n);
            cudaDeviceSynchronize();

            current_n = reduce_blocks;

            //swap input and output buffers for the next iteration
            double *tmp = d_in;
            d_in = d_out;
            d_out = tmp;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

         //copy the final result back to the host
        cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

        std::cout << "Block size: " << threads
                  << " | Time: " << elapsed.count()
                  << " sec | Result: " << result << "\n";

        cudaFree(d_partial);
        cudaFree(d_tmp);
    }

    return 0;
}