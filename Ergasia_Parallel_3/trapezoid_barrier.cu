#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000   // βάλε εδώ ό,τι N θες

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid, block-level reduction με __syncthreads()
__global__ void trapezoid_barrier(double a, double h, int n, double *partial) {

    __shared__ double cache[256];  // blockDim.x = 256

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    // κάθε thread υπολογίζει 1 trapezoid
    if (i < n) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5;
    }

    // γράφουμε το αποτέλεσμα στο shared memory
    cache[tid] = value;

    // BARRIER: περιμένουμε όλα τα threads του block
    __syncthreads();

    // parallel reduction μέσα στο block
    int step = blockDim.x / 2;
    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        __syncthreads();
        step /= 2;
    }

    // thread 0 γράφει το άθροισμα του block στη global μνήμη
    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

// 2ος kernel: reduction πάνω στα partial sums (πάλι με shared + syncthreads)
__global__ void final_reduce(double *input, double *output, int n) {

    __shared__ double cache[256];  // blockDim.x = 256

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

    double *d_partial;   // partial sums από 1ο kernel
    double *d_reduce;    // προσωρινά για 2ο kernel
    double result = 0.0;

    cudaMalloc(&d_partial, blocks * sizeof(double));
    cudaMalloc(&d_reduce,  blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    // 1ος kernel: υπολογισμός τραπεζίων + block-level reduction (με barriers)
    trapezoid_barrier<<<blocks, threads>>>(a, h, n, d_partial);
    cudaDeviceSynchronize();

    // 2ος kernel: reduction των block partial sums μέχρι να μείνει 1 τιμή
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
    std::chrono::duration<double> elapsed = end - start;

    // μόνο 1 double από τη GPU
    cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    cudaFree(d_reduce);

    return 0;
}