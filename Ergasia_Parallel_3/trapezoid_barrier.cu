#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

// 1 thread = 1 trapezoid (ξεκάθαρη κατανομή εργασίας)
__global__ void trapezoid_barrier(double a, double h, double *partial) {

    __shared__ double cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    // ΚΑΘΕ THREAD κάνει 1 υπολογισμό (1 trapezoid)
    if (i < N) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5;
    }

    // γράφουμε στο shared memory
    cache[tid] = value;

    //BARRIER 1: περιμένουμε όλα τα threads του block
    __syncthreads();

    // reduction μέσα στο block
    int step = blockDim.x / 2;

    while (step > 0) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }

        __syncthreads();

        step /= 2;
    }

    // thread 0 γράφει το αποτέλεσμα του block
    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    double *d_partial, *h_partial;

    h_partial = (double*)malloc(blocks * sizeof(double));
    cudaMalloc(&d_partial, blocks * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    trapezoid_barrier<<<blocks, threads>>>(a, h, d_partial);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_partial,
               blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        sum += h_partial[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: " << elapsed.count() << " sec\n";

    cudaFree(d_partial);
    free(h_partial);

    return 0;
}