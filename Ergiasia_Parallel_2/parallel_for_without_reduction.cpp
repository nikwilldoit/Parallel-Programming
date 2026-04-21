#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define N 100000000

double f(double x) {
    return x * x;
}

int main(int argc, char* argv[]) {
    double a = 0.0;
    double b = 10.0;
    double W = (b - a) / N;   // αντίστοιχο του W στα slides

    double integral = 0.0;

    int num_threads = 4;      // ή πάρε το από argv / OMP_NUM_THREADS

    auto start = std::chrono::high_resolution_clock::now();

    // parallel περιοχή όπως στη θεωρία, με mysum + critical
    #pragma omp parallel num_threads(num_threads) firstprivate(a, W)
    {
        double mysum = 0.0;

        #pragma omp for
        for (int i = 0; i < N; i++) {
            double x1 = a + i * W;
            double x2 = a + (i + 1) * W;
            mysum += (f(x1) + f(x2)) * W / 2.0;
        }

        #pragma omp critical
        {
            integral += mysum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral from " << a << " to " << b
              << " = " << integral << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds"
              << std::endl;

    return 0;
}