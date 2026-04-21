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
    double W = (b - a) / N;   // όπως τα slides χρησιμοποιούν W

    double integral = 0.0;    // αντίστοιχο του pi

    int num_threads = 4;      // κάν’ το παραμετρικό για τα πειράματα

    auto start = std::chrono::high_resolution_clock::now();

    // parallel for με reduction, στο στυλ των slides
    #pragma omp parallel for firstprivate(W, a) reduction(+:integral) num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        double x1 = a + i * W;
        double x2 = a + (i + 1) * W;
        integral += (f(x1) + f(x2)) * W / 2.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral from " << a << " to " << b
              << " = " << integral << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds"
              << std::endl;

    return 0;
}