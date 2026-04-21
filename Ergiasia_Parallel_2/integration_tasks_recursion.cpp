#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

double f(double x) {
    int iters = (int)(x * 1000) % 1000;
    double sum = 0;
    for (int i = 0; i < iters; i++) {
        sum += sin(x) * cos(x);
    }
    return sum;
}

double integrate_seq(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0;

    for (int i = 0; i < n; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }

    return sum;
}

double integrate_task(double a, double b, int depth, int max_depth) {

    // cutoff για αποφυγή explosion tasks
    if (depth >= max_depth || (b - a) < 0.001) {
        return integrate_seq(a, b, 1000);
    }

    double mid = (a + b) / 2.0;
    double left = 0.0, right = 0.0;

    #pragma omp task shared(left)
    left = integrate_task(a, mid, depth + 1, max_depth);

    #pragma omp task shared(right)
    right = integrate_task(mid, b, depth + 1, max_depth);

    #pragma omp taskwait

    return left + right;
}

int main() {

    double a = 0.0;
    double b = 10.0;

    int max_depth = 4;

    double result = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = integrate_task(a, b, 0, max_depth);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Integral (recursive tasks) = " << result
              << ", time = "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}