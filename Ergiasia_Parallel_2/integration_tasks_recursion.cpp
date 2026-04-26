#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define N 100000000

double f(double x) {
    //cast x from double to int
    if (static_cast<int>(x) % 2 == 0)
        return std::sin(x) * std::sin(x) * std::sqrt(x + 1.0); //heavier path
    else
        return x * x; //lighter path
}

//gets trapezoidal integral of f(x) over [a,b] using n subintervals
double trapezoid_segment(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }
    return sum;
}

void integrate_task(double a, double b, int depth, double &result) {
    const int max_depth = 4;
    const int base_n = 10000;

    if (depth >= max_depth) {
        double local = trapezoid_segment(a, b, base_n);
        #pragma omp atomic
        result += local;
        return;
    }

    double mid = 0.5 * (a + b);

    #pragma omp task firstprivate(a, mid, depth) shared(result)
    {
        integrate_task(a, mid, depth + 1, result);
    }

    #pragma omp task firstprivate(b, mid, depth) shared(result)
    {
        integrate_task(mid, b, depth + 1, result);
    }
}

int main(int argc, char* argv[]) {
    double a = 0.0;
    double b = 10.0;

    int num_threads = 4; //default number of threads
    if (argc >= 2) {
        num_threads = std::stoi(argv[1]);
    }

    double result = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            integrate_task(a, b, 0, result);
            #pragma omp taskwait
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[TASKS-RECURSIVE] threads = " << num_threads << ", integral(approx) = " << result << ", time = " << elapsed.count() << " s\n";

    return 0;
}