#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define N 100000000

double f(double x) {
    // Ακανόνιστο workload
    int iters = (int)(x * 1000) % 1000;
    double sum = 0;
    for (int i = 0; i < iters; i++) {
        sum += sin(x) * cos(x);
    }
    return sum;
}

int main(int argc, char* argv[]) {

    int num_tasks = 8; // ή από argv

    double a = 0.0;
    double b = 10.0;
    double W = (b - a) / N;

    double integral = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            int chunk = N / num_tasks;

            for (int t = 0; t < num_tasks; t++) {

                int start_i = t * chunk;
                int end_i = (t == num_tasks - 1) ? N : (t + 1) * chunk;

                #pragma omp task firstprivate(start_i, end_i, a, W) shared(integral)
                {
                    double local_sum = 0.0;

                    for (int i = start_i; i < end_i; i++) {
                        double x1 = a + i * W;
                        double x2 = a + (i + 1) * W;
                        local_sum += (f(x1) + f(x2)) * W / 2.0;
                    }

                    #pragma omp atomic
                    integral += local_sum;
                }
            }

            #pragma omp taskwait
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Integral (tasks no recursion) = " << integral
              << ", time = "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    return 0;
}