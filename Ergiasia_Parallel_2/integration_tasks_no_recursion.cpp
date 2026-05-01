#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define N 100000000

double f(double x) {
    if (static_cast<int>(x) % 2 == 0)
        return std::sin(x) * std::sin(x) * std::sqrt(x + 1.0);
    else
        return x * x;
}

int main(int argc, char* argv[]) {

    double a = 0.0;
    double b = 10.0;

    int num_tasks = 8;
    int num_threads = 4;

    if (argc >= 2) num_tasks = std::stoi(argv[1]);
    if (argc >= 3) num_threads = std::stoi(argv[2]);

    if (num_tasks <= 0 || num_threads <= 0) {
        std::cerr << "Invalid input\n";
        return 1;
    }

    double h = (b - a) / N;
    double integral = 0.0;

    int chunk = N / num_tasks;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            for (int t = 0; t < num_tasks; t++) {

                int i_start = t * chunk;
                int i_end = (t == num_tasks - 1) ? N : (t + 1) * chunk;

                #pragma omp task firstprivate(i_start, i_end) shared(integral)
                {
                    double local_sum = 0.0;

                    for (int i = i_start; i < i_end; i++) {
                        double x1 = a + i * h;
                        double x2 = a + (i + 1) * h;
                        local_sum += (f(x1) + f(x2)) * h / 2.0;
                    }

                    #pragma omp atomic
                    integral += local_sum;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "tasks = " << num_tasks
              << ", threads = " << num_threads
              << ", integral = " << integral
              << ", time = " << elapsed.count() << " s\n";

    return 0;
}