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
    double h = (b - a) / N;

    double integral = 0.0; //final result

    int num_threads = 4; //default number of threads

    //read number of threads from command line
    if (argc >= 2) {
        num_threads = std::stoi(argv[1]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    //parallel with specified number of threads
    #pragma omp parallel num_threads(num_threads) firstprivate(a, h)
    {
        double mysum = 0.0; //private partial sum for each thread

        //distribute loop iterations among threads
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            mysum += (f(x1) + f(x2)) * h / 2.0;
        }

        //combine partial sums into the global result
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