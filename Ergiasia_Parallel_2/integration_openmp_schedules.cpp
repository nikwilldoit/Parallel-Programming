#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <omp.h>

#define N 100000000

double f(double x) {
    if ((int)x % 2 == 0)
        return sin(x) * sin(x) * sqrt(x + 1);
    else
        return x * x;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <static|dynamic|guided> <chunk>\n";
        return 1;
    }

    std::string sched_type = argv[1];
    int chunk = std::stoi(argv[2]);

    double a = 0.0;
    double b = 10.0;
    double W = (b - a) / N;

    double integral = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    if (sched_type == "static") {
        #pragma omp parallel for firstprivate(a, W) reduction(+:integral) schedule(static, chunk)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * W;
            double x2 = a + (i + 1) * W;
            integral += (f(x1) + f(x2)) * W / 2.0;
        }

    } else if (sched_type == "dynamic") {
        #pragma omp parallel for reduction(+:integral) schedule(dynamic, chunk)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * W;
            double x2 = a + (i + 1) * W;
            integral += (f(x1) + f(x2)) * W / 2.0;
        }

    } else if (sched_type == "guided") {
        #pragma omp parallel for reduction(+:integral) schedule(guided, chunk)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * W;
            double x2 = a + (i + 1) * W;
            integral += (f(x1) + f(x2)) * W / 2.0;
        }

    } else {
        std::cerr << "Unknown schedule type\n";
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Schedule: " << sched_type << ", chunk = " << chunk << ", integral = " << integral << ", time = " << elapsed.count() << " s\n";

    return 0;
}