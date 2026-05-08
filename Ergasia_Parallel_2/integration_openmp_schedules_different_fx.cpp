#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <omp.h>

#define N 100000000

//non uniform function to simulate load imbalance
double f(double x) {
    if (static_cast<int>(x) % 2 == 0)
        return std::sin(x) * std::sin(x) * std::sqrt(x + 1.0);
    else
        return x * x;
}

int main(int argc, char* argv[]) {

    //checks if required arguments are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <static|dynamic|guided> <chunk> [num_threads]\n";
        return 1;
    }

    //reads scheduling type (static, dynamic, guided)
    std::string sched_type = argv[1];

    //read chunk size (number of iterations per chunk)
    int chunk = std::stoi(argv[2]);

    //default number of threads
    int num_threads = 4;
    //optional argument of number of threads
    if (argc >= 4) {
        num_threads = std::stoi(argv[3]);
    }

    double a = 0.0;
    double b = 10.0;
    double h = (b - a) / N;

    double integral = 0.0; //variable to store final result

    auto start = std::chrono::high_resolution_clock::now();

    if (sched_type == "static") {
        //static scheduling
        #pragma omp parallel for firstprivate(a, h) reduction(+:integral) schedule(static, chunk) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            integral += (f(x1) + f(x2)) * h / 2.0;
        }

    } else if (sched_type == "dynamic") {
        //dynamic scheduling
        #pragma omp parallel for firstprivate(a, h) reduction(+:integral) schedule(dynamic, chunk) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            integral += (f(x1) + f(x2)) * h / 2.0;
        }

    } else if (sched_type == "guided") {
        //guided scheduling
        #pragma omp parallel for firstprivate(a, h) reduction(+:integral) schedule(guided, chunk) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            integral += (f(x1) + f(x2)) * h / 2.0;
        }

    } else {
        //handle invalid scheduling type
        std::cerr << "Unknown schedule type\n";
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Schedule: " << sched_type
              << ", chunk = " << chunk
              << ", threads = " << num_threads
              << ", integral = " << integral
              << ", time = " << elapsed.count() << " s\n";

    return 0;
}