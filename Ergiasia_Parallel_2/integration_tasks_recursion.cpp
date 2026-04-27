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

//recursive task based integration
void integrate_task(double a, double b, int depth, double &result) {

    //maximum recursion depth
    //prevents excessive task creation overhead
    const int max_depth = 4;

    //number of subintervals for sequential computation
    const int base_n = 10000;


    //when maximum depth is reached perform sequential computation
    if (depth >= max_depth) {

        double local = trapezoid_segment(a, b, base_n);

        //result is shared among all threadsand requires synchronization
        #pragma omp atomic
        result += local;

        return;
    }

    double mid = 0.5 * (a + b);

    //create task for left half [a,mid]
    #pragma omp task firstprivate(a, mid, depth) shared(result)
    {
        integrate_task(a, mid, depth + 1, result);
    }

    //create task for right half [mid, b]
    #pragma omp task firstprivate(b, mid, depth) shared(result)
    {
        integrate_task(mid, b, depth + 1, result);
    }

    //waits for child tasks created at this level to finish before returning
    #pragma omp taskwait
}

int main(int argc, char* argv[]) {

    double a = 0.0;
    double b = 10.0;

    //default number of threads
    int num_threads = 4;

    //read number of threads from command line
    if (argc >= 2) {
        num_threads = std::stoi(argv[1]);
    }

    double result = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    
    #pragma omp parallel num_threads(num_threads)
    {
        //only one thread creates the initial tasks
        #pragma omp single
        {
            integrate_task(a, b, 0, result);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "threads = " << num_threads
              << ", integral(approx) = " << result
              << ", time = " << elapsed.count() << " s\n";

    return 0;
}