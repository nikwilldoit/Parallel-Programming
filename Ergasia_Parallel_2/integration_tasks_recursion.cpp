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

//recursive task based integration
void integrate_task(int i_start, int i_end, double a, double h, double &result) {

    //threshold determines when to stop recursion and compute directly
    const int threshold = 1000000;

    //base case to compute directly when problem size is small enough
    if ((i_end - i_start) <= threshold) {

        double local = 0.0; //local accumulator for this task

        for (int i = i_start; i < i_end; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            local += (f(x1) + f(x2)) * h / 2.0;
        }

        //safely update shared result
        #pragma omp atomic
        result += local;

        return;
    }

    //split problem into two subproblems
    int mid = (i_start + i_end) / 2;

    //creates first task for left half of the interval
    #pragma omp task firstprivate(i_start, mid) shared(result)
    {
        integrate_task(i_start, mid, a, h, result);
    }

    //creates second task for right half of the interval
    #pragma omp task firstprivate(mid, i_end) shared(result)
    {
        integrate_task(mid, i_end, a, h, result);
    }

    //waits for both subtasks to complete before continuing
    #pragma omp taskwait
}

int main(int argc, char* argv[]) {

    double a = 0.0;
    double b = 10.0;

    //default number of threads
    int num_threads = 4;

    //optional command line argument for number of threads
    if (argc >= 2)
        num_threads = std::stoi(argv[1]);

    //final result accumulator
    double result = 0.0;
    double h = (b - a) / N;

    auto start = std::chrono::high_resolution_clock::now();

    //parallel
    #pragma omp parallel num_threads(num_threads)
    {
        //only one thread starts the recursive task generation
        #pragma omp single
        {
            integrate_task(0, N, a, h, result);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "threads = " << num_threads
              << ", integral = " << result
              << ", time = " << elapsed.count() << " s\n";

    return 0;
}