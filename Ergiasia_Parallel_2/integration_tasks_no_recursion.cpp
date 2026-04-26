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

int main(int argc, char* argv[]) {
    double a = 0.0;
    double b = 10.0;

    int num_tasks = 8; //default number of tasks (chunks)
    int num_threads = 4; //default number of threads

    if (argc >= 2) {
        num_tasks = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        num_threads = std::stoi(argv[2]);
    }

    double h = (b - a) / N;
    double integral = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        //only one thread will create the tasks
        #pragma omp single
        {
            //create num_tasks tasks each responsible for a chunk of [0,N]
            for (int t = 0; t < num_tasks; t++) {
                //each task gets its own copy of t and h and shares integral and a
                #pragma omp task firstprivate(t, h) shared(integral, a)
                {
                    //gets the index range [i_start,i_end] for this task
                    int i_start = t * (N / num_tasks);
                    int i_end;
                    if(t == (num_tasks - 1)){
                        //last task takes the rest up to N
                        i_end=N
                    }
                    else{
                        i_end=(t + 1) * (N / num_tasks)
                    }

                    //local partial sum for this task
                    double local_sum = 0.0;
                    for (int i = i_start; i < i_end; i++) {
                        double x1 = a + i * h;
                        double x2 = a + (i + 1) * h;
                        local_sum += (f(x1) + f(x2)) * h / 2.0;
                    }

                    //add tasks local_sum to the global
                    #pragma omp atomic
                    integral += local_sum;
                }
            }
            //waits until all created tasks are finished
            #pragma omp taskwait
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[TASKS-CHUNKS] tasks = " << num_tasks << ", threads = " << num_threads << ", integral = " << integral << ", time = " << elapsed.count() << " s\n";

    return 0;
}