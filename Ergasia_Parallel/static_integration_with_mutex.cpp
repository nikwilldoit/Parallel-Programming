#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

using namespace std;

//struct holding all parameters each thread needs
struct manyparams {
    int threadid;
    int thread_count;
    double a; //start of integration interval
    double h;
    long N;
};

//global shared sum
double global_sum = 0.0;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

//f(x) to be integrated
double f(double x) {
    return x * x;
}

void* threadfunc(void* arg) {
    manyparams* data=(manyparams*)arg;

    int threadid = data->threadid;
    int thread_count =data->thread_count;
    double a = data->a;
    double h = data->h;
    long N = data->N;

    long tables= N / thread_count; //the period assigned to each thread

    long start = threadid * tables; //the first repitition i that the thread is going to work (0,tables,2*tables,ktlp)
    
    long end;
    //the last thread takes everything until N
    if(threadid == thread_count - 1) {
        end=N;
    }
    else { //all other threads start+tables

        
        end = start + tables;
    }

    double local_sum = 0.0;

    for (long i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    //add this threads local_sum to the global_sum
    pthread_mutex_lock(&sum_mutex);
    global_sum += local_sum;
    cout << "Partial integral so far = " << global_sum << '\n'; //progressive printing of the integral
    pthread_mutex_unlock(&sum_mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        return 1;
    }

    long N = stol(argv[1]); //number of trapezoids

    int thread_count =stoi(argv[2]); //number of threads

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    //array of threads
    pthread_t* threads= new pthread_t[thread_count];
    //array of parameter structs one per thread
    manyparams* thread_data = new manyparams[thread_count];

    global_sum = 0.0;

    auto start_time = chrono::high_resolution_clock::now();

    //create and launch all threads
    for (int t = 0; t < thread_count; t++) {
        thread_data[t].threadid = t;
        thread_data[t].thread_count = thread_count;
        thread_data[t].a = a;
        thread_data[t].h = h;
        thread_data[t].N = N;

        pthread_create(&threads[t], NULL, threadfunc, &thread_data[t]);
    }

    //wait for all threads to finish
    for (int t = 0; t < thread_count; ++t) {
        pthread_join(threads[t], NULL);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Integral from " << a << " to " << b << " = " << global_sum << '\n';
    cout << "Execution time: " << elapsed.count() << " seconds with " << thread_count << " threads\n";

    delete[] threads;
    delete[] thread_data;

    return 0;
}