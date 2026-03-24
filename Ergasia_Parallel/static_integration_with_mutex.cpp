#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

using namespace std;

struct Thread {
    int threadid;
    int thread_count;
    double a;
    double h;
    long N;
};

// global shared sum
double global_sum = 0.0;
pthread_mutex_t sum_mutex;

double f(double x) {
    return x * x;
}

void* integrate_range(void* arg) {
    Thread* data=(Thread*)arg;

    int threadid = data->threadid;
    int thread_count = data->thread_count;
    double a = data->a;
    double h = data->h;
    long N = data->N;

    long base  = N / thread_count;

    long start = threadid * base;
    
    long end;
    if(threadid == thread_count - 1) {
        end=N;
    } else {
        end = start + base;
    }

    double local_sum = 0.0;

    for (long i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    pthread_mutex_lock(&sum_mutex);
    global_sum += local_sum;
    pthread_mutex_unlock(&sum_mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        return 404;
    }

    long N = stol(argv[1]);
    int thread_count =stoi(argv[2]);

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    pthread_t* threads   = new pthread_t[thread_count];
    Thread* tdata    = new Thread[thread_count];

    pthread_mutex_init(&sum_mutex, nullptr);
    global_sum = 0.0;

    auto start_time = chrono::high_resolution_clock::now();

    for (int t = 0; t < thread_count; t++) {
        tdata[t].threadid = t;
        tdata[t].thread_count = thread_count;
        tdata[t].a = a;
        tdata[t].h = h;
        tdata[t].N = N;

        pthread_create(&threads[t], NULL, integrate_range, &tdata[t]);
    }

    for (int t = 0; t < thread_count; ++t) {
        pthread_join(threads[t], NULL);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Integral from " << a << " to " << b << " = " << global_sum << '\n';
    cout << "Execution time: " << elapsed.count() << " seconds with " << thread_count << " threads\n";

    pthread_mutex_destroy(&sum_mutex);
    delete[] threads;
    delete[] tdata;

    return 0;
}