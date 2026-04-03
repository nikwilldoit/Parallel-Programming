#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

using namespace std;

struct manyparams {
    int threadid;
    int thread_count;
    double a;
    double h;
    long N;
};

double global_sum = 0.0;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

double f(double x) {
    if (x <= 5.0) {
        return x * x;
    } else {
        double x2 = x * x;
        for (int i = 0; i < 50; i++) {
            x2 += 0.000001 * x;
        }
        return x2;
    }
}

void* threadfunc(void* arg) {
    manyparams* data=(manyparams*)arg;

    int threadid = data->threadid;
    int thread_count =data->thread_count;
    double a = data->a;
    double h = data->h;
    long N = data->N;

    long tables= N / thread_count;

    long start = threadid * tables;
    

    double local_sum = 0.0;

    for (long i = threadid; i < N; i+= thread_count) {
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
    int thread_count = stoi(argv[2]);

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    pthread_t* threads= new pthread_t[thread_count];
    manyparams* thread_data = new manyparams[thread_count];

    global_sum = 0.0;

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < thread_count; i++) {
        thread_data[i].threadid = i;
        thread_data[i].thread_count = thread_count;
        thread_data[i].a = a;
        thread_data[i].h = h;
        thread_data[i].N = N;

        pthread_create(&threads[i], NULL, threadfunc, &thread_data[i]);
    }

    for (int t = 0; t < thread_count; t++) {
        pthread_join(threads[t], NULL);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Integral from " << a << " to " << b << " = " << global_sum << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    delete[] threads;
    delete[] thread_data;

    return 0;
}