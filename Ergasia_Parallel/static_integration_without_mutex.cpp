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
    double* local_sums;   //pinakas pou grafei gia kathe thread
};

double f(double x) {
    return x * x;
}

void* thrfunc(void* arg) {
    manyparams* data=(manyparams*)arg;

    int threadid = data->threadid;
    int thread_count = data->thread_count;
    double a = data->a;
    double h = data->h;
    long N = data->N;
    double* local_sums = data->local_sums;

    long base  = N / thread_count;

    long start = threadid * base;
    
    long end;
    if(threadid == thread_count - 1) {
        end=N;
    } else {
        end = start + base;
    }
    double local_sum = 0.0;

    for (long i = start; i < end; ++i) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    //kathe thread grafei mono sthn dikh tou thesh ston pinaka local_sums
    //kai etsi den xreiazetai mutex
    local_sums[threadid] = local_sum;

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

    pthread_t* threads = new pthread_t[thread_count];
    manyparams* tdata = new manyparams[thread_count];
    double* local_sums = new double[thread_count];

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < thread_count; i++) {
        tdata[i].threadid = i;
        tdata[i].thread_count = thread_count;
        tdata[i].a = a;
        tdata[i].h = h;
        tdata[i].N = N;
        tdata[i].local_sums = local_sums;

        pthread_create(&threads[i], NULL, thrfunc, &tdata[i]);
    }

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    //sum apo ta apotelesmata twn threads
    double global_sum = 0.0;
    for (int i = 0; i < thread_count; i++) {
        global_sum += local_sums[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Integral from " << a << " to " << b << " = " << global_sum << '\n';
    cout << "Execution time: " << elapsed.count() << " seconds with " << thread_count << " threads\n";

    delete[] threads;
    delete[] tdata;
    delete[] local_sums;

    return 0;
}