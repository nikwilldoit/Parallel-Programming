#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

using namespace std;

struct manyparams {
    double a;
    double h;
    long   N;
};

int taskid = 0; //epomenh ergasia
int NTASK = 0;  //plithos ergasiwn
int K = 1000;   //megethos task

pthread_mutex_t tlock = PTHREAD_MUTEX_INITIALIZER;   //gia taskid


double global_sum = 0.0;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

double f(double x) {
    double x2 = x * x * x + sqrt(x + 1.0);
    for (int i = 0; i < 15; ++i) {
        x2 += 0.000001 * x * x;
    }
    return x2;
}

void* thrfunc(void* arg) {
    manyparams* data = (manyparams*) arg;

    double a = data->a;
    double h = data->h;
    long   N = data->N;

    while (true) {
        int t;

        //pare epomeno task apo oura
        pthread_mutex_lock(&tlock);
        t = taskid++;
        pthread_mutex_unlock(&tlock);

        //denuparxoun alla task
        if (t >= NTASK) {
            break;
        }

        long start = (long)t * (long)K;
        long end   = start + K;
        if (end > N) end = N;

        double local_sum = 0.0;

        for (long i = start; i < end; ++i) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            local_sum += (f(x1) + f(x2)) * h / 2.0;
        }

        pthread_mutex_lock(&sum_mutex);
        global_sum += local_sum;
        pthread_mutex_unlock(&sum_mutex);
    }

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        //argv[1] = N argv[2] = thread_count argv[3] = K (megethos task)
        cerr << "Usage: " << argv[0] << " N thread_count K\n";
        return 1;
    }

    long N = stol(argv[1]);
    int thread_cnt = stoi(argv[2]);
    K = stoi(argv[3]);

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    //upologismos plithous task
    NTASK  = (int)((N + K - 1) / K);
    taskid = 0;
    global_sum = 0.0;

    pthread_t*   threads = new pthread_t[thread_cnt];
    manyparams   params;
    params.a = a;
    params.h = h;
    params.N = N;

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < thread_cnt; ++i) {
        pthread_create(&threads[i], NULL, thrfunc, &params);
    }

    for (int i = 0; i < thread_cnt; ++i) {
        pthread_join(threads[i], NULL);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Integral from " << a << " to " << b << " = " << global_sum << '\n';
    cout << "Execution time: " << elapsed.count()
         << " seconds with " << thread_cnt
         << " threads (dynamic queue, K = " << K << ")\n";

    delete[] threads;

    return 0;
}