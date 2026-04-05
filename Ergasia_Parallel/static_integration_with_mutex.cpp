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

//global shared sum
double global_sum = 0.0;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

//h f(x) pou oloklirwnw
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

    long tables= N / thread_count; //to upodiastima pou tha douleuei kathe thread

    long start = threadid * tables; //h prwth epanalipsi i pou tha doulevei to thread (0,tables,2*tables,ktlp)
    
    long end;
    //to teleutaio thread tha douleuei mexri to N
    if(threadid == thread_count - 1) {
        end=N;
    }
    else { //ta alla mexri to start+tables

        
        end = start + tables;
    }

    double local_sum = 0.0;

    for (long i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        local_sum += (f(x1) + f(x2)) * h / 2.0;
    }

    //prosthetw to local_sum sto global_sum me mutex
    pthread_mutex_lock(&sum_mutex);
    global_sum += local_sum;
    cout << "Partial integral so far = " << global_sum << '\n'; //ektupwsh ana oloklhrwsh
    pthread_mutex_unlock(&sum_mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        return 1;
    }

    long N = stol(argv[1]); //plithos trapeziwn

    int thread_count =stoi(argv[2]); //plithos thread

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    //pinakas me threads gia join sthn sunexeia
    pthread_t* threads= new pthread_t[thread_count];
    //pinakas apo manyparams gia ta dedomena se kathe thread
    manyparams* thread_data = new manyparams[thread_count];

    global_sum = 0.0;

    auto start_time = chrono::high_resolution_clock::now();

    //dhmiourgia kai gemisma thread
    for (int t = 0; t < thread_count; t++) {
        thread_data[t].threadid = t;
        thread_data[t].thread_count = thread_count;
        thread_data[t].a = a;
        thread_data[t].h = h;
        thread_data[t].N = N;

        pthread_create(&threads[t], NULL, threadfunc, &thread_data[t]);
    }

    //anamonh gia teleiwma twn thread
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