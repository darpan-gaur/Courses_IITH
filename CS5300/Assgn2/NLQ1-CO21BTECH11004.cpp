#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <unistd.h>

using namespace std;

// NLQ implementaion from book
template <typename T>
class HWQueue {
    atomic<T>* items;
    atomic<int> tail;
    int CAPACITY;

public:
    // HWQueue() {
    //     items = new atomic<T>[CAPACITY];
    //     for (int i = 0; i < CAPACITY; i++) {
    //         items[i] = NULL;
    //     }
    //     tail = 0;
    // }

    void setQueue(int capacity) {
        CAPACITY = capacity;
        items = new atomic<T>[CAPACITY];
        for (int i = 0; i < CAPACITY; i++) {
            items[i] = -1;
        }
        tail = 0;
    }

    void enq(T x) {
        int i = tail.fetch_add(1);
        items[i] = x;
        // cout << "Enqueued " << x << endl;
    }

    T deq() {
        while (true) {
            int range = tail.load();
            for (int i = 0; i < range; i++) {
                T value = items[i].exchange(-1);
                if (value != -1) {
                    // cout << "Dequeued " << value << endl;
                    return value;
                }
            }
        }
    }
};

// chrono start systemClock in microseconds
auto start = chrono::system_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::microseconds>(start).count();

// random number generator
random_device rd;
mt19937 gen(rd());

// global variables
int numOps;
double rndLt, lambda;

// global queue
HWQueue<int> q;

// global arrays for measuring time
int *threadEnqTime, *threadDeqTime, *thrTime;

// thread function as given in question
void testThread(int thrId){
    double unifVal, sleepTime;
    long startTime, endTime;

    // random number generator
    uniform_real_distribution<double> unif(0, 1);
    exponential_distribution<double> exp(10.0/lambda);

    for (int i=0;i<numOps;i++) {
        double p = unif(gen);
        if (p < rndLt) {
            int v = rand() % 10000;
            startTime = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() - S;
            q.enq(v);
            endTime = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() - S;
            threadEnqTime[thrId] += endTime - startTime;
        }
        else {
            startTime = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() - S;
            q.deq();
            endTime = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() - S;
            threadDeqTime[thrId] += endTime - startTime;
        }
        thrTime[thrId] += endTime - startTime;
        sleepTime = exp(gen);

        sleep(sleepTime);
    }
}

int main() {
    // Open input file
    FILE *inFile = fopen("inp-params.txt","r");

    if (inFile == NULL) {
        printf("Error! Could not open input file\n");
    }

    // take input n, numOps, rndLt, lambda
    int n;
    fscanf(inFile, "%d %d %lf %lf", &n, &numOps, &rndLt, &lambda);
    cout << "n: " << n << " numOps: " << numOps << " rndLt: " << rndLt << " lambda: " << lambda << endl;
    
    // aloocate memory for arrays
    threadEnqTime = new int[n];
    threadDeqTime = new int[n];
    thrTime = new int[n];
    for (int i=0;i<n;i++) {
        threadEnqTime[i] = 0;
        threadDeqTime[i] = 0;
        thrTime[i] = 0;
    }

    // set queue capacity
    q.setQueue(n*numOps);
    
    // create threads
    thread thr[n];
    for (int i=0;i<n;i++) {
        thr[i] = thread(testThread, i);
    }

    // join threads
    for (int i=0;i<n;i++) {
        thr[i].join();
    }

    // find throughput
    int totalThrTime = 0;
    for (int i=0;i<n;i++) {
        totalThrTime += thrTime[i];
    }
    int throughput = (double)(numOps*n*1000000) / (double)totalThrTime;

    cout << "Throughput: " << throughput << endl;

    // write to output file
    FILE *outFile = fopen("NLQ1-out.txt","w");
    fprintf(outFile, "%d\n", throughput);
    fclose(outFile);

    // free memory
    delete[] threadEnqTime;
    delete[] threadDeqTime;
    delete[] thrTime;

    // close input file
    fclose(inFile);

    return 0;
}