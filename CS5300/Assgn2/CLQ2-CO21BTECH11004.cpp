#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <random>
#include <unistd.h>

using namespace std;

// CLQ implementaion with front and rear methods
template <typename T>
class LockBasedQueue {
    int head, tail, sz;
    T* items;
    mutex lock;

public:
    // LockBasedQueue(int capacity) {
    //     head = 0; tail = 0;
    //     items = new T[capacity];
    // }
    void setQueue(int capacity) {
        head = 0; tail = 0;
        items = new T[capacity];
        sz = capacity;
    }

    void enq(T x) {
        lock.lock();
        if (tail - head == sz) {
            cout << "Queue is full" << endl;
            lock.unlock();
            return;
        }
        items[tail % sz] = x;
        tail++;
        // cout << "Enqueued " << x << endl;
        lock.unlock();
    }

    T deq() {
        lock.lock();
        if (tail == head) {
            cout << "Queue is empty" << endl;
            lock.unlock();
            return -1; 
        }
        T x = items[head % sz];
        head++;
        // cout << "Dequeued " << x << endl;
        lock.unlock();
        return x;
    }

    T front() {
        lock.lock();
        if (tail == head) {
            cout << "Queue is empty" << endl;
            lock.unlock();
            return -1; 
        }
        T x = items[head % sz];
        lock.unlock();
        return x;
    }
    T rear() {
        lock.lock();
        if (tail == head) {
            cout << "Queue is empty" << endl;
            lock.unlock();
            return -1; 
        }
        T x = items[(tail-1) % sz];
        lock.unlock();
        return x;
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
LockBasedQueue<int> q;

// global arrays for measuring time
int *threadEnqTime, *threadDeqTime, *thrTime;

// test thread function as given in question
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
    
    // allocate memory for arrays
    threadEnqTime = new int[n];
    threadDeqTime = new int[n];
    thrTime = new int[n];
    for (int i=0;i<n;i++) {
        threadEnqTime[i] = 0;
        threadDeqTime[i] = 0;
        thrTime[i] = 0;
    }

    // set queue size
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

    // print throughput
    cout << "Throughput: " << throughput << endl;

    // write to output file
    FILE *outFile = fopen("CLQ2-out.txt","w");
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