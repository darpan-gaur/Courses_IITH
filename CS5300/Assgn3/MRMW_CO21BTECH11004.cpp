#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace std;

template <class T>
class stampedValue {
public:
    long stamp;
    T value;
    stampedValue() {

    }
    stampedValue(T init) {
        stamp = 0;
        value = init;
    }
    stampedValue (long ts, T v) {
        stamp = ts;
        value = v;
    }
    stampedValue<T> max(stampedValue<T> x, stampedValue<T> y) {
        if (x.stamp > y.stamp) {
            return x;
        }
        else {
            return y;
        }
    }
    stampedValue<T> minValue() {
        return stampedValue<T>((T)NULL);
    }
};

template <class T>
class atomicMRMWRegister {
private:
    stampedValue<int>* a_table;
    int sz;
public:
    atomicMRMWRegister(int capacity, T init) {
        a_table = new stampedValue<int>[capacity];
        sz = capacity;
        stampedValue<int> value = stampedValue<T>(init);
        for (int i=0;i<sz;i++) {
            a_table[i] = value;
        }
    }
    void write(T value, int me){
        stampedValue<T> maxStamp = maxStamp.minValue();
        for (int i=0;i<sz;i++) {
            maxStamp = maxStamp.max(maxStamp, a_table[i]);
        }
        a_table[me] = stampedValue<T>(maxStamp.stamp+1, value);
    }
    T read() {
        stampedValue<T> maxStamp = maxStamp.minValue();
        for (int i=0;i<sz;i++) {
            maxStamp = maxStamp.max(maxStamp, a_table[i]);
        }
        return maxStamp.value;
    }
};

double lambda;
int capacity=1, numOps;

long *threadTime;

// random number generator
random_device rd;
mt19937 gen(rd());

// open output file
FILE *outFile;

atomicMRMWRegister<int> shVar(capacity, 0);

// start time in nanoseconds
auto startTime = chrono::high_resolution_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::nanoseconds>(startTime).count();

void testAtomic(int threadID) {
    int var;    // local variable
    int id = threadID;
    // random number generator
    uniform_real_distribution<double> unif(0, 1);
    exponential_distribution<double> exp(lambda);

    double p = 0.5;
    for (int i=0;i<numOps;i++) {
        string action;
        if (unif(gen)>0.5) action = "read";
        else action = "write";

        auto reqTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto RT = chrono::duration_cast<chrono::nanoseconds>(reqTime).count();

        // write to file time in nanoseconds
        fprintf(outFile, "%dth action requested at %ld by thread %d\n", i, RT-S, id);
        // cout << i << "th action requested at  " << reqTime.time_since_epoch().count() << " by thread " << id << "\n";

        if (action == "read") {
            var = shVar.read();
            fprintf(outFile, "Thread %d read %d\n", id, var);
            // cout << "Thread " << id << " read " << var << endl;
        }
        else {
            var = id*numOps;
            shVar.write(var, threadID);
            fprintf(outFile, "Thread %d wrote %d\n", id, var);
            // cout << "Thread " << id << " wrote " << var << endl;
        }

        auto complTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto CT = chrono::duration_cast<chrono::nanoseconds>(complTime).count();
        // add time in nanoseconds to threadTime
        threadTime[id] += CT-RT;
        fprintf(outFile, "%dth action completed at %ld by thread %d\n", i, CT-S, id);
        // cout << i << "th action completed at  " << complTime.time_since_epoch().count() << " by thread " << id << "\n";
        
        sleep(exp(gen));
    }
}   

int main() {
    // open input file
    FILE *inFile = fopen("inp-params.txt", "r");

    if (inFile == NULL) {
        printf("Error: Could not open input file\n");
        return 0;
    }

    // take input capacity, numOps, lambda
    fscanf(inFile, "%d %d %lf", &capacity, &numOps, &lambda);
    cout << "capacity: " << capacity << " numOps: " << numOps << " lambda: " << lambda << endl;

    // open output file
    outFile = fopen("logFileMRMW.txt", "w");

    // allocate memory to threadTime
    threadTime = new long[capacity];
    for (int i=0;i<capacity;i++) {
        threadTime[i] = 0;
    }

    // initialize register
    shVar = atomicMRMWRegister<int>(capacity, 0);

    // create threads
    thread thr[capacity];
    for (int i=0;i<capacity;i++) {
        thr[i] = thread(testAtomic, i);
    }

    // join threads
    for (int i=0;i<capacity;i++) {
        thr[i].join();
    }

    // print avg time
    double avgTime = 0;
    for (int i=0;i<capacity;i++) {
        avgTime += threadTime[i];
    }
    avgTime /= (double)capacity;
    avgTime /= (double)numOps;

    cout << "Average time per operation: " << avgTime << " nanoseconds" << endl;

    // close files
    fclose(inFile);
    fclose(outFile);

    // free memory
    delete[] threadTime;
    return 0;
}