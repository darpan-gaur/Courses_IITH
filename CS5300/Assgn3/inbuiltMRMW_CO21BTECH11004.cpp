#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <atomic>

using namespace std;

double lambda;
int capacity=1, numOps;

long *threadTime;

// random number generator
random_device rd;
mt19937 gen(rd());

// open output file
FILE *outFile;

// make atomic shared variable
atomic<int> shVar;

// start time in nanoseconds
auto startTime = chrono::high_resolution_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::nanoseconds>(startTime).count();

void testAtomic(int threadID) {
    int var;    // local variable
    int id = threadID;
    
    // random number generator
    uniform_int_distribution<> dis(0, 1);
    exponential_distribution<> exp(lambda);

    double p = 0.5;
    for (int i=0;i<numOps;i++) {
        string action;
        if (dis(gen)>0.5) action = "read";
        else action = "write";

        auto reqTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto RT = chrono::duration_cast<chrono::nanoseconds>(reqTime).count();
        fprintf(outFile, "%dth action requested at %ld by thread %d\n", i, RT-S, id);

        if (action=="read") {
            var = shVar.load();
            fprintf(outFile, "Thread %d read %d\n", id, var);
        }
        else {
            shVar.store(id*numOps);
            fprintf(outFile, "Thread %d wrote %d\n", id, id*numOps);
        }

        auto complTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto CT = chrono::duration_cast<chrono::nanoseconds>(complTime).count();

        threadTime[id] += CT-RT;
        fprintf(outFile, "%dth action completed at %ld by thread %d\n", i, CT-S, id);


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
    outFile = fopen("logFileInbuiltMRMW.txt", "w");

    // allocate memory to threadTime
    threadTime = new long[capacity];
    for (int i=0;i<capacity;i++) {
        threadTime[i] = 0;
    }

    // initialize register
    shVar.store(0);

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