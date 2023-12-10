#include <iostream>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <string>
#include <unistd.h>

using namespace std;

class QNode {
public:
    bool locked = false;
    QNode* next = NULL;
};

class MCSLock {
private:
    atomic<QNode*> tail;
    vector<QNode*> myNode;
public:
    MCSLock(int n) {
        tail.store(NULL);
        myNode = vector<QNode*>(n);
        for(int i = 0; i < n; i++) {
            myNode[i] = new QNode();
        }
    }
    void lock(int threadID) {
        QNode* qnode = myNode[threadID];
        QNode* pred = tail.exchange(qnode);
        if(pred != NULL) {
            qnode->locked = true;
            pred->next = qnode;
            while(qnode->locked) {}
        }
    }
    void unlock(int threadID) {
        QNode* qnode = myNode[threadID];
        if (qnode->next == NULL) {
            if (tail.compare_exchange_strong(qnode, NULL)) {
                return;
            }
            while(qnode->next == NULL) {}
        }
        qnode->next->locked = false;
        qnode->next = NULL;
    }
    ~MCSLock() {
        delete tail;
        for (int i=0;i<myNode.size();i++) {
            delete myNode[i];
        }
    }
};

// Global variables
int n,k;
double lambda1, lambda2;

// lock object
MCSLock* test;

// random number generator
random_device rd;
mt19937 gen(rd());

// start time in nanoseconds
auto startTime = chrono::high_resolution_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::nanoseconds>(startTime).count();

// store average entry time
vector<long> entryTime;

// open output file
FILE *outFile = fopen("MCS.txt", "w");

void testCS(int threadID) {
    // exponential distribution
    exponential_distribution<double> exp1(1000.0/lambda1);
    exponential_distribution<double> exp2(1000.0/lambda2);

    for (int i=1;i<=k;i++) {
        auto reqEntryTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto RET = chrono::duration_cast<chrono::nanoseconds>(reqEntryTime).count();
        
        fprintf(outFile, "%d th CS Entry Request at %ld by thread %d\n", i, RET-S, threadID);

        test->lock(threadID);
        auto actEntryTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto AET = chrono::duration_cast<chrono::nanoseconds>(actEntryTime).count();
        
        fprintf(outFile, "%d th CS Entry at %ld by thread %d\n", i, AET-S, threadID);
        entryTime[threadID] += AET-RET;
        sleep(exp1(gen));

        auto reqExitTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto RET2 = chrono::duration_cast<chrono::nanoseconds>(reqExitTime).count();
        
        fprintf(outFile, "%d th CS Exit Request at %ld by thread %d\n", i, RET2-S, threadID);
        test->unlock(threadID);

        auto actExitTime = chrono::high_resolution_clock::now().time_since_epoch();
        auto AET2 = chrono::duration_cast<chrono::nanoseconds>(actExitTime).count();
        
        fprintf(outFile, "%d th CS Entry Request at %ld by thread %d\n", i, RET-S, threadID);
        sleep(exp2(gen));
    }
}

int main() {
    // open input file
    FILE *inFile = fopen("inp-params.txt", "r");

    if (inFile == NULL) {
        printf("Error: Could not open input file\n");
        return 0;
    }

    // take input n, k, lambda1, lambda2
    fscanf(inFile, "%d %d %lf %lf", &n, &k, &lambda1, &lambda2);
    // cout << n << " " << k << " " << lambda1 << " " << lambda2 << endl;

    // initialize entryTime vector
    entryTime = vector<long>(n, 0);

    // create lock object
    test = new MCSLock(n);

    auto sTime = chrono::high_resolution_clock::now().time_since_epoch();
    auto ST = chrono::duration_cast<chrono::nanoseconds>(sTime).count();

    // create n threads
    vector<thread> threads;
    for (int i=0;i<n;i++) {
        threads.push_back(thread(testCS, i));
    }

    // join threads
    for (int i=0;i<n;i++) {
        threads[i].join();
    }

    auto eTime = chrono::high_resolution_clock::now().time_since_epoch();
    auto ET = chrono::duration_cast<chrono::nanoseconds>(eTime).count();

    // find and print average entry time
    double avgEntryTime = 0;
    for (int i=0;i<n;i++) {
        avgEntryTime += entryTime[i];
    }
    avgEntryTime = (double)avgEntryTime/(n*k);
    avgEntryTime = avgEntryTime/(1.0e9);
    // cout << "Average Entry Time: " << avgEntryTime << endl;
    cout.precision(11);
    cout << fixed << avgEntryTime << " ";

    // print throughput
    double throughput = (double)(n*k)/(ET-ST);
    throughput = throughput*1.0e9;
    // cout << "Throughput: " << throughput << endl;
    cout << fixed << throughput << endl;

    // close files
    fclose(inFile);
    fclose(outFile);
    return 0;
}