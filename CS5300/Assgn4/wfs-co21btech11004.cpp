/*
Name: - Darpan Gaur
Roll No: - CO21BTECH11004
*/
#include <iostream>
#include <atomic>
#include <vector>
#include <thread>
#include <random>
#include <unistd.h>     // for sleep()
#include <chrono> 
#include <string>
#include <mutex>
#include <algorithm>    // std::sort

using namespace std;

// class to store a value, timestamp and threadID
template <class T>
class stampedValue {
public:
    long stamp;
    T value;
    int pid;
    // empty constructor 
    // for declaring globllay and then assigning values later
    stampedValue() {

    }
    stampedValue (long ts, T v, int threadID) {
        stamp = ts;
        value = v;
        pid = threadID;
    }
};

// atomic wrapper for stampedValue
template <class T>
class atomicStampedValue {
public:
    atomic<stampedValue<T>> block;
    // empty constructor with default values
    atomicStampedValue() {
        block.store({0, 0, 0});
    }
    atomicStampedValue(long stamp, T value, int threadID) {
        block.store({stamp, value, threadID});
    }
    // copy constructor for easy syntax
    atomicStampedValue(const atomicStampedValue &obj) {
        block.store(obj.block.load());
    }
    // operator overloading for =
    // It is not possible to assign one atomic variable to another atomic variable directly.
    atomicStampedValue& operator=(const atomicStampedValue &obj) {
        block.store(obj.block.load());
        return *this;
    }
    // operator overloading for !=
    // it compares the stamp and pid of the two atomicStampedValue objects
    bool operator!=(const atomicStampedValue &obj) {
        return (block.load().stamp != obj.block.load().stamp) || (block.load().pid != obj.block.load().pid);
    }
};

// implementaion of MRMW Wait-Free Snapshot object from research paper
template <class T>
class SimpleSnapshot {
private:
    int capacity;
    // array of atomic MRMW registers
    vector<atomicStampedValue<T>> reg;
    // array of arrays to store the help snapshots
    vector<vector<T>> helpsnap;
    int nW;
public:
    SimpleSnapshot() {

    }
    SimpleSnapshot(int capacity, T init, int nW) {
        this->capacity = capacity;
        this->nW = nW;
        for (int i = 0; i < capacity; i++) {
            reg.push_back(atomicStampedValue<T>(0, init, 0));
        }
        helpsnap.assign(nW, vector<T>(capacity, init));
    }
    void update(int pos,T value, int pid, long stamp) {
        int me = pos;
        atomicStampedValue<T> oldValue = reg[me];
        atomicStampedValue<T> newValue = atomicStampedValue<T>(stamp+1, value, pid);
        reg[me] = newValue;
        T *snap = scan();
        for (int i = 0; i < capacity; i++) {
            helpsnap[pid][i] = snap[i];
        }
    }
    atomicStampedValue<T> *collect() {
        atomicStampedValue<T> *copy = new atomicStampedValue<T>[capacity];
        for (int j = 0; j < capacity; j++) {
            copy[j] = reg[j];
        }
        return copy;
    }
    T *scan() {
        bool canHelp[nW] = {false};
        atomicStampedValue<T> *aa, *bb;
        aa = collect();
        while (true) {
            bb = collect();
            bool flag = true;
            for (int x=0; x<capacity; x++) {
                if (aa[x] != bb[x]) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                T *res = new T[capacity];
                for (int i=0; i<capacity; i++) {
                    res[i] = bb[i].block.load().value;
                }
                return res;
            }
            for (int x=0;x<capacity;x++) {
                if (bb[x] != aa[x]) {
                    int w = bb[x].block.load().pid;
                    // if two times update is called by same thread
                    if (canHelp[w]) {
                        T *res = new T[capacity];
                        for (int i=0; i<capacity; i++) {
                            res[i] = helpsnap[w][i];
                        }
                        return res;
                    }
                    else {
                        canHelp[w] = true;
                    }
                }
            }
            aa = bb;
        }
        T *res = new T[capacity];
        return res;
    }
};

int nW, nS, M, k;
double muW, muS;

// create atomic int simple snapshot objects
SimpleSnapshot<int>* MRMW_Snap;

// a variable to inform the writer threads they have to terminate
atomic<bool> term;

// random number generator
random_device rd;
mt19937 gen(rd());

// start time in microseconds
auto startTime = chrono::high_resolution_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::microseconds>(startTime).count();

// output log
vector<pair<string,long>> outLog;

// mutex for log
mutex mtx;

// variables to store average and worst case time for snap and 
atomic<long> avgSnapTime(0);
atomic<long> worstSnapTime(0);
atomic<long> avgWriteTime(0);
atomic<long> worstWriteTime(0);
atomic<long> numWrites(0);

void writer(int threadID) {
    int v, pid = threadID;
    int t1, l;
    long stamp = 0;
    exponential_distribution<> exp(1/muW);
    uniform_int_distribution<> disL(0, M-1);
    uniform_int_distribution<> disV(1, 100);
    vector<pair<string, long>> localLog;
    while (!term) {
        v = disV(gen);
        l = disL(gen);
        
        // start time of write
        auto beginWrite = chrono::high_resolution_clock::now().time_since_epoch();
        auto BW = chrono::duration_cast<chrono::microseconds>(beginWrite).count();
        
        MRMW_Snap->update(l, v, threadID, stamp++);
        
        // end time of write
        auto endWrite = chrono::high_resolution_clock::now().time_since_epoch();
        auto EW = chrono::duration_cast<chrono::microseconds>(endWrite).count();
        
        string str = "Thr" + to_string(pid) + "'s write of " + to_string(v) + " on location " + to_string(l) + " at " + to_string(EW-S) + "\n";
        localLog.push_back({str, EW-S});

        avgWriteTime += EW-BW;
        if (EW-BW > worstWriteTime) {
            worstWriteTime = EW-BW;
        }
        numWrites++;

        t1 = exp(gen);
        sleep(t1);
    }
    mtx.lock();
    for (int i = 0; i < localLog.size(); i++) {
        outLog.push_back(localLog[i]);
    }
    mtx.unlock();
}

void snapshot(int threadId) {
    int i=0, t2;
    exponential_distribution<> exp(1/muS);
    vector<pair<string, long>> localLog;
    while (i<k) {
        auto beginCollect = chrono::high_resolution_clock::now().time_since_epoch();
        auto BC = chrono::duration_cast<chrono::microseconds>(beginCollect).count();
        // take snapshot
        int *snap = MRMW_Snap->scan();
        auto endCollect = chrono::high_resolution_clock::now().time_since_epoch();
        auto EC = chrono::duration_cast<chrono::microseconds>(endCollect).count();

        // log snapshot
        string str = "Snapshot Thr" + to_string(threadId) + "'s snapshot: ";
        for (int j = 0; j < M; j++) {
            str += to_string(snap[j]) + "-";
        }
        str += "which finished at " + to_string(EC-S) + "\n";
        localLog.push_back({str, EC-S});

        avgSnapTime += EC-BC;
        if (EC-BC > worstSnapTime) {
            worstSnapTime = EC-BC;
        }

        t2 = exp(gen);
        sleep(t2);
        i++;
    }
    // copy local log to global log
    mtx.lock();
    for (int i = 0; i < localLog.size(); i++) {
        outLog.push_back(localLog[i]);
    }
    mtx.unlock();
}

int main() {
    // open input file
    FILE *inFile = fopen("inp-params.txt", "r");

    if (inFile == NULL) {
        printf("Error: Could not open input file\n");
        return 0;
    }

    // take input nW, nS, M, muW, muS, k
    fscanf(inFile, "%d %d %d %lf %lf %d", &nW, &nS, &M, &muW, &muS, &k);
    cout  << "nW: " << nW << " nS: " << nS << " M: " << M << " muW: " << muW << " muS: " << muS << " k: " << k << endl;

    // intitialize atomic int simple snapshot objects
    MRMW_Snap = new SimpleSnapshot<int>(M, 0, nW);
    
    term = false;

    // writer threads and snapshot threads
    thread writerThreads[nW];
    thread snapshotThreads[nS];

    // create writer threads
    for (int i = 0; i < nW; i++) {
        writerThreads[i] = thread(writer, i);
    }

    // create snapshot threads
    for (int i = 0; i < nS; i++) {
        snapshotThreads[i] = thread(snapshot, i);
    }

    // join snapshot threads
    for (int i = 0; i < nS; i++) {
        snapshotThreads[i].join();
    }

    term = true;
    // join writer threads
    for (int i = 0; i < nW; i++) {
        writerThreads[i].join();
    }

    // print average and worst case time for snap and write
    cout << "Average snapshot time: " << (double)avgSnapTime/(nS*k) << endl;
    cout << "Worst snapshot time: " << worstSnapTime << endl;
    cout << "Average write time: " << (double)avgWriteTime/numWrites << endl;
    cout << "Worst write time: " << worstWriteTime << endl;
    // average and worst for write and snap combined
    cout << "Average time for write and snap combined: " << (double)(avgSnapTime+avgWriteTime)/(nS*k+numWrites) << endl;
    cout << "Worst time for write and snap combined: " << max(worstSnapTime, worstWriteTime) << endl;

    // sort output log by time in ascending order
    sort(outLog.begin(), outLog.end(), [](const pair<string, long> &a, const pair<string, long> &b) {
        return a.second < b.second;
    });

    // write output log to file
    FILE *outFile = fopen("output-log-wfs.txt", "w");
    if (outFile == NULL) {
        printf("Error: Could not open output file\n");
        return 0;
    }
    for (int i = 0; i < outLog.size(); i++) {
        fprintf(outFile, "%s", outLog[i].first.c_str());
    }

    // close input file
    fclose(inFile);
    fclose(outFile);

    // free memory
    delete MRMW_Snap;
    return 0;
}