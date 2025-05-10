/*
Name    :- Darpan Gaur
Roll No :- CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <set>
#include <map>
#include <random>
#include <unistd.h> // for sleep

using namespace std;

// global variables
int n, m, numTrans, constVal,  numIters;
double lambda;

vector<long long> commitDelay, totalAbortCnt;

vector<int> threadTrans;

// random number generator
random_device rd;
mt19937 gen(rd());

// global start time in microseconds
auto startTime = chrono::high_resolution_clock::now();
auto S = chrono::duration_cast<chrono::microseconds>(startTime.time_since_epoch()).count();

// transaction class
class transaction {
    public:
        mutex tLock; // lock for transaction
        int tID; // transaction id
        int tStatus; // transaction status: 0 - active, 1 - committed, 2 - aborted

        vector<int> localMem; // local memory of transaction

        chrono::high_resolution_clock::time_point beginT;   // begin time of transaction
        chrono::high_resolution_clock::time_point endT;     // end time of transaction

        // read and write sets
        set<int> readSet;
        set<int> writeSet;


        // constructor
        transaction(int Id) {
            tID = Id;
            tStatus = 0;
            localMem = vector<int>(m, -1);
            beginT = chrono::high_resolution_clock::now();
        }
        

        // get-set functions

        
        // read function

        // write function

        // try commit function
};

// data item class
class dataItem {
    public:
        mutex dataItemLock; // lock for data item
        int val; // value of data item
        set<transaction*> readList; // read list

        // constructor
        dataItem(int v) {
            val = v;
            readList = set<transaction*>();
        }

        // read function
        int read() {
            return val;
        }

        // write function
        void write(int newVal) {
            val = newVal;
        }

};

// BOCC class
class boccScheduler {
    public:
        mutex schLock; // lock for scheduler -> for generating transaction id
        vector<dataItem*> dataItems; // data items
        map<int, transaction*> tList; // id->transaction map
        int idCounter; // transaction id counter

        // constructor
        boccScheduler() {
            dataItems = vector<dataItem*>();
            for (int i=0; i<m; i++) {
                dataItems.push_back(new dataItem(-1));
            }
            tList = map<int, transaction*>();
            idCounter = 0;
        }

        // begin transaction function
        int beginTrans() {
            schLock.lock(); 

            int id = idCounter++;   
            transaction* t = new transaction(id);  
            tList[id] = t;

            schLock.unlock();
            
            return id;
        }

        // read function
        int read(int i, int x, int* l) {
            // obtain locks
            // schLock.lock();
            dataItems[x]->dataItemLock.lock();
            transaction* t = tList[i];
            t->tLock.lock();    // check if this lock is required
        
            // check if aborted
            if (t->tStatus == 2) {
                dataItems[x]->readList.erase(t);

                // unlock
                t->tLock.unlock();
                dataItems[x]->dataItemLock.unlock();
                schLock.unlock();
                return -1;
            }
            // store value in local memory
            *l = dataItems[x]->read();
            t->readSet.insert(x);

            // unlock
            t->tLock.unlock();
            dataItems[x]->dataItemLock.unlock();
            // schLock.unlock();

            return 0;
        }

        // write function
        int write(int i, int x, int l) {
            // obtain locks
            // schLock.lock();
            dataItems[x]->dataItemLock.lock();
            transaction* t = tList[i];
            t->tLock.lock();    // check if this lock is required

            // update transaction variables
            t->localMem[x] = l;
            t->writeSet.insert(x);

            // update read list
            dataItems[x]->readList.insert(t);

            // unlock
            t->tLock.unlock();
            dataItems[x]->dataItemLock.unlock();
            // schLock.unlock();

            return 0;
        }

        // tryCommit function
        int tryCommit(int i) {
            /*
            Returns 
                    '1' if transaction is committed
                    '2' if transaction is aborted
            */
            // lock
            // schLock.lock();
            
            set<int> rIntersectW; // read-write intersection
            // lock data item in increasing order of id
            for (auto x : tList[i]->readSet) {
                dataItems[x]->dataItemLock.lock();
                
                // size of read list
                for (auto t : dataItems[x]->readList) {
                    if (t->tStatus != 1) continue; // if not committed
                    if (tList[i]->beginT > t->endT) continue; // if   not concurrent
                    rIntersectW.insert(t->tID);
                }
                
            }
            // remove current transaction if present
            rIntersectW.erase(i);   
            if (rIntersectW.size() == 0) {
                // commit
                transaction* t = tList[i];
                t->endT = chrono::high_resolution_clock::now();
                t->tStatus = 1;
                for (auto x : tList[i]->writeSet) {
                    dataItems[x]->write(tList[i]->localMem[x]);
                    dataItems[x]->dataItemLock.unlock();
                }
                // schLock.unlock();
                return 1;
            }
            else {
                // abort
                transaction* t = tList[i];
                t->endT = chrono::high_resolution_clock::now();
                t->tStatus = 2;
                for (auto x : tList[i]->readSet) {
                    dataItems[x]->readList.erase(tList[i]);
                    dataItems[x]->dataItemLock.unlock();
                }
                // schLock.unlock();
                return 2;
            }
        }
        // garbage collection

        // write function for destructor
        ~boccScheduler() {
            for (auto x : dataItems) {
                delete x;
            }
            for (auto x : tList) {
                delete x.second;
            }
        }
};

// global scheduer object
boccScheduler* scheduler;

// open output file
FILE *logFile = fopen("boccLog.txt", "w");

// uptMem function
void updtMem(int threadID) {
    int status = 2;     // declare status variable
    int abortCnt = 0;   // keep track of abort count

    // time variables
    chrono::high_resolution_clock::time_point critStartTime, critEndTime;

    // exponential distribution
    // exponential_distribution<double> exp(1000.0/lambda);
    exponential_distribution<double> exp(lambda);

    int t_trans = threadTrans[threadID]; // number of transactions for this thread

    // each thread invokes numTrans Transactions!
    for (int curTrans=0; curTrans<t_trans; curTrans++) {
        abortCnt = 0;   // reset abort count
        auto critStartTime = chrono::high_resolution_clock::now(); // keep track of critical section start time
        auto CST = chrono::duration_cast<chrono::microseconds>(critStartTime.time_since_epoch()).count();

        do {
            int id = scheduler->beginTrans();    // begin a new transaction

            int locVal; 
            for (int i=0; i<numIters; i++) {

                int randInt = rand()%m; // get the next random index to be updated
                int randVal = rand()%constVal;  // gets a random value using the constant constVal

                // reads the shared value at index randInd into locVal
                scheduler->read(id, randInt, &locVal);

                auto readTime = chrono::high_resolution_clock::now(); // read time
                auto RT = chrono::duration_cast<chrono::microseconds>(readTime.time_since_epoch()).count();
                fprintf(logFile, "Thread ID %d Transaction %d reads from %d a value %d at time %ld\n", threadID, id, randInt, locVal, RT-S);

                locVal += randVal;  // update the value

                // request to write back
                scheduler->write(id, randInt, locVal);

                auto writeTime = chrono::high_resolution_clock::now(); // write time
                auto WT = chrono::duration_cast<chrono::microseconds>(writeTime.time_since_epoch()).count();
                fprintf(logFile, "Thread ID %d Transaction %d writes to %d a value %d at time %ld\n", threadID, id, randInt, locVal, WT-S);

                // sleep for random amount of time which simulates some complex computation
                // sleep(exp(gen));
                this_thread::sleep_for(chrono::milliseconds((int)lambda));
            }
            
            status = scheduler->tryCommit(id);   // try to commit the transaction
            
            auto commitTime = chrono::high_resolution_clock::now(); // commit time
            auto CT = chrono::duration_cast<chrono::microseconds>(commitTime.time_since_epoch()).count();
            fprintf(logFile, "Transaction %d tryCommits with result %d at time %ld\n\n", id, status, CT-S);

            if (status == 2) {
                abortCnt++;
            }   
        }
        while (status != 1);    

        auto critEndTime = chrono::high_resolution_clock::now(); // keep track of critical section end time
        auto CET = chrono::duration_cast<chrono::microseconds>(critEndTime.time_since_epoch()).count();
        commitDelay[threadID] += CET - CST; // update commit delay
        totalAbortCnt[threadID] += abortCnt; // update total abort count
    }
}

int main() {
    // open input file
    FILE *inFile = fopen("inp-params.txt", "r");

    // error handling
    if (inFile == NULL) {
        cout << "Error in opening file" << endl;
        return 0;
    }
    
    // read input from file
    fscanf(inFile, "%d %d %d %d %lf %d", &n, &m, &numTrans, &constVal, &lambda, &numIters);

    // print input
    cout << "Input Params: N=" << n << " M = " << m << " numTrans = " << numTrans << " constVal = " << constVal << " lambda = " << lambda << " numIters = " << numIters << endl;

    // allocate memory
    commitDelay = vector<long long>(n, 0);
    totalAbortCnt = vector<long long>(n, 0);
    threadTrans = vector<int>(n, 0);

    for (int i = 0; i < n; i++) {
        threadTrans[i] = numTrans/n;
    }
    threadTrans[n-1] += numTrans%n;

    // boccScheduler scheduler;
    scheduler = new boccScheduler();

    // create n threads
    vector<thread> threads;
    for (int i = 0; i < n; i++) {
        threads.push_back(thread(updtMem, i));
    }

    // join threads
    for (int i = 0; i < n; i++) {
        threads[i].join();
    }

    // print avg commit delay and total abort count
    long long totalDelay = 0, totalAborts = 0;
    for (int i = 0; i < n; i++) {
        totalDelay += commitDelay[i];
        totalAborts += totalAbortCnt[i];
    }
    // cout << "Avg totalDelay: " << (double)totalDelay/(n*numTrans) << endl;
    // cout << "Avg AbortCnt: " << (double)totalAborts/(n*numTrans) << endl;
    cout << "Avg totalDelay: " << (double)totalDelay/(numTrans) << endl;
    cout << "Avg AbortCnt: " << (double)totalAborts/(numTrans) << endl;

    // close file
    fclose(inFile);
    fclose(logFile);

    // delete scheduler
    delete scheduler;

    return 0;
}