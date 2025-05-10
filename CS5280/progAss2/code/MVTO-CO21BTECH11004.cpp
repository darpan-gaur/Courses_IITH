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
int n, m, numTrans, constVal;
double lambda, readRatio;

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
        // set<transaction*> readList; // read list
        set<int> W_TS; // write timestamp
        set<pair<int, int>> R_TS; // read timestamp

        // constructor
        dataItem(int v) {
            val = v;
            // readList = set<transaction*>();
            W_TS = set<int>();
            // W_TS.insert(-2);
            R_TS = set<pair<int, int>>();
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

// BTO class
class MVTOscheduler {
    public:
        mutex schLock; // lock for scheduler -> for generating transaction id
        vector<dataItem*> dataItems; // data items
        map<int, transaction*> tList; // id->transaction map
        int idCounter; // transaction id counter

        // constructor
        MVTOscheduler() {
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
            // t->tLock.lock();    // check if this lock is required
            
            if (t->localMem[x] != -1) {
                *l = t->localMem[x];
                // unlock
                // t->tLock.unlock();
                dataItems[x]->dataItemLock.unlock();
                // schLock.unlock();
                return 0;
            }
            // timestamp check
            int ts = -1;
            for (auto k: dataItems[x]->W_TS) {
                if (k < t->tID && k > ts) {
                    ts = k;
                }
            }
            if (ts == -1 && dataItems[x]->W_TS.size() > 0) {
                t->tStatus = 2; // abort
                // unlock
                // t->tLock.unlock();
                dataItems[x]->dataItemLock.unlock();
                return -1;
            }
                
            // check if aborted
            if (t->tStatus == 2) {
                // dataItems[x]->readList.erase(t);
                // if (dataItems[x]->readList.find(t) != dataItems[x]->readList.end()) {
                //     dataItems[x]->readList.erase(t);
                // }

                // unlock
                // t->tLock.unlock();
                dataItems[x]->dataItemLock.unlock();
                // schLock.unlock();
                return -1;
            }
            // update R_TS
            // dataItems[x]->R_TS = max(dataItems[x]->R_TS, t->tID);
            dataItems[x]->R_TS.insert({t->tID, ts});


            // store value in local memory
            *l = dataItems[x]->read();
            t->readSet.insert(x);

            // unlock
            // t->tLock.unlock();
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
            // t->tLock.lock();    // check if this lock is required

            // timestamp check
            // if (t->tID < dataItems[x]->W_TS || t->tID < dataItems[x]->R_TS) {
            //     t->tStatus = 2; // abort
            //     // unlock
            //     // t->tLock.unlock();
            //     dataItems[x]->dataItemLock.unlock();
            //     return -1;
            // }
            // iterate through pair in R_TS
            // for (auto [j, k] : dataItems[x]->R_TS) {
            //     if (j > t->tID && k < t->tID) {
            //         t->tStatus = 2; // abort
            //         // unlock
            //         // t->tLock.unlock();
            //         dataItems[x]->dataItemLock.unlock();
            //         return -1;
            //     }
            // }

            // // update W_TS
            // // dataItems[x]->W_TS = max(dataItems[x]->W_TS, t->tID);
            // dataItems[x]->W_TS.insert(t->tID);

            // update transaction variables
            t->localMem[x] = l;
            t->writeSet.insert(x);

            // update read list
            // dataItems[x]->readList.insert(t);

            // unlock
            // t->tLock.unlock();
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

            // lock the transaction
            transaction* t = tList[i];
            // t->tLock.lock();
            // check if transaction is aborted
            // if (t->tStatus == 2) {
            //     for (auto x : t->readSet) {
            //         dataItems[x]->dataItemLock.lock();
            //         // dataItems[x]->readList.erase(t);
            //         // if (dataItems[x]->readList.find(t) != dataItems[x]->readList.end()) {
            //         //     dataItems[x]->readList.erase(t);
            //         // }
            //         dataItems[x]->dataItemLock.unlock();
            //     }
            //     // t->tLock.unlock();
            //     return 2;
            // }

            // update data items
            for (auto x : t->writeSet) {
                dataItems[x]->dataItemLock.lock();
                // timestamp check
                for (auto [j, k] : dataItems[x]->R_TS) {
                    if (j > t->tID && k < t->tID) {
                        t->tStatus = 2; // abort
                        // unlock
                        // t->tLock.unlock();
                        dataItems[x]->dataItemLock.unlock();
                        return -1;
                    }
                }
                // update W_TS
                dataItems[x]->W_TS.insert(t->tID);

                dataItems[x]->write(t->localMem[x]);
                dataItems[x]->dataItemLock.unlock();
            }
            if (t->tStatus == 2) {
                return 2;
            }
            // update transaction status
            t->tStatus = 1;
            t->endT = chrono::high_resolution_clock::now();
            
            // unlock transaction
            // t->tLock.unlock();
            return 1;
        }   
        // garbage collection

        // destructor
        ~MVTOscheduler() {
            for (auto x : dataItems) {
                delete x;
            }
            for (auto x : tList) {
                delete x.second;
            }
        }
};

// global scheduler object
MVTOscheduler* scheduler;

// open output file
FILE *logFile = fopen("MVTOlog.txt", "w");

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

        // Based on the readChance, this transaction is set to be read-only or not
        // generate random number between 0 and 1
        double readChance = (double)rand()/RAND_MAX;
        int readOnly = readChance < readRatio ? 1 : 0;

        auto critStartTime = chrono::high_resolution_clock::now(); // keep track of critical section start time
        auto CST = chrono::duration_cast<chrono::microseconds>(critStartTime.time_since_epoch()).count();

        do {
            int id = scheduler->beginTrans();    // begin a new transaction

            int locVal; 
            int numIters = rand()%m; // get the number of iterations for this transaction
            // cout << numIters << endl;
            numIters = rand()%500;
            for (int i=0; i<numIters; i++) {

                int randInt = rand()%m; // get the next random index to be updated
                int randVal = rand()%constVal;  // gets a random value using the constant constVal

                // reads the shared value at index randInd into locVal
                scheduler->read(id, randInt, &locVal);

                auto readTime = chrono::high_resolution_clock::now(); // read time
                auto RT = chrono::duration_cast<chrono::microseconds>(readTime.time_since_epoch()).count();
                fprintf(logFile, "Thread ID %d Transaction %d reads from %d a value %d at time %ld\n", threadID, id, randInt, locVal, RT-S);
                
                if (readOnly == 0) {
                    locVal += randVal;  // update the value

                    // request to write back
                    scheduler->write(id, randInt, locVal);

                    auto writeTime = chrono::high_resolution_clock::now(); // write time
                    auto WT = chrono::duration_cast<chrono::microseconds>(writeTime.time_since_epoch()).count();
                    fprintf(logFile, "Thread ID %d Transaction %d writes to %d a value %d at time %ld\n", threadID, id, randInt, locVal, WT-S);
                }

                // sleep for random amount of time which simulates some complex computation
                sleep(exp(gen));
                // this_thread::sleep_for(chrono::milliseconds((int)lambda));
                // this_thread::sleep_for(chrono::microseconds((int)exp(gen)));
            }
            
            status = scheduler->tryCommit(id);   // try to commit the transaction
            
            auto commitTime = chrono::high_resolution_clock::now(); // commit time
            auto CT = chrono::duration_cast<chrono::microseconds>(commitTime.time_since_epoch()).count();
            fprintf(logFile, "Transaction %d tryCommits with result %d at time %ld\n\n", id, status, CT-S);

            if (status == 2) {
                abortCnt++;
            }   

            // cout << "Thread ID: " << threadID << " Current Transaction: " << curTrans << " currIter: " << numIters << " Num Aborts: " << abortCnt << endl;
        }
        while (status != 1);    

        auto critEndTime = chrono::high_resolution_clock::now(); // keep track of critical section end time
        auto CET = chrono::duration_cast<chrono::microseconds>(critEndTime.time_since_epoch()).count();
        commitDelay[threadID] += CET - CST; // update commit delay
        totalAbortCnt[threadID] += abortCnt; // update total abort count
        // fprintf(logFile, "Thread ID %d Transaction %d commits with result %d at time %ld\n", threadID, curTrans, status, CET-S);
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
    // fscanf(inFile, "%d %d %d %d %lf %d", &n, &m, &numTrans, &constVal, &lambda, &numIters);
    // fscanf(inFile, "%d %d %d %d %lf", &n, &m, &numTrans, &constVal, &lambda);
    fscanf(inFile, "%d %d %d %d %lf %lf", &n, &m, &numTrans, &constVal, &lambda, &readRatio);

    // print input
    // cout << "Input Params: N=" << n << " M = " << m << " numTrans = " << numTrans << " constVal = " << constVal << " lambda = " << lambda << " numIters = " << numIters << endl;
    // cout << "Input Params: N=" << n << " M = " << m << " numTrans = " << numTrans << " constVal = " << constVal << " lambda = " << lambda << endl;
    cout << "Input Params: N=" << n << " M = " << m << " numTrans = " << numTrans << " constVal = " << constVal << " lambda = " << lambda << " readRatio = " << readRatio << endl;

    // allocate memory
    commitDelay = vector<long long>(n, 0);
    totalAbortCnt = vector<long long>(n, 0);
    threadTrans = vector<int>(n, 0);

    for (int i = 0; i < n; i++) {
        threadTrans[i] = numTrans/n;
    }
    threadTrans[n-1] += numTrans%n;
    
    // boccScheduler scheduler;
    scheduler = new MVTOscheduler();

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
    // cout << (double)totalDelay/(n*numTrans) << " " << (double)totalAborts/(n*numTrans) << endl;

    // close file
    fclose(inFile);
    fclose(logFile);

    // delete scheduler
    delete scheduler;

    return 0;
}