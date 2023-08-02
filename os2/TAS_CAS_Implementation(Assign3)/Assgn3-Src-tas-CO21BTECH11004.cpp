#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <random>
#include <unistd.h>
#include <pthread.h>
#include <iomanip>

using namespace std;



// Global Variables
double lamda1 = 0.5;
double lamda2 = 0.5;
int n,k;
long *AvgTime,*WorstTime;

// Global pointer to output file
FILE *out_file;

// atomic lock for stream
atomic_flag lock_stream = ATOMIC_FLAG_INIT;

// chrono start system_clock in nano seconds
auto start = chrono::system_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::microseconds>(start).count();


// random number generator
random_device rd;
mt19937 gen(rd());

void testCS(long thread_id)
{
    int i;    

    // find t1 and t2 delay which are exponentially distributed with average lamda1 and lamda2
    exponential_distribution<double> t1(1/lamda1);
    exponential_distribution<double> t2(1/lamda2);

    for (i=0;i<k;i++){


        // get reqEnterTime : - sytem time in microseconds    
        auto start = chrono::system_clock::now().time_since_epoch();
        auto reqEnterTime = chrono::duration_cast<chrono::microseconds>(start).count();

        // printing request time to output file
        fprintf(out_file,"%dth CS Request at %ldns by thread %ld\n",i,reqEnterTime-S,thread_id);
        
        // entry−sec() ; // Entry Section

        while (atomic_flag_test_and_set(&lock_stream)) {
            // do nothing
        }

        // get actEnterTime :- sytem time in microseconds
        auto end = chrono::system_clock::now().time_since_epoch();
        auto actEnterTime = chrono::duration_cast<chrono::microseconds>(end).count();

        
        // Adding time to enter critical section at index (thread number) in AvgTime global array
        AvgTime[thread_id-1] += (actEnterTime-reqEnterTime);

        // Updating worst time to enter critical section at index (thread number) in WorstTime global array 
        WorstTime[thread_id-1] = max(WorstTime[thread_id-1],(actEnterTime-reqEnterTime));

        // printing entry time to the output file
        fprintf(out_file,"%dth CS Entry at %ldns by thread %ld\n",i,actEnterTime-S,thread_id);

        // sleep for t1
        sleep(t1(gen));     // simulation of critical section

        // exit−sec () ; // Exit Section

        // get exitTime :- sytem time in microseconds
        auto end2 = chrono::system_clock::now().time_since_epoch();
        auto exitTime = chrono::duration_cast<chrono::microseconds>(end2).count();

        // printing exit time to the output file
        fprintf(out_file,"%dth CS Exit at %ldns by thread %ld\n",i,exitTime-S,thread_id);

        lock_stream.clear();

        // sleep for t2
        sleep(t2(gen));     // Simulation of Reminder Section
    }
}

int main()
{   
    int i;

    // Open input file
    FILE* in_file = fopen("input.txt","r"); // read only

    if (in_file == NULL) {
        printf("Error! Could not open input file\n");
    }

    // open output file
    out_file = fopen("output.txt","w"); // write only

    // take input n,k,lamda1 and lamda2 from input file
    fscanf(in_file,"%d %d %lf %lf",&n,&k,&lamda1,&lamda2);

    // Dynamically allocate memory to Global arry :- AvgTime and WorstTime
    AvgTime = (long *)malloc(n*sizeof(long long));
    WorstTime = (long *)malloc(n*sizeof(long long));

    // Initializing AvgTime and WorstTime array to 0.
    for (i=0;i<n;i++){
        AvgTime[i] = 0;
        WorstTime[i] = 0;
    }

    thread tid[n];
    
    // storing thread number as thread_id
    long thread_id[n];
    for (i=0;i<n;i++) thread_id[i] = i+1;

    // for creation of n threads
    for (int i=0;i<n;i++){
        /* create the thread */
        tid[i] = thread(testCS,thread_id[i]);
       
    }

    // joining of n threads
    for (int i=0;i<n;i++){
        tid[i].join();
    }

    // Calculating average wait time and Worst wait time
    long at=0,wt=0;
    for (i=0;i<n;i++){
        at += AvgTime[i];
        wt = max(wt,WorstTime[i]);
    }
    
    // printing average and worst time
    cout << "Average Time = " << at/(n*k*1.0) << "\n";
    cout << "Worst Time = " << wt << "\n";

    free(AvgTime);          // free the memory allocated to AvgTime arr
    free(WorstTime);        // free the memeory allocated to WorstTime arr
    fclose(in_file);        // Close input.txt
    fclose(out_file);       // Close output.txt

    return 0;
}

