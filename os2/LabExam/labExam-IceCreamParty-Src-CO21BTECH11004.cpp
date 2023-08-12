#include <iostream>
#include <stdlib.h> 
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <random>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <semaphore>

using namespace std;

// global variables
int k,m,n;                          // k families, m trays, n times icecream eat
double Alpha, Beta, Lambda, Mu;     // exponentially distributed waiting times
int vendorExit=0;

// Global pointer to time to eat and time to refill
long *eatTime;
long refillTime;

// Global pointer to output file
FILE *outFile;

int iceCreamInPot = 0; 

// define semaphore
sem_t glock;
sem_t fillThePot,potIsFull;

// random number generator
random_device rd;
mt19937 gen(rd());

// chrono start systemClock in microseconds
auto start = chrono::system_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::microseconds>(start).count();

void cook(long cID){
    // int x=0;
    long x=0;
    while (true){
        if (vendorExit) break;
        sem_wait(&fillThePot);
        // passenger exit time
        auto startFillT = chrono::system_clock::now().time_since_epoch();
        auto SF = chrono::duration_cast<chrono::microseconds>(startFillT).count();

        // put serving in pot ----
        // m tray put time 
        exponential_distribution<double> putSl(m*1000.0/Lambda);

        // wait for m tray put time
        sleep(putSl(gen));

        // sem_wait(&glock);
        // iceCreamInPot = m;
        // sem_post(&glock);
        // ----

        // // passenger exit time
        // auto fillT = chrono::system_clock::now().time_since_epoch();
        // auto F = chrono::duration_cast<chrono::microseconds>(fillT).count();

        // // print exit time to output file
        // fprintf(outFile,"vendor refills the ice-cream %ld times at %ld\n",++x,(F-S));
        iceCreamInPot = m;

        // refill(&sleep) time
        exponential_distribution<double> refSl(1000.0/Mu);

        // wait for wander time
        sleep(refSl(gen));

        // passenger exit time
        auto fillT = chrono::system_clock::now().time_since_epoch();
        auto F = chrono::duration_cast<chrono::microseconds>(fillT).count();

        // print exit time to output file
        fprintf(outFile,"vendor refills the ice-cream %ld times at %ld\n",++x,(F-S));

        refillTime += F-SF;
        sem_post(&potIsFull);
    }
}

void familyFunc(long fID){
    // cout << "Falmily thread Id: - " << fID << "\n";
    int i;

    for (i=0;i<n;i++) {

        // family hungry time
        auto hungry = chrono::system_clock::now().time_since_epoch();
        auto H = chrono::duration_cast<chrono::microseconds>(hungry).count();

        // print request time to output file
        fprintf(outFile,"Family %ld becomes hungry at %ld\n",fID,(H-S));

        sem_wait(&glock);
        if (iceCreamInPot == 0){
            sem_post(&fillThePot);
            
            sem_wait(&potIsFull);
            // iceCreamInPot = m;
        }
        iceCreamInPot--;
        // // family eat time
        // auto eatTime = chrono::system_clock::now().time_since_epoch();
        // auto E = chrono::duration_cast<chrono::microseconds>(eatTime).count();

        // // print request time to output file
        // fprintf(outFile,"Family %ld eats from the Pot at %ld\n",fID,(E-S));
        sem_post(&glock);

        // family eat time
        auto eatTm = chrono::system_clock::now().time_since_epoch();
        auto E = chrono::duration_cast<chrono::microseconds>(eatTm).count();

        // print request time to output file
        fprintf(outFile,"Family %ld eats from the Pot at %ld\n",fID,(E-S));

        // eat -- sleep
        // eat time 
        exponential_distribution<double> eatSl(1000.0/Alpha);

        // wait for wander time
        sleep(eatSl(gen));


        eatTime[fID-1] += E-H;

        // community service --> sleep
        // wander time (assume)
        exponential_distribution<double> comSerSl(1000/Beta);

        // wait for wander time
        sleep(comSerSl(gen));


    }
    
    // family exit time
    auto exit = chrono::system_clock::now().time_since_epoch();
    auto ExT = chrono::duration_cast<chrono::microseconds>(exit).count();

    // print exit time to output file
    fprintf(outFile,"Family %ld has eaten n times. Hence, exits at %ld\n",fID,(ExT-S));
    
}


int main(){
    int i;

    // Open input file
    FILE *inFile = fopen("inp.txt","r");

    if (inFile == NULL) {
        printf("Error! Could not open input file\n");
    }

    // take input k, m, n, alpha, beta, lambda, mu
    fscanf(inFile,"%d %d %d %lf %lf %lf %lf", &k, &m, &n, &Alpha, &Beta, &Lambda, &Mu);
    
    // check input 
    // printf("K:- %d M:- %d N:- %d alpha:- %lf beta:- %lf lambda:- %lf mu:- %lf\n", k, m, n, Alpha, Beta, Lambda, Mu); 

    iceCreamInPot = m;

    // Open output file
    outFile = fopen("output.txt","w");  // write only

    // initialize the semaphore
    sem_init(&glock,0,1);
    sem_init(&fillThePot,0,0);
    sem_init(&potIsFull,0,0);

    // dynamically allocating memory to time to eat
    eatTime = (long *)malloc(k*sizeof(long));

    // initialize passengerTime and carTime
    for (i = 0; i < k; i++) eatTime[i] = 0;

    // create vendor thread
    long cookId[1];
    for (i=0;i<1;i++) cookId[i] = i+1;



    thread cookThread[1];
    for (i=0;i<1;i++) cookThread[i] = thread(cook, cookId[i]);

    // family exit time
    auto lst = chrono::system_clock::now().time_since_epoch();
    auto LST = chrono::duration_cast<chrono::microseconds>(lst).count();

    // printed here as thread stat working after creating so in for loop will crate problem
    fprintf(outFile,"K thread created : - %ld\n",(LST-S));

    // storing family number as familyID
    long familyId[k];
    for (i = 0; i < k; i++) familyId[i] = i+1;

    // for creation of k family threads
    thread familyThread[k];
    for (i = 0; i < k; i++) familyThread[i] = thread(familyFunc,familyId[i]);

    // join family threads
    for (i = 0; i < k; i++) familyThread[i].join();

    // family exit time
    auto lexit = chrono::system_clock::now().time_since_epoch();
    auto LExT = chrono::duration_cast<chrono::microseconds>(lexit).count();

    fprintf(outFile,"Last Thread Exits at time : - %ld\n",(LExT-S));
    
    vendorExit = 1;
    sem_post(&fillThePot);

    // join cook thread
    for (i=0;i<1;i++) cookThread[i].join();

    // find average waiting time to eat
    long sumEatTime = 0;
    for (i=0;i<k;i++) {
        sumEatTime += eatTime[i] ;
    }

    cout << "Average waiting time to eat: - " << sumEatTime/k << "\n";
    cout << "Average waiting time to refill : - " << refillTime/k << "\n";

    // destroy semaphores
    sem_destroy(&glock);       
    sem_destroy(&fillThePot);
    sem_destroy(&potIsFull);

    // free memory
    free(eatTime);

    fclose(inFile);     // Close input file
    fclose(outFile);    // Close output file

    return 0;
}