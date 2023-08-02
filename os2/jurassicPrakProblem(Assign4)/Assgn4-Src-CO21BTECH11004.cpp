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

// Global variables
int P,C,k,passengerInMuseum=1;
double lambdaP,lambdaC,lambdaW;

// global pointer to car_track array
int *car_track;

// Global pointer to passenger time and car time
long *passengerTime;
long *carTime;

// Global pointer to output file
FILE *outFile;

// chrono start systemClock in microseconds
auto start = chrono::system_clock::now().time_since_epoch();
auto S = chrono::duration_cast<chrono::microseconds>(start).count();

// random number generator
random_device rd;
mt19937 gen(rd());

// define semaphore
sem_t *passenger_sem;
sem_t binSem,carAvail;

void passenger(long passenger_id){
    int i,j;

    // find tp : - wait between two successive ride units of a passenger
    exponential_distribution<double> tp(1/lambdaP);

    // passenger arrival time
    auto arrival = chrono::system_clock::now().time_since_epoch();
    auto A = chrono::duration_cast<chrono::microseconds>(arrival).count();

    // print entry entry time to output file
    fprintf(outFile,"Passenger %ld enters the museum at %ld\n",passenger_id,(A-S));

    // wander time (assume)
    exponential_distribution<double> wander(1/lambdaW);

    // wait for wander time
    sleep(wander(gen));

    for (i=0;i<k;i++){

        // passenger request time
        auto request = chrono::system_clock::now().time_since_epoch();
        auto R = chrono::duration_cast<chrono::microseconds>(request).count();

        // print request time to output file
        fprintf(outFile,"Passenger %ld made a ride request at %ld\n",passenger_id,(R-S));

        // wait for carAvail
        sem_wait(&carAvail);
        
        // wait for binSem
        sem_wait(&binSem);
        for (j=0;j<C;j++) {
            if (car_track[j]==0) {
                car_track[j] = passenger_id;
                
                // print request acceptence status to output file
                fprintf(outFile,"Car %d accepts passenger %ld's request\n",j+1,passenger_id);

                // signal binSem
                sem_post(&binSem);
                
                break;
            }
        }
        
        // passengen riding start time
        auto rideStart = chrono::system_clock::now().time_since_epoch();
        auto RS = chrono::duration_cast<chrono::microseconds>(rideStart).count();

        // print ride start time to output file
        fprintf(outFile,"Passenger %ld started riding at %ld\n",passenger_id,(RS-S));

        // wait for passenger_sem[passenger_id]
        sem_wait(&passenger_sem[passenger_id-1]);

        // passenger finished riding time
        auto rideFinish = chrono::system_clock::now().time_since_epoch();
        auto RF = chrono::duration_cast<chrono::microseconds>(rideFinish).count();

        // print ride finish time to output file
        fprintf(outFile,"Passenger %ld finished riding at %ld\n",passenger_id,(RF-S));

        // wait bwtween two successive ride units of a passenger
        sleep(tp(gen));

    }
    // passenger exit time
    auto exit = chrono::system_clock::now().time_since_epoch();
    auto E = chrono::duration_cast<chrono::microseconds>(exit).count();

    // update passengerTime
    passengerTime[passenger_id-1] = E-A;

    // print exit time to output file
    fprintf(outFile,"Passenger %ld exits the museum at %ld\n",passenger_id,(E-S));
    // cout << "Passenger " << passenger_id << " exits the museum at " << (E-S) << endl;
}

void car(long car_id){
    int i,j;

    // find tc : - wait between two successive ride units of a car
    exponential_distribution<double> tc(1/lambdaC);
    exponential_distribution<double> tw(1/lambdaW);

    while(passengerInMuseum){

        // wait for car_sem[thread_id]
        
        // sem_wait(&car_sem[car_id-1]);
        if (passengerInMuseum==0) {
            break;
        }
        if (car_track[car_id-1]==0) {
            continue;
        }
        
        // car riding start time
        auto rideStart = chrono::system_clock::now().time_since_epoch();
        auto RS = chrono::duration_cast<chrono::microseconds>(rideStart).count();

        // print car strted riding passenger to output file
        fprintf(outFile,"Car %ld is riding passenger %d\n",car_id,car_track[car_id-1]);
    

        // passenger riding time
        sleep(tw(gen));

        // print car finished riding passenger to output file
        fprintf(outFile,"Car %ld has finished Passenger %d's tour\n",car_id,car_track[car_id-1]);

        // car riding finish time
        auto rideEnd = chrono::system_clock::now().time_since_epoch();
        auto RE = chrono::duration_cast<chrono::microseconds>(rideEnd).count();

        // update carTime
        carTime[car_id-1] += RE-RS;

        // signal passenger_sem[car_track[car_id-1]]
        sem_post(&passenger_sem[car_track[car_id-1]-1]);

        sleep(tc(gen));

        // update car_track
        car_track[car_id-1] = 0;

        // signal carAvail
        sem_post(&carAvail);
    }

}

int main() {
    int i;

    // Open input file
    FILE *inFile = fopen("inp-params.txt","r");

    if (inFile == NULL) {
        printf("Error! Could not open input file\n");
    }
    // take input P,C,k and lambdaP,lambdaC
    fscanf(inFile,"%d %d %d %lf %lf",&P,&C,&k,&lambdaP,&lambdaC);

    // wanter time
    lambdaW = 0.01;

    // Open output file
    outFile = fopen("output.txt","w");  // write only
    
    // dynamically allocate memory for car_track
    car_track = (int *)malloc(C*sizeof(int));

    // initialize car_track
    for (i = 0; i < C; i++) car_track[i] = 0;

    // dynamically allocating memory to passengerTime and carTime
    passengerTime = (long *)malloc(P*sizeof(long));
    carTime = (long *)malloc(C*sizeof(long));

    // initialize passengerTime and carTime
    for (i = 0; i < P; i++) passengerTime[i] = 0;
    for (i = 0; i < C; i++) carTime[i] = 0;

    // initialize binSem and carAvail
    sem_init(&binSem,0,1);
    sem_init(&carAvail,0,C);

    // dynamically allocate memory for counting semaphores
    passenger_sem = (sem_t *)malloc(P*sizeof(sem_t));

    // initialize counting semaphore
    for (i = 0; i < P; i++) sem_init(&passenger_sem[i],0,0);

    // storing passenger number as passengerId
    long passengerId[P];
    for (i = 0; i < P; i++) passengerId[i] = i+1;

    // storing car number as carId
    long carId[C];
    for (i = 0; i < C; i++) carId[i] = i+1;

    // for creation of P passenger threads
    thread passengerThread[P];
    for (i = 0; i < P; i++) passengerThread[i] = thread(passenger,passengerId[i]);

    // for creation of C car threads
    thread carThread[C];
    for (i = 0; i < C; i++) carThread[i] = thread(car,carId[i]);

    // join passenger threads
    for (i = 0; i < P; i++) passengerThread[i].join();
    
    // passengerExited=0;
    passengerInMuseum = 0;

    // join car threads
    for (i = 0; i < C; i++) carThread[i].join();

    // find average passenger time
    long sumPassengerTime = 0;
    for (i = 0; i < P; i++) sumPassengerTime += passengerTime[i];
    double avgPassengerTime = (double)sumPassengerTime/P;
    cout << "Average time taken by passenger to complete the tour: " << avgPassengerTime << endl;

    // find average car time
    long sumCarTime = 0;
    for (i = 0; i < C; i++) sumCarTime += carTime[i];
    double avgCarTime = (double)sumCarTime/C;
    cout << "Average time taken by car to complete the tour: " << avgCarTime << endl;

    // destroy semaphores
    sem_destroy(&binSem);       
    sem_destroy(&carAvail);
    for (i = 0; i < P; i++) sem_destroy(&passenger_sem[i]);
    free(passengerTime);    // free passengerTime
    free(carTime);          // free carTime
    free(car_track);    // free car_track
    fclose(inFile);     // Close input file
    fclose(outFile);    // Close output file
    return 0;   
}

