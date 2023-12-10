/*
Name    :- DARPAN GAUR
Roll No :- CO21BTECH11004
*/

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

// global variables
int n,m,N;
int *arr;
int idx=1;

// mutex lock
mutex mtx;

// function to check if a number is prime or not
bool isPrime(int x) {
    if (x<2) return false;

    for (int i=2;i*i<=x;i++) if (x%i == 0) return false;
    
    return true;
}

int increment(int &i) {
    i++;
    return i;
}

void threadFunc(int id) {
    int i;
    while (idx<=N) {
        mtx.lock();
        i = increment(idx);
        if (idx>3 && idx<N) {
            i = increment(idx);
        }
        mtx.unlock();
        if (isPrime(i)) {
            arr[i] = 1;
        }
    }
}

int main() {
    int i;

    // open input file
    FILE *inFile = fopen("inp-params.txt", "r");

    if (inFile == NULL) {
        cout << "Error opening file" << endl;
        return 0;
    }

    fscanf(inFile, "%d %d", &n, &m);

    N = 1;
    for (i=0;i<n;i++) N *= 10;

    // dynamically allocate array
    arr = new int[N+1];
    for (i=0;i<=N;i++) {
        arr[i] = 0;
    }

    // start timer
    auto start = chrono::high_resolution_clock::now();

    // create threads
    thread threads[m];

    for (i=0;i<m;i++) {
        threads[i] = thread(threadFunc, i);
    }

    // join threads
    for (i=0;i<m;i++) {
        threads[i].join();
    }

    // stop timer
    auto stop = chrono::high_resolution_clock::now();

    // print time taken
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time taken: " << duration.count() << " microseconds" << endl;

    // write to output file
    FILE *outFile = fopen("Primes-DAM1.txt", "w");

    if (outFile == NULL) {
        cout << "Error opening file" << endl;
        return 0;
    }

    // write to output file
    for (i=0;i<=N;i++) {
        if (arr[i] == 1) {
            fprintf(outFile, "%d ", i);
        }
    }
    
    // open Times.txt file
    FILE *timesFile = fopen("Times.txt", "w");

    // write to Times.txt file
    fprintf(timesFile, "DAM1 :- %ld microseconds\n", duration.count());

    // close files
    fclose(inFile);
    fclose(outFile);
    fclose(timesFile);

    // free memory
    delete[] arr;

    return 0;
}