/*
Name    :- DARPAN GAUR
Roll No :- CO21BTECH11004
*/

#include <iostream>
#include <thread>
#include <chrono>

using namespace std;

// global variables
int n,m,N;
int *arr;

// function to check if a number is prime or not
bool isPrime(int x) {
    if (x<2) return false;

    for (int i=2;i*i<=x;i++) if (x%i == 0) return false;
    
    return true;
}

// divide the work among threads based on id
void threadFunc(int id) {
    int i;
    if (id == 0 && N>=2) {
        if (isPrime(2)) {
            arr[2] = 1;
        }
    }
    for (i=2*id+1;i<=N;i+=2*m) {
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

    // for creation of m threads
    thread threads[m];
    
    for (i=0;i<m;i++) {
        threads[i] = thread(threadFunc, i);
    }

    for (i=0;i<m;i++) {
        threads[i].join();
    }

    // stop timer
    auto stop = chrono::high_resolution_clock::now();

    // print time taken
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time taken: " << duration.count() << " microseconds" << endl;

    // make output file that contains prime numbers
    FILE *outFile = fopen("Primes-SAM1.txt", "w");

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

    // open Time.txt file
    FILE *timeFile = fopen("Times.txt", "w");

    // write time taken to Time.txt file
    fprintf(timeFile, "SAM1 :- %ld microseconds\n", duration.count());

    // close files
    fclose(inFile);
    fclose(outFile);
    fclose(timeFile);

    // free allocated memory
    delete[] arr;

    return 0;
}