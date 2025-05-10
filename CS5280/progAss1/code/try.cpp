#include <iostream>
#include <chrono>
#include <random>
#include <unistd.h>

using namespace std;

// random number generator
random_device rd;
mt19937 gen(rd());

int main() {
    double lambda = 300;
    // exponential distribution
    exponential_distribution<double> exp(lambda);

    auto start = chrono::high_resolution_clock::now();
    auto S = chrono::duration_cast<chrono::microseconds>(start.time_since_epoch()).count();

    // cout << "Start time: " << S << endl;

    // sleep for random amount of time which simulates some complex computation
    sleep(exp(gen));

    auto end = chrono::high_resolution_clock::now();
    auto E = chrono::duration_cast<chrono::microseconds>(end.time_since_epoch()).count();

    // cout << "End time: " << E << endl;

    cout << "Time taken: " << E - S << endl;

    return 0;
}