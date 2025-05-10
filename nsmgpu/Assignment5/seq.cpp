#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

/*
Matrix multiplication
*/

int main(int argc, char *argv[]){
    if (argc != 3) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
    }

    // take input from flie
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Cannot open input file %s\n", argv[1]);
        exit(1);
    }
    int m, k, n;
    fscanf(fp, "%d %d %d", &m, &k, &n);
    long long *A = (long long *)malloc(sizeof(long long) * m * k);
    long long *B = (long long *)malloc(sizeof(long long) * k * n);
    long long *C = (long long *)malloc(sizeof(long long) * m * n);
    for (int i = 0; i < m * k; i++) {
        fscanf(fp, "%lld", &A[i]);
    }
    for (int i = 0; i < k * n; i++) {
        fscanf(fp, "%lld", &B[i]);
    }
    fclose(fp);

    // matrix multiplication
    auto start = high_resolution_clock::now();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            C[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    auto stop = high_resolution_clock::now();
    // time in milliseconds
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " milliseconds" << endl;

    // write output to file
    fp = fopen(argv[2], "w");
    if (fp == NULL) {
        printf("Cannot open output file %s\n", argv[2]);
        exit(1);
    }
    // fprintf(fp, "%d %d\n", m, n);
    for (int i = 0; i < m * n; i++) {
        fprintf(fp, "%lld ", C[i]);
    	if ((i+1)%n == 0) {
		fprintf(fp, "\n");
	}
    }
    fprintf(fp, "\n");
    fclose(fp);

    free(A);
    free(B);
    free(C);
    return 0;
}
