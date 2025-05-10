#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]){
    // take input from input.txt
    FILE *fp;
    char *inputFile = argv[1];
    fp = fopen(inputFile, "r");
    if (fp == NULL){
        printf("Error opening input file\n");
        exit(1);
    }
    int m, n;
    fscanf(fp, "%d %d", &m, &n);
    // dynamically allocate memory for A and B
    int **A = (int **)malloc(m * sizeof(int *));
    for(int i=0; i<m; i++){
        A[i] = (int *)malloc(n * sizeof(int));
    }
    int **B = (int **)malloc(m * sizeof(int *));
    for(int i=0; i<m; i++){
        B[i] = (int *)malloc(n * sizeof(int));
    }
    // read A and B
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            fscanf(fp, "%d", &A[i][j]);
        }
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            fscanf(fp, "%d", &B[i][j]);
        }
    }
    fclose(fp);


    // compute C
    // int C[n][m];
    int **C = (int **)malloc(n * sizeof(int *));
    for(int i=0; i<n; i++){
        C[i] = (int *)malloc(m * sizeof(int));
    }

    // start timer
    auto start = high_resolution_clock::now();
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            C[i][j] = (A[j][i] + B[j][i]) - (B[j][i] - A[j][i]);
        }
    }
    // stop timer
    auto stop = high_resolution_clock::now();

    // print time taken by kernel
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Time taken by kernel: %f seconds\n", duration.count() / 1000000.0);

    // write output to output.txt
    char *outputFile = argv[2];
    fp = fopen(outputFile, "w");
    if (fp == NULL){
        printf("Error opening output file\n");
        exit(1);
    }
    // fprintf(fp, "%d %d\n", n, m);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            fprintf(fp, "%d ", C[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // free memory
    for(int i=0; i<m; i++){
        free(A[i]);
    }
    free(A);
    for(int i=0; i<m; i++){
        free(B[i]);
    }
    free(B);
    for(int i=0; i<n; i++){
        free(C[i]);
    }
    free(C);

    return 0;
}