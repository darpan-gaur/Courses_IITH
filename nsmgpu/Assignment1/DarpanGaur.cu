/*
Name    :- Darpan Gaur
Email   :- darpangaur2003@gmail.com
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// per_row_column_kernel   1D grid and 1D blocks
__global__ void per_row_column_kernel(int *A, int *B, int *C, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    if (row < m) {
        for (int col = 0; col < n; col++) {
            i = row * n + col;
            C[col * m + row] = (A[i] + B[i]) - (B[i] - A[i]);
        }
    }
}

// per_column_row_kernel   1D grid and 2D blocks
__global__ void per_column_row_kernel(int *A, int *B, int *C, int m, int n) {
    int col = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int i;
    if (col < n) {
        for (int row = 0; row < m; row++) {
            i = row * n + col;
            C[col * m + row] = (A[i] + B[i]) - (B[i] - A[i]);
        }
    }
}

// per_element_kernel   2D grid and 2D blocks
__global__ void per_element_kernel(int *A, int *B, int *C, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i;
    if (row < m && col < n) {
        i = row * n + col;
        C[col * m + row] = (A[i] + B[i]) - (B[i] - A[i]);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        // Run the program as ./a.out <inputFile> <outputFile>
        printf("Usage: ./a.out <inputFile> <outputFile>\n");
        exit(1);
    }

    // take A and B as input from input file
    FILE *fp;
    char *inputFile = argv[1];
    fp = fopen(inputFile, "r");
    if (fp == NULL) {
        printf("Error opening input file\n");
        exit(1);
    }
    int m, n;
    fscanf(fp, "%d %d", &m, &n);
    int *A = (int *) malloc(m * n * sizeof(int));
    int *B = (int *) malloc(m * n * sizeof(int));
    int *C = (int *) malloc(m * n * sizeof(int));
    for (int i = 0; i < m * n; i++) {
        fscanf(fp, "%d", &A[i]);
    }
    for (int i = 0; i < m * n; i++) {
        fscanf(fp, "%d", &B[i]);
    }
    fclose(fp);

    // allocate memory on device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_B, m * n * sizeof(int));
    cudaMalloc(&d_C, m * n * sizeof(int));

    // copy A and B from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * n * sizeof(int), cudaMemcpyHostToDevice);

    // define grid and block dimensions
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1 , 1, 1);
    
    // start timer
    auto start = high_resolution_clock::now();

    //--------------------------------------------------------------------------------
    // perRowColumnKernel   1D grid and 1D blocks   // uncomment to run this kernel and comment other kernels
    // blockDim.x = 1024;
    // gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    // per_row_column_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n); 
    // cudaDeviceSynchronize();
    //--------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------
    // perColumnRowKernel   1D grid and 2D blocks   // uncomment to run this kernel and comment other kernels
    // blockDim.x = 32;
    // blockDim.y = 32;
    // gridDim.x = (n + blockDim.x*blockDim.y - 1) / (blockDim.x*blockDim.y);
    // per_column_row_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n);
    // cudaDeviceSynchronize();
    //--------------------------------------------------------------------------------
    

    //--------------------------------------------------------------------------------
    // perElementKernel   2D grid and 2D blocks  // uncomment to run this kernel and comment other kernels
    blockDim.x = 32;
    blockDim.y = 32;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    per_element_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n);
    cudaDeviceSynchronize();
    //--------------------------------------------------------------------------------

    // stop timer
    auto stop = high_resolution_clock::now();

    // print time taken by kernel
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Time taken by kernel: %f seconds\n", duration.count() / 1000000.0);

    // copy C from device to host
    cudaMemcpy(C, d_C, m * n * sizeof(int), cudaMemcpyDeviceToHost);

    // write C to output file
    char *outputFile = argv[2];
    fp = fopen(outputFile, "w");
    if (fp == NULL) {
        printf("Error opening output file\n");
        exit(1);
    }
    // fprintf(fp, "%d %d\n", n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m - 1; j++) {
            fprintf(fp, "%d ", C[i * m + j]);
        }
        fprintf(fp, "%d\n", C[i * m + m - 1]);
    }
    fclose(fp);

    // free memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free memory on host
    free(A);
    free(B);
    free(C);
    return 0;
}