#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// sequential matrix transpose
void transpose(int *a, int *b, int m, int n) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            b[j * m + i] = a[i * n + j];
        }
    }
}

__global__ void naiveTranspose(int *a, int *b, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        b[col * m + row] = a[row * n + col];
    }
}

__global__ void sharedMemTranspose(int *a, int *b, int m, int n) {
    __shared__ int data[32][32]; // dynamically allocate shared memory
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        data[threadIdx.y][threadIdx.x] = a[row + col * m];
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // print shared memory
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                printf("%d ", data[i ][ j]);
            }
            printf("\n");
        }
    }
    
    row = blockIdx.y * blockDim.y + threadIdx.x;
    col = blockIdx.x * blockDim.x + threadIdx.y;
    if (row < n && col < m) {
        b[row + col * n] = data[threadIdx.x][threadIdx.y];
    } 
}

    __global__ void transposeGPUcoalescing(int* input, int n, int m, int* output){
        int input_offset = blockIdx.x * 32 + blockIdx.y * m * 32;
        int output_offset = blockIdx.x * n * 32 + blockIdx.y * 32;
        __shared__ float tile[32][32+1];
        for (int i=0; i<32; ++i) {
        tile[i][threadIdx.x] = input[input_offset + threadIdx.x + i*m];
        }
        __syncthreads();
        for (int i=0; i<32; ++i) {
        output[output_offset + threadIdx.x + i*n] = tile[threadIdx.x][i];
        }
    }

__global__ void sharedMem2Transpose(int *a, int *b, int m, int n) {
    __shared__ int data[32][33];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        // printf("row = %d, col = %d, threadIdx.x = %d, threadIdx.y = %d\n", row, col, threadIdx.x, threadIdx.y);
        data[threadIdx.y][threadIdx.x] = a[row + col * m];
    }
    

    __syncthreads();
    row = blockIdx.y * blockDim.y + threadIdx.x;
    col = blockIdx.x * blockDim.x + threadIdx.y;
    if (row < n && col < m) {
        // printf("row = %d, col = %d, threadIdx.x = %d, threadIdx.y = %d\n", row, col, threadIdx.x, threadIdx.y);
        b[row + col * n] = data[threadIdx.x][threadIdx.y];
    }
    
}


bool checkTranspose(int *a, int *b, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void printMatrix(int *a, int m, int n) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            printf("%d ", a[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <m> <n>\n", argv[0]);
        exit(1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    printf("m = %d, n = %d\n", m, n);
    int *a = (int*)malloc(m * n * sizeof(int));
    // populate matrix with random values
    for (int i = 0; i < m * n; i++) {
        a[i] = rand() % 100;
    }
    int *aT = (int*)malloc(m * n * sizeof(int));
    // transpose matrix
    auto start = high_resolution_clock::now();
    transpose(a, aT, m, n);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Sequential transpose: %ld microseconds\n", duration.count());

    // printMatrix(a, m, n);
    // printf("\n");
    // printMatrix(aT, n, m);
    // printf("\n");

    int *d_A, *d_AT;
    int *aT_gpu = (int*)malloc(m * n * sizeof(int));
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_AT, m * n * sizeof(int));
    cudaMemcpy(d_A, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);

    // naive GPU transpose
    blockDim.x = 32;
    blockDim.y = 32;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    // blockDim.x = block_dim;
    // blockDim.y = block_dim;

    start = high_resolution_clock::now();
    naiveTranspose<<<gridDim, blockDim>>>(d_A, d_AT, m, n);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cudaMemcpy(aT_gpu, d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Navie GPU transpose is %s\n", checkTranspose(aT, aT_gpu, m, n) ? "correct" : "incorrect");
    printf("Naive GPU transpose: %ld microseconds\n", duration.count());

    // // shared memory GPU transpose
    // start = high_resolution_clock::now();
    // // send shared memory size to kernel dynamically 2d array
    // sharedMemTranspose<<<gridDim, blockDim>>>(d_A, d_AT, m, n);
    // cudaDeviceSynchronize();
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);

    // cudaMemcpy(aT_gpu, d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Shared memory GPU transpose is %s\n", checkTranspose(aT, aT_gpu, m, n) ? "correct" : "incorrect");
    // printf("Shared memory GPU transpose: %ld microseconds\n", duration.count());

    // printMatrix(aT_gpu, n, m);

    // transpose GPU coalescing
    start = high_resolution_clock::now();
    // send shared memory size to kernel dynamically 2d array
    // set d_AT to 0
    // cuda Memset
    cudaMemset(d_AT, 0, m * n * sizeof(int));   
    transposeGPUcoalescing<<<gridDim, blockDim>>>(d_A, m, n, d_AT);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cudaMemcpy(aT_gpu, d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Transpose GPU coalescing is %s\n", checkTranspose(aT, aT_gpu, n, m) ? "correct" : "incorrect");
    printf("Transpose GPU coalescing: %ld microseconds\n", duration.count());

    // printMatrix(aT_gpu, n, m);

    

    // shared memory GPU transpose 2
    // start = high_resolution_clock::now();
    // // set aT_gpu to 0
    // for (int i = 0; i < m * n; i++) {
    //     aT_gpu[i] = 0;
    // }
    // sharedMem2Transpose<<<gridDim, blockDim>>>(d_A, d_AT, m, n);
    // cudaDeviceSynchronize();
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    
    // cudaMemcpy(aT_gpu, d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Shared memory GPU transpose 2 is %s\n", checkTranspose(aT, aT_gpu, m, n) ? "correct" : "incorrect");
    // printf("Shared memory GPU transpose 2: %ld microseconds\n", duration.count());

    // printMatrix(aT_gpu, n, m);

    // free memory
    cudaFree(d_A);
    cudaFree(d_AT);

    free(a);
    free(aT);
    free(aT_gpu);

    return 0;
}