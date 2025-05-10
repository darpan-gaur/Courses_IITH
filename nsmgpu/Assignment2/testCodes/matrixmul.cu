#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// sequential matrix multiplication
void matmul(int *a, int *b, int *c, int p, int q, int r) {
    for (int i=0; i<p; i++) {
        for (int j=0; j<r; j++) {
            int sum = 0;
            for (int k=0; k<q; k++) {
                sum += a[i*q+k] * b[k*r+j];
            }
            c[i*r+j] = sum;
        }
    }
}

// kernel function for matrix multiplication
__global__ void matmul_kernel(int *a, int *b, int *c, int p, int q, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < p && col < r) {
        int sum = 0;
        for (int i=0; i<q; i++) {
            sum += a[row*q+i] * b[i*r+col];
        }
        c[row*r+col] = sum;
    }
}

// kernel function for matrix multiplication using shared memory
__global__ void matmul_kernel_shared(int *a, int *b, int *c, int p, int q, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int a_shared[32][32];
    __shared__ int b_shared[32][32];
    // if (row < p && col < r) {
    //     int sum = 0;
    //     for (int i=0; i<q; i+=32) {
    //         a_shared[threadIdx.y][threadIdx.x] = a[row*q+i+threadIdx.x];
    //         b_shared[threadIdx.y][threadIdx.x] = b[(i+threadIdx.y)*r+col];
    //         __syncthreads();
    //         for (int j=0; j<32; j++) {
    //             sum += a_shared[threadIdx.y][j] * b_shared[j][threadIdx.x];
    //         }
    //         __syncthreads();
    //     }
    //     c[row*r+col] = sum;
    // }
    int cVal = 0;
    for (int k=0;k < (32 + q - 1)/32; k++) {
        if (k*32 + threadIdx.x < q && row < p) {
            a_shared[threadIdx.y][threadIdx.x] = a[row*q+k*32+threadIdx.x];
        }
        else {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }
        if (k*32 + threadIdx.y < q && col < r) {
            b_shared[threadIdx.y][threadIdx.x] = b[(k*32+threadIdx.y)*r+col];
        }
        else {
            b_shared[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int j=0; j<32; j++) {
            cVal += a_shared[threadIdx.y][j] * b_shared[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row<p && col<r) {
        c[((blockIdx.y * blockDim.y + threadIdx.y)*r)+(blockIdx.x*blockDim.x)+threadIdx.x] = cVal;
    }
}

// check if the result is correct or not
bool check_result(int *c, int *c2, int p, int r) {
    for (int i=0; i<p*r; i++) {
        if (c[i] != c2[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    
    if (argc != 4) {
        printf("Usage: %s p q r\n", argv[0]);
        exit(1);
    }

    int p = atoi(argv[1]);
    int q = atoi(argv[2]);
    int r = atoi(argv[3]);
    int *a, *b, *c, *c2;
    int *d_a, *d_b, *d_c;
    // fill the arrays 'a' and 'b' with random integers between -10 to 10
    a = (int*)malloc(p*q*sizeof(int));
    b = (int*)malloc(q*r*sizeof(int));
    c = (int*)malloc(p*r*sizeof(int));
    c2 = (int*)malloc(p*r*sizeof(int));
    for (int i=0; i<p*q; i++) {
        a[i] = rand()%21 - 10;
    }
    for (int i=0; i<q*r; i++) {
        b[i] = rand()%21 - 10;
    }

    auto start = high_resolution_clock::now();
    matmul(a, b, c, p, q, r);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Time taken by CPU: %ld microseconds\n", duration.count());

    // allocate memory on the device
    cudaMalloc(&d_a, p*q*sizeof(int));
    cudaMalloc(&d_b, q*r*sizeof(int));
    cudaMalloc(&d_c, p*r*sizeof(int));

    // copy the arrays 'a' and 'b' to the device
    cudaMemcpy(d_a, a, p*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, q*r*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);

    blockDim.x = 32;
    blockDim.y = 32;
    gridDim.x = (r + blockDim.x - 1)/blockDim.x;
    gridDim.y = (p + blockDim.y - 1)/blockDim.y;

    // launch the kernel
    start = high_resolution_clock::now();
    matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, p, q, r);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cudaMemcpy(c2, d_c, p*r*sizeof(int), cudaMemcpyDeviceToHost);
    printf("matmul_kernel: Result: %s\n", check_result(c, c2, p, r) ? "CORRECT" : "INCORRECT");
    printf("Time taken by GPU: %ld microseconds\n", duration.count());

    // launch the kernel
    start = high_resolution_clock::now();
    matmul_kernel_shared<<<gridDim, blockDim>>>(d_a, d_b, d_c, p, q, r);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cudaMemcpy(c2, d_c, p*r*sizeof(int), cudaMemcpyDeviceToHost);
    printf("matmul_kernel_shared: Result: %s\n", check_result(c, c2, p, r) ? "CORRECT" : "INCORRECT");
    printf("Time taken by GPU with shared memory: %ld microseconds\n", duration.count());
    
    // free the memory allocated on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free the memory allocated on the host
    free(a);
    free(b);
    free(c);
    free(c2);
}