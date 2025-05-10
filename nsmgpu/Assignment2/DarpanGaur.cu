#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matadd(int *a, int *b, int *c, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

__global__ void Transpose(int *a, int *b, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        b[col * m + row] = a[row * n + col];
    }
}

// transpose using shared memory
__global__ void transpose(int *a, int *b, int m, int n) {
    __shared__ int data[32][33];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        data[threadIdx.y][threadIdx.x] = a[row + col * m];
    }
    __syncthreads();
    row = blockIdx.y * blockDim.y + threadIdx.x;
    col = blockIdx.x * blockDim.x + threadIdx.y;
    if (row < n && col < m) {
        b[row + col * n] = data[threadIdx.x][threadIdx.y];
    } 
}

__global__ void matmul(int *a, int *b, int *c, int p, int q, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int cVal = 0;
    if (row < p && col < r) {
        for (int k=0;k<q;k++) {
            cVal += a[row*q+k] * b[k*r+col];
        }
        c[row*r+col] = cVal;
    }
}

// matrix multiplication using shared memory
// __global__ void matmul(int *a, int *b, int *c, int p, int q, int r) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ int a_shared[32][32];
//     __shared__ int b_shared[32][32];

//     int cVal = 0;
//     for (int k=0;k < (32 + q - 1)/32; k++) {
//         if (k*32 + threadIdx.x < q && row < p) {
//             a_shared[threadIdx.y][threadIdx.x] = a[row*q+k*32+threadIdx.x];
//         }
//         else {
//             a_shared[threadIdx.y][threadIdx.x] = 0;
//         }
//         if (k*32 + threadIdx.y < q && col < r) {
//             b_shared[threadIdx.y][threadIdx.x] = b[(k*32+threadIdx.y)*r+col];
//         }
//         else {
//             b_shared[threadIdx.y][threadIdx.x] = 0;
//         }
//         __syncthreads();
//         for (int j=0; j<32; j++) {
//             cVal += a_shared[threadIdx.y][j] * b_shared[j][threadIdx.x];
//         }
//         __syncthreads();
//     }
//     if (row<p && col<r) {
//         c[((blockIdx.y * blockDim.y + threadIdx.y)*r)+(blockIdx.x*blockDim.x)+threadIdx.x] = cVal;
//     }
// }

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        exit(1);
    }

    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
        printf("Unable to open input file %s\n", argv[1]);
        exit(1);
    }
    int p, q, r, s;
    fscanf(input, "%d %d %d %d", &p, &q, &r, &s);
    // Allocate memory for matrices A, B, C, D, X
    int *A = (int *)malloc(p * q * sizeof(int));
    int *B = (int *)malloc(p * q * sizeof(int));
    int *C = (int *)malloc(r * p * sizeof(int));
    int *D = (int *)malloc(r * s * sizeof(int));
    int *X = (int *)malloc(q * s * sizeof(int));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++)
            fscanf(input, "%d", &A[i * q + j]);
    }
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++)
            fscanf(input, "%d", &B[i * q + j]);
    }
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < p; j++)
            fscanf(input, "%d", &C[i * p + j]);
    }
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < s; j++)
            fscanf(input, "%d", &D[i * s + j]);
    }
    fclose(input);

    // Allocate memory on GPU
    int *d_A, *d_B, *d_C, *d_D, *d_X;
    cudaMalloc(&d_A, p * q * sizeof(int));
    cudaMalloc(&d_B, p * q * sizeof(int));
    cudaMalloc(&d_C, r * p * sizeof(int));
    cudaMalloc(&d_D, r * s * sizeof(int));
    cudaMemcpy(d_A, A, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, r * p * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, r * s * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    // fins AT, BT, CT
    gridDim.x = (q + blockDim.x - 1) / blockDim.x;
    gridDim.y = (p + blockDim.y - 1) / blockDim.y;
    int *d_AT, *d_BT, *d_CT;
    cudaMalloc(&d_AT, q * p * sizeof(int));
    cudaMalloc(&d_BT, q * p * sizeof(int));
    cudaMalloc(&d_CT, p * r * sizeof(int));

    (p==q) ? transpose<<<gridDim, blockDim>>>(d_A, d_AT, p, q) : Transpose<<<gridDim, blockDim>>>(d_A, d_AT, p, q);
    (p==q) ? transpose<<<gridDim, blockDim>>>(d_B, d_BT, p, q) : Transpose<<<gridDim, blockDim>>>(d_B, d_BT, p, q);

    gridDim.x = (p + blockDim.x - 1) / blockDim.x;
    gridDim.y = (r + blockDim.y - 1) / blockDim.y;
    (p==r) ? transpose<<<gridDim, blockDim>>>(d_C, d_CT, p, r) : Transpose<<<gridDim, blockDim>>>(d_C, d_CT, r, p);

    // find ATBT = AT + BT
    int *d_ATBT;
    cudaMalloc(&d_ATBT, q * p * sizeof(int));
    gridDim.x = (q + blockDim.x - 1) / blockDim.x;
    gridDim.y = (p + blockDim.y - 1) / blockDim.y;
    matadd<<<gridDim, blockDim>>>(d_AT, d_BT, d_ATBT, q, p);

    // find ATBTCT = ATBT * CT
    gridDim.x = (q + blockDim.x - 1) / blockDim.x;
    gridDim.y = (r + blockDim.y - 1) / blockDim.y;
    int *d_ATBTCT;
    cudaMalloc(&d_ATBTCT, q * r * sizeof(int));
    matmul<<<gridDim, blockDim>>>(d_ATBT, d_CT, d_ATBTCT, q, p, r);

    // find X = ATBTCT * D
    gridDim.x = (q + blockDim.x - 1) / blockDim.x;
    gridDim.y = (s + blockDim.y - 1) / blockDim.y;
    cudaMalloc(&d_X, q * s * sizeof(int));
    matmul<<<gridDim, blockDim>>>(d_ATBTCT, d_D, d_X, q, r, s);

    cudaMemcpy(X, d_X, q * s * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *output = fopen(argv[2], "w");
    if (output == NULL) {
        printf("Unable to open output file %s\n", argv[2]);
        exit(1);
    }
    // fprintf(output, "%d %d\n", q, s);
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < s; j++)
            fprintf(output, "%d ", X[i * s + j]);
        fprintf(output, "\n");
    }
    fclose(output);

    // free 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_X);
    cudaFree(d_AT);
    cudaFree(d_BT);
    cudaFree(d_CT);
    cudaFree(d_ATBT);
    cudaFree(d_ATBTCT);

    free(A);
    free(B);
    free(C);
    free(D);
    free(X);

    return 0;
}