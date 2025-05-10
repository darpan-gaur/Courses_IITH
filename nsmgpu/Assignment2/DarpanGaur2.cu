#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


// matrix multiplication kernel: C = A * B using shared memory
__global__ void matMultiply(int *a, int *b, int *c, int p, int q, int r) {
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    __shared__ int sA[32][32];
    __shared__ int sB[32][32];

    int val = 0;

    for (int k=0;k<(q+31)/32;k++) {
        if (k*32 + threadIdx.x < q && row < p) {
            sA[threadIdx.y][threadIdx.x] = a[row*q + k*32 + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }
        if (k*32 + threadIdx.y < q && col < r) {
            sB[threadIdx.y][threadIdx.x] = b[(k*32 + threadIdx.y)*r + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i=0;i<32;i++) {
            val += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < p && col < r) {
        c[((blockIdx.y * blockDim.y + threadIdx.y) * r) + (blockIdx.x * blockDim.x)+ threadIdx.x] = val;
    }
}

// matrix transpose kernel: B = A^T using shared memory
__global__ void matTranspose(int *a, int *b, int m, int n) {
    __shared__ int data[32][33];
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;

    if ((row < m) && (col < n)) {
        data[threadIdx.y][threadIdx.x] = b[col * m + row];
    }
    __syncthreads();
    row = blockIdx.y * 32 + threadIdx.x;
    col = blockIdx.x * 32 + threadIdx.y;
    if ((row < n) && (col < m)) {
        a[col * n + row] = data[threadIdx.x][threadIdx.y];
    }
}

// matrix addition kernel: A = A + B
__global__ void matAdd(int *a, int *b, int m, int n) {
    int row = blockDim.x * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n && col < m) {
        a[col * n + row] += b[col * n + row];
    }
}

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

    // caluate AT
    int *h_AT = (int *)malloc(q * p * sizeof(int));
    int *d_A, *d_AT;
    cudaMalloc(&d_A, p * q * sizeof(int));
    cudaMalloc(&d_AT, q * p * sizeof(int));

    cudaMemcpy(d_A, A, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AT, h_AT, q * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32, 1);

    dim3 grid0Dim((q + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y, 1);

    matTranspose<<<grid0Dim, blockDim>>>(d_AT, d_A, q, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_AT, d_AT, q * p * sizeof(int), cudaMemcpyDeviceToHost);

    // calculate BT
    int *h_BT = (int *)malloc(q * p * sizeof(int));
    int *d_B, *d_BT;
    cudaMalloc(&d_B, p * q * sizeof(int));
    cudaMalloc(&d_BT, q * p * sizeof(int));

    cudaMemcpy(d_B, B, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT, h_BT, q * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid1Dim((q + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y, 1);

    matTranspose<<<grid1Dim, blockDim>>>(d_BT, d_B, q, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_BT, d_BT, q * p * sizeof(int), cudaMemcpyDeviceToHost);

    // calculate CT
    int *h_CT = (int *)malloc(p * r * sizeof(int));
    int *d_C, *d_CT;
    cudaMalloc(&d_C, r * p * sizeof(int));
    cudaMalloc(&d_CT, p * r * sizeof(int));

    cudaMemcpy(d_C, C, r * p * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CT, h_CT, p * r * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid2Dim((p + blockDim.x - 1) / blockDim.x, (r + blockDim.y - 1) / blockDim.y, 1);

    matTranspose<<<grid2Dim, blockDim>>>(d_CT, d_C, p, r);
    cudaDeviceSynchronize();

    cudaMemcpy(h_CT, d_CT, p * r * sizeof(int), cudaMemcpyDeviceToHost);

    // calculate A = AT + BT
    int *h_A = (int *)malloc(q * p * sizeof(int));

    dim3 grid3Dim((q + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y, 1);

    matAdd<<<grid3Dim, blockDim>>>(d_AT, d_BT, q, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_AT, q * p * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate E = CT * D
    int *h_E = (int *)malloc(p * s * sizeof(int));
    int *d_E, *d_D;

    cudaMalloc(&d_E, p * s * sizeof(int));
    cudaMalloc(&d_D, r * s * sizeof(int));

    cudaMemcpy(d_D, D, r * s * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_E, h_E, p * s * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CT, h_CT, p * r * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid4Dim((s + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y, 1);

    matMultiply<<<grid4Dim, blockDim>>>(d_CT, d_D, d_E, p, r, s);
    cudaDeviceSynchronize();

    cudaMemcpy(h_E, d_E, p * s * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate X = A * E
    int *d_X;
    cudaMalloc(&d_X, q * s * sizeof(int));
    cudaMemcpy(d_A, h_A, q * p * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, p * s * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid5Dim((s + blockDim.x - 1) / blockDim.x, (q + blockDim.y - 1) / blockDim.y, 1);

    matMultiply<<<grid5Dim, blockDim>>>(d_A, d_E, d_X, q, p, s);

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
    cudaFree(d_AT);
    cudaFree(d_B);
    cudaFree(d_BT);
    cudaFree(d_C);
    cudaFree(d_CT);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_X);

    free(A);
    free(B);
    free(C);
    free(D);
    free(X);

    return 0;
}