#include <bits/stdc++.h>
#include <fstream>
using namespace std;
#include "cuda.h"
#define BLOCK_SIZE 16

ofstream outfile; //the handle for printing the output

__global__ void tile_matrix_multiply(long int* A, long int* B, long int* C, long int height, long int width, long int common_len)
{
    __shared__ long int tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ long int tileB[BLOCK_SIZE][BLOCK_SIZE];
    long int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    long int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    long int value = 0;
    long int num_tiles = (long int)ceil((double)common_len / BLOCK_SIZE);
    for (long int i=0; i<num_tiles ; i++)
    {
        // Assign the right values to the shared memory matrix.
        // Beware that border tiles might not have values for all the
        // elements in the tile matrix
        // Changed the algorithm now to remove thread divergence. So don't have to worry
        // about border values. height width always divisible by BLOCK_SIZE
        tileA[threadIdx.y][threadIdx.x] = A[row * common_len + i * BLOCK_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * width + col];

        __syncthreads();

        for (long int k=0; k<BLOCK_SIZE; k++)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    C[row * width + col] = value;
}

// Transpose Kernel using shared memoryand tiling
__global__ void transpose(long int* I, long int * O, long int height, long int width)
{
    __shared__ long int tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    long int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    long int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = I[row * width + col];
    __syncthreads();

    row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    O[row * height + col] = tile[threadIdx.x][threadIdx.y];
}

__global__ void add(long int* A, long int* B, long int* result, long int height, long int width)
{
    long int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    long int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    long int id = row * width + col;
    result[id] = A[id] + B[id];
}

// Printing the output to the file
 void printMatrix(long int *arr, long int rows, long int cols, long int rows_pad, long int cols_pad, char* filename) {
    // Since the matrix output is padded, I am printing only the non padded matrix
    // by taking in both padded values of rows and columns as well
	outfile.open(filename);
	for(long int i = 0; i < rows_pad; i++) {
		for(long int j = 0; j < cols_pad; j++) {
            if(i<rows && j<cols-1)
			    outfile<<arr[i * cols_pad + j]<<" ";
            else if(i<rows && j == cols - 1)
                outfile<<arr[i * cols_pad + j];
		}
        if(i>=rows)
            break;
		outfile<<"\n";
	}
	outfile.close();
}


// Getting the input
void get_input(long int* A, long int p, long int q, long int p_pad, long int q_pad)
{
    for(long int i=0; i<p_pad; i++)
        for(long int j=0; j<q_pad; j++)
            {
                if (i<p && j<q)
                    cin>>A[i*q_pad + j];
                else
                    A[i*q_pad + j]=0;
            }
}

long int get_padded(long int m, long int block_size)
{
    return ceil((double)m/block_size) * block_size;
}

int main()
{
    long int p,q,r,s;
    long int *A, *B, *C, *D, *result, *dtemp1, *dtemp2, *dtemp3, *dresult, *dA, *dB, *dC, *dD;
    cin >> p >> q >> r >> s;
    long int p_pad, q_pad, r_pad, s_pad;
    p_pad = get_padded(p, BLOCK_SIZE);
    q_pad = get_padded(q, BLOCK_SIZE);
    r_pad = get_padded(r, BLOCK_SIZE);
    s_pad = get_padded(s, BLOCK_SIZE);

    A = new long int[p_pad*q_pad];
    B = new long int[p_pad*q_pad];
    C = new long int[r_pad*p_pad];
    D = new long int[r_pad*s_pad];
    result = new long int[q_pad*s_pad];

    cudaMalloc(&dA, p_pad * q_pad * sizeof(long int));
    cudaMalloc(&dB, p_pad * q_pad * sizeof(long int));
    cudaMalloc(&dC, r_pad * p_pad * sizeof(long int));
    cudaMalloc(&dD, r_pad * s_pad * sizeof(long int));
    cudaMalloc(&dtemp1, p_pad * q_pad * sizeof(long int));
    cudaMalloc(&dtemp2, r_pad * q_pad * sizeof(long int));
    cudaMalloc(&dtemp3, q_pad * r_pad * sizeof(long int));
    cudaMalloc(&dresult, q_pad * s_pad * sizeof(long int));



    get_input(A, p,q, p_pad, q_pad);
    get_input(B, p,q, p_pad, q_pad);
    get_input(C, r,p, r_pad, p_pad);
    get_input(D, r,s, r_pad, s_pad);


    cudaMemcpy(dA, A, p_pad * q_pad* sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, p_pad * q_pad* sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, r_pad * p_pad* sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(dD, D, r_pad * s_pad* sizeof(long int), cudaMemcpyHostToDevice);

    /// First Add kernel (A+B)
    long int blockDimx = ceil((double)q_pad/BLOCK_SIZE);
    long int blockDimy = ceil((double)p_pad/BLOCK_SIZE);
    dim3 grid1(blockDimx , blockDimy);
    dim3 block1(BLOCK_SIZE, BLOCK_SIZE);
    add<<<grid1, block1>>>(dA, dB, dtemp1, p_pad, q_pad);

    /// Multiply kernel C(A+B)
    blockDimx = ceil((double)q_pad/BLOCK_SIZE);
    blockDimy = ceil((double)r_pad/BLOCK_SIZE);
    dim3 grid2(blockDimx , blockDimy);
    dim3 block2(BLOCK_SIZE, BLOCK_SIZE);
    tile_matrix_multiply<<<grid2, block2>>>(dC, dtemp1, dtemp2, r_pad, q_pad, p_pad);

    /// Transpose (C(A+B)).T = (AT + BT)CT
    blockDimx = ceil((double)q_pad/BLOCK_SIZE);
    blockDimy = ceil((double)r_pad/BLOCK_SIZE);
    dim3 grid3(blockDimx , blockDimy);
    dim3 block3(BLOCK_SIZE, BLOCK_SIZE);
    transpose<<<grid3, block3>>>(dtemp2, dtemp3, r_pad, q_pad);

    /// Multiply with D = (AT + BT).CT.D
    blockDimx = ceil((double)s_pad/BLOCK_SIZE);
    blockDimy = ceil((double)q_pad/BLOCK_SIZE);
    dim3 grid4(blockDimx , blockDimy);
    dim3 block4(BLOCK_SIZE, BLOCK_SIZE);
    tile_matrix_multiply<<<grid4, block4>>>(dtemp3, dD, dresult, q_pad, s_pad, r_pad);

    cudaMemcpy(result, dresult, q_pad * s_pad* sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(result, q, s, q_pad, s_pad, "output.txt");
    return 0;
}
