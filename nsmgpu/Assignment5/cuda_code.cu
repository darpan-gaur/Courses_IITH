// compile command: nvcc -lnvToolsExt naive_matmul.cu

#include <stdio.h>
#include <thread>
#include <cuda_runtime.h>
#include <iostream>
// #include "nvToolsExt.h"

#define BLOCK_SIZE 16

template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
							T * __restrict__ dest,        //inout: pointer to C matrix data
							const T * __restrict__ left,  //in: pointer to A matrix data
							const T * __restrict__ right) //in: pointer to B matrix data
{
    size_t ty = blockIdx.y*blockDim.y + threadIdx.y; 
	size_t tx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t n_pos = ty; 
	while(n_pos < n){
		size_t m_pos = tx; 
		while(m_pos < m) {
			T tmp = static_cast<T>(0.0);
			for(size_t k_pos = 0; k_pos < k; ++k_pos)
			{
				tmp += left[m_pos*k + k_pos] * right[k_pos*n + n_pos];
			}
			dest[m_pos*n + n_pos] += tmp;
			m_pos += gridDim.x*blockDim.x; 
		}
		n_pos += gridDim.y*blockDim.y; 
	}
	return;
}

int main() {

    // Sample workloads.
	// constexpr int m = 256, n = 256, k = 256;
	// constexpr int m = 1024, n = 1024, k = 256;

    // This is always the default workload unless stated otherwise in the questions.
    // To change it simply change the value of m, n & k variables.
	constexpr int m = 1024, n = 1024, k = 1024;

    int *h_a, *h_b, *h_c;

	h_a = (int*)malloc(m*k*sizeof(int));
	h_b = (int*)malloc(k*n*sizeof(int));
	h_c = (int*)malloc(m*n*sizeof(int));

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i==0)
                h_a[i * k + j] = 1;
            else
                h_a[i * k + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j==0)
                h_b[i * n + j] = 1;
            else
                h_b[i * n + j] = rand() % 1024;
        }
    }

    // Allocate memory on the device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*k);
    cudaMalloc((void **) &d_b, sizeof(int)*k*n);
    cudaError_t err = cudaMalloc((void **) &d_c, sizeof(int)*m*n);

	if( err != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return -1;
	}

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*k*n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // DO NOT EDIT CODE BELOW THIS POINT.

    // Launch kernel 
	// nvtxRangePushA("launch kernel");
    auto start = std::chrono::high_resolution_clock::now();
	auto runs = 1000;
	for (int ij=0; ij<runs; ij++)
		gpu_gemm_nn<int><<<dimGrid, dimBlock>>>(m, n, k, d_c, d_a, d_b);

    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*n, cudaMemcpyDeviceToHost);
    // cudaThreadSynchronize();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

	// complete marker
	// nvtxRangePop(); 

	std::cout<< "Check " << (h_c[0] == runs * k) << "/1" << std::endl;


	std::chrono::duration<double, std::milli> elapsed = end-start;
    std::cout << "Naive matmul " << elapsed.count() / runs << " ms\n";
	return 0;
}
