#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace sycl;

#define BLOCK_SIZE 16

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
    }
    
    queue q;
    q.get_device().get_info<info::device::name>();
//     cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    
    // take input from flie
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Cannot open input file %s\n", argv[1]);
        exit(1);
    }
    int m, k, n;
    fscanf(fp, "%d %d %d", &m, &k, &n);
    
    // USM allocation using malloc_shared
    long long *A = malloc_host<long long>(m*k, q);
    long long *B = malloc_host<long long>(k*n, q);
    long long *C = malloc_host<long long>(m*n, q);
    
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            fscanf(fp, "%lld", &A[i*k+j]);
        }
    }
    for (int i=0; i<k; i++) {
        for (int j=0; j<n; j++) {
            fscanf(fp, "%lld", &B[i*n+j]);
        }
    }
    fclose(fp);
    
    long long *d_A = malloc_device<long long>(m*k, q);
    long long *d_B = malloc_device<long long>(k*n, q);
    long long *d_C = malloc_device<long long>(m*n, q);
    
    q.memcpy(d_A, A, sizeof(long long)*m*k).wait(); 
    q.memcpy(d_B, B, sizeof(long long)*k*n).wait(); 
    
//     range<2> blockDim(BLOCK_SIZE, BLOCK_SIZE);
//     range<2> gridDim( (n + BLOCK_SIZE - 1) / BLOCK_SIZE , (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     nd_range<2> computeRange(gridDim, blockDim);
    int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    auto start = chrono::high_resolution_clock::now();
    q.submit([&](handler &h){
        local_accessor<long long, 2> sA(range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
        local_accessor<long long, 2> sB(range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
        
        h.parallel_for(nd_range<2>(range<2>(grid_cols * BLOCK_SIZE, grid_rows * BLOCK_SIZE), range<2>(BLOCK_SIZE, BLOCK_SIZE)), [=](nd_item<2> item){
//             int ty = item.get_group(1) * BLOCK_SIZE + item.get_local_id(1);
//             int tx = item.get_group(0) * BLOCK_SIZE + item.get_local_id(0);
            int row = item.get_group(1) * BLOCK_SIZE + item.get_local_id(1);
            int col = item.get_group(0) * BLOCK_SIZE + item.get_local_id(0);
                
            long long tmp=0;
            for (int k_pos=0;k_pos < (k+BLOCK_SIZE-1)/BLOCK_SIZE;k_pos++) {
                if (k_pos*BLOCK_SIZE + item.get_local_id(0) < k && row < m) {
                    sA[item.get_local_id(1)][item.get_local_id(0)] = d_A[row*k + k_pos*BLOCK_SIZE + item.get_local_id(0)]; 
                }
                else sA[item.get_local_id(1)][item.get_local_id(0)] = 0;
                
                if (k_pos*BLOCK_SIZE + item.get_local_id(1) < k && col < n) {
                    sB[item.get_local_id(1)][item.get_local_id(0)] = d_B[col + (k_pos*BLOCK_SIZE + item.get_local_id(1))*n]; 
                }
                else sB[item.get_local_id(1)][item.get_local_id(0)] = 0;
                
                group_barrier(item.get_group());
//                 h.barrier(sycl::access::fence_space::local_space);
//                  item.get_sub_group().barrier();
                
                for (int i=0;i<BLOCK_SIZE;i++) {
                    tmp += sA[item.get_local_id(1)][i] * sB[i][item.get_local_id(0)];
                }
                group_barrier(item.get_group());
//                 item.get_sub_group().barrier();
                
            }
            if (row<m && col<n) {
                d_C[row*n + col] = tmp;
            }
        });
        
    }).wait();
    auto end = chrono::high_resolution_clock::now();
    // get time in ms
    long long duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time: " << duration << " ms" << "\n";
    
    q.memcpy(C, d_C, sizeof(long long)*m*n).wait(); 
    

    
    fp = fopen(argv[2], "w");
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            fprintf(fp, "%lld ", C[i*n+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);
    
    free(A, q);
    free(B, q);
    free(C, q);
    
    return 0;
}
