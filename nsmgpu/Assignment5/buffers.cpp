#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
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
    
    vector<long long> A(m*k);
    vector<long long> B(k*n);
    vector<long long> C(m*n);
    
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
    
//     range<2> blockDim(BLOCK_SIZE, BLOCK_SIZE);
//     range<2> gridDim( (n + BLOCK_SIZE - 1) / BLOCK_SIZE , (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     nd_range<2> computeRange(gridDim, blockDim);
    
    // create buffers
    buffer a(A);
    buffer b(B);
    buffer c(C);
    
    auto start = chrono::high_resolution_clock::now();
    q.submit([&](handler &h){
        // Create accessors to copy buffer to device
        accessor dA(a, h, read_only);
        accessor dB(b, h, read_only);
        accessor dC(c, h, write_only);
        
        int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        h.parallel_for(nd_range<2>(range<2>(grid_cols * BLOCK_SIZE, grid_rows * BLOCK_SIZE), range<2>(BLOCK_SIZE, BLOCK_SIZE)), [=](nd_item<2> item){
            int ty = item.get_group(1) * BLOCK_SIZE + item.get_local_id(1);
            int tx = item.get_group(0) * BLOCK_SIZE + item.get_local_id(0);
            int n_pos = ty;
            while (n_pos < n) {
                int m_pos = tx;
                while (m_pos < m) {
                    long long tmp = 0;
                    for (int k_pos=0; k_pos<k; k_pos++) {
                        tmp += dA[m_pos*k+k_pos] * dB[k_pos*n+n_pos];
                    }
                    dC[m_pos*n+n_pos] = tmp;
    //                 C[m_pos*n+n_pos] = m_pos*n+n_pos;

                    m_pos += item.get_group_range(0) * BLOCK_SIZE;
                }
                n_pos += item.get_group_range(1) * BLOCK_SIZE;
            }
        });
        
    }).wait();
    auto end = chrono::high_resolution_clock::now();
    // get time in ms
    long long duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time: " << duration << " ms" << "\n";
    
    host_accessor hC(c, read_only);
    
    fp = fopen(argv[2], "w");
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            fprintf(fp, "%lld ", C[i*n+j]);
//             fprintf(fp, "%lld ", C[i][j]);
            
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    
    return 0;
    
}
