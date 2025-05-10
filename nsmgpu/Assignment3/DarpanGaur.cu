//to run it
// /usr/local/cuda-10.2/bin/nvcc PageRank.cu -arch=sm_70 -rdc=true
//using global synchrnization if necessary

#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cmath>
#include<algorithm>
#include<cuda.h>
#include<cooperative_groups.h>
#include"helper.hpp"

// namespace cg = cooperative_groups;
// using namespace cooperative_groups;  

__device__ float diff =1.0f;
 __global__  void Compute_PR_Kernel(int * gpu_rev_OA, int * gpu_OA, int * gpu_srcList , float * gpu_node_pr , int V, int E , float beta, float delta, int maxIter) 
{
    //write your kernel here
    int Pi = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_PR_Pi = 0.0f;
    diff = 0.0f;
    if (Pi < V) {
      for (int i=gpu_rev_OA[Pi]; i<gpu_rev_OA[Pi+1]; i++) {
        int Pj = gpu_srcList[i];
        temp_PR_Pi += gpu_node_pr[Pj]/(gpu_OA[Pj+1]-gpu_OA[Pj]);
      }
    }
    __syncthreads();  
    if (Pi < V) {
      float PR_Pi = (1.0-delta)/V + delta*temp_PR_Pi;
      float diff_Pi = fabs(PR_Pi - gpu_node_pr[Pi]);
      // gpu_node_pr[Pi] = PR_Pi;
      atomicExch(&gpu_node_pr[Pi], PR_Pi);
      atomicAdd(&diff, diff_Pi);
    }
}


void Compute_PR(int * rev_OA, int * OA, int * cpu_srcList , float * node_pr , int V, int E)
{
  
  int    *gpu_rev_OA;
  int    *gpu_srcList;
  int    * gpu_OA;
  float  *gpu_node_pr;
  
  //write cudamalloc here
  cudaMalloc(&gpu_node_pr, sizeof(float) * (V));
  cudaMalloc(&gpu_rev_OA, sizeof(int) * (V+1));
  cudaMalloc(&gpu_OA, sizeof(int) * (V+1));
  cudaMalloc(&gpu_srcList, sizeof(int) * (E));
  
  unsigned int block_size;
	unsigned int num_blocks;
 
   for(int i=0; i< V; i++)
     {
         node_pr[i]= 1.0/V;
     }
   
  
  if(V <= 1024)
	{
		block_size = V;
		num_blocks = 1;
	}
	else
	{
		block_size = 1024;
		num_blocks = ceil(((float)V) / block_size);
			
	}
 
  //write cudaMemcpy here
  cudaMemcpy(gpu_node_pr, node_pr, sizeof(float) * (V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_rev_OA, rev_OA, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_OA, OA, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_srcList, cpu_srcList, sizeof(int) * (E), cudaMemcpyHostToDevice);
  
  float beta = 0.0001;
  float delta = 0.85;
  int maxIter = 100;
  
  int iterCount=0;
  float diff_check;
 
  do
  {
  
  //kernel call
  Compute_PR_Kernel<<<num_blocks , block_size>>>(gpu_rev_OA, gpu_OA, gpu_srcList, gpu_node_pr , V,E, beta, delta, maxIter);
  cudaDeviceSynchronize();
    
  //copy the diff value into diff_check
  cudaMemcpyFromSymbol(&diff_check, diff, sizeof(float));
  // printf("Iteration: %d done\n", iterCount);
  iterCount=iterCount+1;
  }while ((diff_check>beta) &&(iterCount < maxIter));
  
  // printf("Number of iterations: %d\n", iterCount);
  
  cudaMemcpy(node_pr,gpu_node_pr , sizeof(float) * (V), cudaMemcpyDeviceToHost);
 
  
  //output
  char *outputfilename = "outputN.txt";
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");

  for (int i = 0; i < V; i++)
  {
    fprintf(outputfilepointer, "%d  %0.9lf\n", i, node_pr[i]);
  }
 }

 int main(int argc , char ** argv)
{

  //graph parse using heper file
  graph G(argv[1]);
  G.parseGraph();
  
  int V = G.num_nodes();
  int E = G.num_edges();
  
  // offset array , reverse offset array and edge list
  float* node_pr;
  int *rev_OA;
  int *OA;
  int *cpu_srcList;
  
  
  //allocation of memory on CPU
  node_pr = (float *)malloc( (V)*sizeof(float));
  rev_OA = (int *)malloc( (V+1)*sizeof(int));
  OA = (int *)malloc( (V+1)*sizeof(int));
  cpu_srcList = (int *)malloc( (E)*sizeof(int));
  
  //reverse offset array for CSR
  for(int i=0; i<= V; i++) {
    int temp = G.rev_indexofNodes[i];
    rev_OA[i] = temp;
  }
  
  printf("\n");
  
   for(int i=0; i< E; i++) {
    int temp = G.srcList[i];
    cpu_srcList[i] = temp;
  }
  
 
   printf("\n");
  
   //offset array for CSR
   for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }
  
  //function call 
  Compute_PR(rev_OA, OA, cpu_srcList , node_pr , V, E);

}