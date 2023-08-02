#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include<sys/time.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <pthread.h>
#include <omp.h>

// Global Variables
int *arr;       // Array to store sudoku
int *chk;       // (3*n) to keep record which clo,row,grid is valid & checked by which thread
int n,k;        // n:- size of sudoku base, k:- no. of threads

// Functions returns 1 if xth row of sudoku is valid else 0
int chkRow(int x) {
    int i,ans=1;
    int ck[n];
    for (i=0;i<n;i++) ck[i] = 0;
    for (i=x*n;i<(x+1)*n;i++) {
        if (arr[i]<1 && arr[i]>n) ans= 0;
        if (ck[arr[i]-1]==0) ck[arr[i]-1] = 1;
        else {
            ans= 0;
        }
    }return ans;
}

// Functions returns 1 if xth col of sudoku is valid else 0
int chkCol(int x) {
    int i,ans=1;
    int ck[n];
    for (i=0;i<n;i++) ck[i] = 0;
    for (i=x;i<n*n;i = i+n){
        if (arr[i]<1 && arr[i]>n) ans= 0;
        if (ck[arr[i]-1]==0) ck[arr[i]-1] = 1;
        else {
            ans= 0;
        }
    }return ans;
}

// Functions returns 1 if xth grid of sudoku is valid else 0
int chkGrid(int x) {
    int i,j=sqrt(n),k,c,ans=1;
    int ck[n];
    for (i=0;i<n;i++) ck[i] = 0;
    c = (x/j)*j*n;
    c += (x%j)*j;
    for (k=0;k<j;k++) {
        for (i=c;i<c+j;i++) {
            if (arr[n*k+i]<1 && arr[n*k+i]>n) ans= 0;
            if (ck[arr[n*k+i]-1]==0) ck[arr[n*k+i]-1] = 1;
            else {
                ans= 0;
            }
        }
    }return ans;
}


int main()
{
    int i,j,mod;

    // Open input file
    FILE* in_file = fopen("input.txt","r"); // read only

    if (in_file == NULL) {
        printf("Error! Could not open input file\n");
    }

    // take input k and n from input file
    fscanf(in_file,"%d %d",&k,&n);
   
    // dynamically allocate arr (store suduku (n*n)) & chk (3*n)  
    arr = (int *)malloc((n*n)*sizeof(int));
    chk = (int *)malloc((3*n)*sizeof(int));

    for (i=0;i<n*n;i++) {
        fscanf(in_file,"%d",&arr[i]);
    }
   
    double time_taken = 0.0;
    clock_t begin = clock();
    struct timeval start, end;
    // start timer.
    gettimeofday(&start, NULL);
        
        #pragma omp parallel for num_threads(k)
        for (i=0;i<3*n;i++) {
            if (i<n) {
                if (chkRow(i)) chk[i] = (omp_get_thread_num()+1);
                else chk[i] = -(omp_get_thread_num()+1);

            }  
            else if (i<2*n) {
                if (chkCol(i-n)) chk[i] = (omp_get_thread_num()+1);
                else chk[i] = -(omp_get_thread_num()+1);
            }
            else {
                if (chkGrid(i-2*n)) chk[i] = (omp_get_thread_num()+1);
                else chk[i] = -(omp_get_thread_num()+1);
            }
        }

    int ans=1;
    for (i=0;i<3*n;i++) {
        if (chk[i]<0) ans=0;
    }

    gettimeofday(&end, NULL);
    time_taken = (end.tv_sec - start.tv_sec) * 1.0e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec));
    printf("%lf\n",time_taken);

    // if (ans) printf("Valid Sudoku\n");
    // else printf("Invalid Sudoku\n");

    // Create the output file by parent process
    FILE* out_file = fopen("OutMain.txt","w");
    for (int i=0;i<k;i++){
        for (j=i;j<3*n;j = j+k) {
            if (j<n) {
                if (chk[i]>0) fprintf(out_file, "Thread %d checks row %d and is valid\n",chk[i],j+1);
                else fprintf(out_file, "Thread %d checks row %d and is invalid\n",chk[i],j+1);
            }
            else if (j<2*n) {
                if (chk[i]>0) fprintf(out_file, "Thread %d checks column %d and is valid\n",chk[i],j%n+1);
                else fprintf(out_file, "Thread %d checks column %d and is invalid\n",chk[i],j%n+1);
            }
            else {
                if (chk[i]>0) fprintf(out_file, "Thread %d checks grid %d and is valid\n",chk[i],j%n+1);
                else fprintf(out_file, "Thread %d checks grid %d and is invalid\n",chk[i],j%n+1);
            }
        }
    }

    if (ans) fprintf(out_file,"Sudoku is %s\n","valid");
    else fprintf(out_file,"Sudoku is %s\n","invalid");

    fprintf(out_file,"The total time taken is %lf\n",time_taken);
    
    fclose(out_file);       // Close OutMain.txt
    fclose(in_file);        // Close input.txt
    free(arr);              // Free the memory allocated to global arr
    free(chk);              // Free the memory allocated to global chk
    return 0;
}