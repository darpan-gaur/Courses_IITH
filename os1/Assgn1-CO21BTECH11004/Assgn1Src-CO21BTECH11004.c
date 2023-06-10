#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

int IsPerfectNumber(long long n){
    long long i,sum=0;
    for (i=1;i<n;i++){
        if (n%i==0) {
            sum += i;
        }
    }
    if (sum == n) return 1;
    else return 0;
}


int main()
{
    int n,k;

    // Open input file
    FILE* in_file = fopen("input.txt","r"); // read only

    // take input n and k from input file
    fscanf(in_file,"%d %d",&n,&k);

    // printf("%d %d\n",n,k);
    
    pid_t pid;

    // creates a shared array of size n for communication between parent and child processes 
    int *arr = mmap(NULL,n*sizeof(int),PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,0,0);

    // initializes all array elements to 0.
    for (int i=0;i<n;i++) arr[i] = 0;

    // for (int i=1;i<=n;i++) {
    //     printf("%d :- %d , ",i,arr[i-1]);
    // }printf("\n");

    char CP_OFN[20];

    // for creation of k child processes
    for (int i=0;i<k;i++){
        // fork
        pid = fork();

        //if child process
        if (pid==0){
            // CP_OFN stores the name of output file for this child process
            snprintf(CP_OFN,20,"OutFile%d",i);

            // create output file with name CP_OFN
            FILE* out_file = fopen(CP_OFN,"w");


            for (int j=1;j<=n;j++){
                // Checks for number given to this child (number for which mod k == i)
                if (j%k==i) {
                    if (IsPerfectNumber(j)) {
                        fprintf(out_file, "%d: %s\n",j, "Is a perfect number");
                        arr[j-1] = 1;   // update the shared memory
                    }else {
                        fprintf(out_file, "%d: %s\n",j, "Not a perfect number");
                    }
                }
            }
            fclose(out_file);   // close the output file
            exit(1);            // child terminates
        }
    }
    for (int i=0;i<k;i++) wait(NULL);   // parent execute wait() call for k child processes.
        
    // for (int i=1;i<=n;i++) {
    //     printf("%d :- %d , ",i,arr[i-1]);
    // }printf("\n");

    // Created the output file by parent process
    FILE* out_file = fopen("OutMain.txt","w");
    for (int i=0;i<k;i++){
        fprintf(out_file,"P%d :- ",i);
        for (int j=i;j<=n;j+=k){
            if (arr[j-1]) {
                fprintf(out_file,"%d",j);
            }
        }fprintf(out_file,"\n");
    }

    int a = munmap(arr,n*sizeof(int));  // unmap the array created by mmap
    fclose(out_file);       // Close OutMain.txt
    fclose(in_file);        // Close input.txt
    return 0;
} 