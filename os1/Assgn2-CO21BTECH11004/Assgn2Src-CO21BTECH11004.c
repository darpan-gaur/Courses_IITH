/*
Name 	:- Darpan Gaur
Roll No :- CO21BTECH11004
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <pthread.h>

int *arr;       // Creating a global array

// IsPerfectNUmber takes a number as a parameter and return 1 if it is a perfect number else return 0
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


// For each thread allocate numbers to them and checks wheather those numbers are perfect or not
// Takes 'mod' as parameter and checks for number wihch gives remainder 'mod' when divided by k
void *ThreadFunc(int mod){
    int n=arr[0];   // initializing n
    int k=arr[1];   // initializing k

    char CP_OFN[20];

    // CP_OFN stores the name of output file for this child process
    snprintf(CP_OFN,20,"OutFile%d",mod);

    // create output file with name CP_OFN
    FILE* out_file = fopen(CP_OFN,"w");

    for (int i=1;i<=n;i++){
        // Checks for number given to this thread (i which gives remainer 'mod' when divided by k)
        if (i%k==mod) {
            if (IsPerfectNumber(i)) {
                fprintf(out_file, "%d: %s\n",i, "Is a perfect number");
                arr[i+1] = 1;   // update the global array
            }
            else {
                fprintf(out_file, "%d: %s\n",i, "Not a perfect number");
            }
        }
    }
    fclose(out_file);   // close the output file
    pthread_exit(0);
}


int main()
{
    int n,k;

    // Open input file
    FILE* in_file = fopen("input.txt","r"); // read only

    if (in_file == NULL) {
        printf("Error! Could not open file\n");
    }

    // take input n and k from input file
    fscanf(in_file,"%d %d",&n,&k);

    // printf("%d %d\n",n,k);
    
    arr = (int *)malloc((n+2)*sizeof(int));
    arr[0] = n;
    arr[1] = k;

    for (int i=2;i<n+2;i++) {
        arr[i] = 0;
    }

    // for (int i=2;i<n+2;i++) {
    //     // if (arr[i-1])
    //     printf("%d :- %d ; ",i-1,arr[i]);
    // }printf("\n");

    pthread_t tid[k];       /* the thread identifier */

    for (int i=0;i<k;i++){
        /* create the thread */
        pthread_create(&tid[i], NULL, ThreadFunc, i);   // NULL as second parameter sets defalut thread attributes
        
    }

    // for creation of k threads
    for (int i=0;i<k;i++){
        pthread_join(tid[i],NULL);
    }


    // for (int i=2;i<n+2;i++) {
    //     // if (arr[i-1])
    //     printf("%d :- %d ; ",i-1,arr[i]);
    // }printf("\n");

    arr[0] = 0;
    arr[1] = 0;
    // Create the output file by parent process
    FILE* out_file = fopen("OutMain.txt","w");
    for (int i=0;i<k;i++){
        fprintf(out_file,"CP %d:- ",i);
        for (int j=2+i;j<=n+1;j+=k){
            if (arr[j-1]) {
                fprintf(out_file,"%d",j-2);
            }
        }fprintf(out_file,"\n");
    }
    fclose(out_file);       // Close OutMain.txt
    fclose(in_file);        // Close input.txt
    free(arr);              // Free the memory allocated to global array
    return 0;
} 
