#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <pthread.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <random>

using namespace std;

// Create Gobal Library
double *x;       
double *y;
int *arr;
int *points;
int *sq_points;
int n,k;

void *ThreadFunc(void* k_i){
    int i,j=0;
    long ki = (long)k_i,circle_points=0,square_points=0;
    double xi,yi;
    // srand(time(NULL)+ki);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (i=0;i<n;i++){
        if (i%k == ki) {
            xi = dist(mt);
            yi = dist(mt);
            x[i]= xi;
            y[i]=yi;;
            if ((xi*xi)+(yi*yi)<=1) {
                arr[i]=1;
                circle_points++;
            }else {
                arr[i]=0;
            }square_points++;
        }
    }
    points[ki] = circle_points;
    sq_points[ki] = square_points;
    pthread_exit(0); 
}

int main() {
    // Open input file
    FILE* in_file = fopen("inp.txt","r"); // read only
    int i,ans=0,j;
    if (in_file == NULL) {
        printf("Error! Could not open file\n");
    }

    // take input n and k from input file
    fscanf(in_file,"%d %d",&n,&k);
    long k_i[k];
    x = (double *)malloc((n)*sizeof(double));
    y = (double *)malloc((n)*sizeof(double));
    arr = (int *)malloc((n)*sizeof(int));
    points = (int *)malloc((k)*sizeof(int));
    sq_points = (int *)malloc((k)*sizeof(int));
    
    
    auto start = chrono::high_resolution_clock::now();
    pthread_t tid[k];       /* the thread identifier */

    for (i=0;i<k;i++){
        k_i[i] = i;
        /* create the thread */
        pthread_create(&tid[i], NULL, ThreadFunc, (void*)k_i[i]);   // NULL as second parameter sets defalut thread attributes
        // pthread_join(tid[i],NULL);
        
    }
    

    // for creation of k threads
    for (i=0;i<k;i++){
        pthread_join(tid[i],NULL);
    }


    

    int TotalPoints=0,GoodPoints=0;

    for (i=0;i<k;i++) {
        GoodPoints+=points[i];
        TotalPoints+=sq_points[i];
        // cout << arr[i] << " ";
    }


    // cout << "\n";
    double pi = 4*((double)GoodPoints/TotalPoints) ;

    auto end = chrono::high_resolution_clock::now();
    double time_taken =chrono::duration_cast<chrono::microseconds>(end - start).count();

    // time_taken *= 1e-9; 
    // cout << "Time taken by program is : " << fixed << time_taken;    cout << " sec" << endl;

    // cout << pi << "\n";
    // cout << TotalPoints << " " << GoodPoints << "\n";

    FILE* out_file = fopen("OutMain.txt","w");
    fprintf(out_file, "Time: %lf micro seconds\n\n", time_taken);
    fprintf(out_file, "Value Computed: %lf\n\n",pi);
    fprintf(out_file, "Log: \n\n");

    for (i=0;i<k;i++) {
        fprintf(out_file, "Thread%d: %d , %d \n",i+1,sq_points[i],points[i]);
        fprintf(out_file, "Points inside the square: ");
        for (j=0;j<n;j++) {
            if (j%k == i) {
                fprintf(out_file, "(%lf , %lf), ",x[j],y[j]);
            }
        }
        fprintf(out_file, "\nPoints inside the circle: ");
        for (j=0;j<n;j++) {
            if (j%k == i && arr[j]) {
                fprintf(out_file, "(%lf , %lf), ",x[j],y[j]);
            }
        }
        fprintf(out_file, "\n\n");
    }
    

    fclose(in_file);
    fclose(out_file);
    free(x);
    free(y);
    free(arr);
    free(points);
    free(sq_points);
}