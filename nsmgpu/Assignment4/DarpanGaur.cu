/*
Darpan Gaur
darpangaur2003@gmail.com
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct myFunc {
    __host__ __device__ 
    int operator () (const int &x, const int &y) const {
        return (x-y < 0 ? 0 : x-y);
    }
};

void solve(int n, int k, int *studentId, int *arrivalTime, int *laptopUsage) {
    // sort arrival time in ascending order, also sort student id accordingly
    thrust::device_vector<int> arrivalTime_d(arrivalTime, arrivalTime + n);
    thrust::device_vector<int> studentId_d(studentId, studentId + n);
    thrust::sort_by_key(arrivalTime_d.begin(), arrivalTime_d.end(), studentId_d.begin());

    // print studenId_d and arrivalTime_d
    // thrust::copy(studentId_d.begin(), studentId_d.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
    // thrust::copy(arrivalTime_d.begin(), arrivalTime_d.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    thrust::device_vector<int> laptop(k, 0);
    
    // initialize laptopUsage to 0
    thrust::copy(laptop.begin(), laptop.end(), laptopUsage);

    int currStdTime = arrivalTime_d[0];
    for (int i=0 ;i<n ;i++) {
        if (arrivalTime_d[i] != currStdTime) {
            thrust::device_vector<int> modify(k, arrivalTime_d[i] - currStdTime);
            thrust::transform(laptop.begin(), laptop.end(), modify.begin(), laptop.begin(), myFunc());
            currStdTime = arrivalTime_d[i];
        }
        int freeLaptop = thrust::min_element(laptop.begin(), laptop.end()) - laptop.begin();
        laptop[freeLaptop] += 1;
        laptopUsage[freeLaptop] += 1;
    }

}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }
    
    int n, k;
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL){
        printf("Error opening input file\n");
        return 1;
    }
    fscanf(fp, "%d %d", &n, &k);

    int *studentId = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++){
        fscanf(fp, "%d", &studentId[i]);
    }
    int *arrivalTime = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++){
        fscanf(fp, "%d", &arrivalTime[i]);
    }
    fclose(fp);

    int *laptopUsage = (int *)malloc(sizeof(int) * k);

    // Come up with a schedule of 0..N-1 students to use the laptops 0..K-1
    solve(n, k, studentId, arrivalTime, laptopUsage);

    // output result
    fp = fopen(argv[2], "w");
    if (fp == NULL){
        printf("Error opening output file\n");
        return 1;
    }
    for (int i = 0; i < k; i++){
        fprintf(fp, "%d ", laptopUsage[i]);
    }
    fclose(fp);

    // free memory
    free(studentId);
    free(arrivalTime);
    free(laptopUsage);

    return 0;
}