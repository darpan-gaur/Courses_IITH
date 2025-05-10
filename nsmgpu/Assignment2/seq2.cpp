#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>

using namespace std;

// X = (AT + BT ) CT D where XT is the transpose of X
/*
2.1Input
* Four integers p, q, r, s
* Matrix A of size p x q
* Matrix B of size p x q
* Matrix C of size r x p
* Matrix D of size r x s
2.2
Output
* Output is Matrix X of size q x s
*/

void print(int p, int q, int *M) {
    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            printf("%d ", M[i*q + j]);
        }
        printf("\n");
    }
}

void transpose(int p, int q, int *M, int *T) {
    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            T[j*p + i] = M[i*q + j];
        }
    }
}

void matmul(int p, int q, int r, int *A, int *B, int *C) {
    // A: pxq B: qxr C: pxr
    for (int i=0;i<p;i++) {
        for (int j=0;j<r;j++) {
            C[i*r + j] = 0;
            for (int k=0;k<q;k++) {
                C[i*r + j] += A[i*q + k] * B[k*r + j];
            }
        }
    }
}

void matadd(int p, int q, int *A, int *B, int *C) {
    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            C[i*q + j] = A[i*q + j] + B[i*q + j];
        }
    }
}

void solve(int p, int q, int r, int s, int *A, int *B, int *C, int *D, int *X) {
    // find AT, BT, CT
    int *AT = (int *)malloc(q * p * sizeof(int));
    int *BT = (int *)malloc(q * p * sizeof(int));
    int *CT = (int *)malloc(r * p * sizeof(int));
    transpose(p, q, A, AT);
    transpose(p, q, B, BT);
    transpose(r, p, C, CT);

    // find AT + BT
    int *mid = (int *)malloc(p * q * sizeof(int));
    matadd(q, p, AT, BT, mid);

    // find mid * CT
    int *mid2 = (int *)malloc(q * r * sizeof(int));
    matmul(q, p, r, mid, CT, mid2);

    // find mid2 * D
    matmul(q, r, s, mid2, D, X);
}

random_device rd;
mt19937 gen(rd());

int main() {

    uniform_int_distribution<> dim(2, 512);
    int p = dim(gen);
    int q = dim(gen);
    int r = dim(gen);
    int s = dim(gen);

    uniform_int_distribution<> val(-10, 10);

    // make A, B, C, D having random values between -10 to 10
    int *A = (int *)malloc(p * q * sizeof(int));
    int *B = (int *)malloc(p * q * sizeof(int));
    int *C = (int *)malloc(r * p * sizeof(int));
    int *D = (int *)malloc(r * s * sizeof(int));
    int *X = (int *)malloc(q * s * sizeof(int));

    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            A[i*q + j] = val(gen);
            B[i*q + j] = val(gen);
        }
    }
    for (int i=0;i<r;i++) {
        for (int j=0;j<p;j++) {
            C[i*p + j] = val(gen);
        }
    }
    for (int i=0;i<r;i++) {
        for (int j=0;j<s;j++) {
            D[i*s + j] = val(gen);
        }
    }

    // write to input.txt
    FILE *input = fopen("input.txt", "w");
    fprintf(input, "%d %d %d %d\n", p, q, r, s);
    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            fprintf(input, "%d ", A[i*q + j]);
        }
        fprintf(input, "\n");
    }
    for (int i=0;i<p;i++) {
        for (int j=0;j<q;j++) {
            fprintf(input, "%d ", B[i*q + j]);
        }
        fprintf(input, "\n");
    }
    for (int i=0;i<r;i++) {
        for (int j=0;j<p;j++) {
            fprintf(input, "%d ", C[i*p + j]);
        }
        fprintf(input, "\n");
    }
    for (int i=0;i<r;i++) {
        for (int j=0;j<s;j++) {
            fprintf(input, "%d ", D[i*s + j]);
        }
        fprintf(input, "\n");
    }
    fclose(input);

    solve(p, q, r, s, A, B, C, D, X);

    FILE *output = fopen("output.txt", "w");
    // fprintf(output, "%d %d\n", q, s);
    for (int i=0;i<q;i++) {
        for (int j=0;j<s;j++) {
            fprintf(output, "%d ", X[i*s + j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);

    free(A);
    free(B);
    free(C);
    free(D);
    free(X);

    return 0;
}