#include <stdio.h>
#include <stdlib.h>

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

void solve(int p, int q, int r, int s, int *A, int *B, int *C, int *D, int *X) {
    // mid = AT + BT 
    int *mid = (int *)malloc(p * q * sizeof(int));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            mid[j*p + i] = A[i*q + j] + B[i*q + j];
        }
    }
    // print(q, p, mid);

    // mid2 = mid * CT
    int *mid2 = (int *)malloc(q * r * sizeof(int));
    // mid: qxp CT: pxr
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < r; j++) {
            mid2[i*r + j] = 0;
            for (int k = 0; k < p; k++) {
                mid2[i*r + j] += mid[i*p + k] * C[j*p + k];
            }
        }
    }
    // print(q, r, mid2);
    // X = mid2 * D
    // mid2: qxr D: rxs
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < s; j++) {
            X[i*s + j] = 0;
            for (int k = 0; k < r; k++) {
                X[i*s + j] += mid2[i*r + k] * D[k*s + j];
            }
        }
    }
    // print(q, s, X);
    free(mid);
    free(mid2);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }      
    FILE *input = fopen(argv[1], "r");
    int p, q, r, s;
    fscanf(input, "%d %d %d %d", &p, &q, &r, &s);
    int *A = (int *)malloc(p * q * sizeof(int));
    int *B = (int *)malloc(p * q * sizeof(int));
    int *C = (int *)malloc(r * p * sizeof(int));
    int *D = (int *)malloc(r * s * sizeof(int));
    int *X = (int *)malloc(q * s * sizeof(int));
    for (int i = 0; i < p * q; i++) {
        fscanf(input, "%d", &A[i]);
    }
    for (int i = 0; i < p * q; i++) {
        fscanf(input, "%d", &B[i]);
    }
    for (int i = 0; i < r * p; i++) {
        fscanf(input, "%d", &C[i]);
    }
    for (int i = 0; i < r * s; i++) {
        fscanf(input, "%d", &D[i]);
    }
    fclose(input);

    solve(p, q, r, s, A, B, C, D, X);

    FILE *output = fopen(argv[2], "w");
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