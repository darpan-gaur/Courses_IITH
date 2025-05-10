/*
Name    :- Darpan Gaur
Roll No :- CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

// Function to read input from a file
vector<int> inputHandler(string filename) {
    vector<int> v;
    ifstream file(filename);
    int x;
    while(file >> x){
        v.push_back(x);
    }
    return v;
}

// Function to merge two sorted arrays and count inversions
long long inversionCountMerge(vector<int> &A, int s, int mid, int e) {
    int nl = mid - s + 1;
    int nr = e - mid;
    vector<int> L(nl), R(nr);
    for (int i = 0; i < nl; i++) {
        L[i] = A[s + i];
    }
    for (int i = 0; i < nr; i++) {
        R[i] = A[mid + 1 + i];
    }
    int i = 0, j = 0, k = s;
    long long count = 0;

    while (i < nl && j < nr) {
        if (L[i] <= R[j]) {
            A[k] = L[i];
            i++;
        } else {
            A[k] = R[j];
            j++;
            count += nl - i;
        }
        k++;
    }

    while (i < nl) {
        A[k] = L[i];
        i++;
        k++;
    }

    while (j < nr) {
        A[k] = R[j];
        j++;
        k++;
    }

    return count;
}

// Function to count inversions in an array
long long inversionCount(vector<int> &A, int s, int e) {
    if (s >= e) {
        return 0;
    }
    long long count = 0;
    int mid = s + (e - s) / 2;

    count += inversionCount(A, s, mid);
    count += inversionCount(A, mid + 1, e);
    count += inversionCountMerge(A, s, mid, e);

    return count;
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        cout << "Usage: " << argv[0] << " <input csv file path>\n";
        return 1;
    }
    vector<int> A = inputHandler(argv[1]);
    
    cout << "Inversion Count: " << inversionCount(A, 0, A.size() - 1) << endl;

    return 0;
}