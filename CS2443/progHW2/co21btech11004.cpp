/*
Name    :- Darpan Gaur
Roll No :- CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

// function to read input from a file
string inputHandler(string filename) {
    string s;
    ifstream file(filename);
    if (file.is_open()) {
        // read all the lines from the file
        string line;
        while (getline(file, line)) {
            s += line; // append each line to the string
        }
        file.close(); // close the file
    } else {
        cout << "Unable to open file";
    }
    return s; // return the string
}

int editDistance(string s1, string s2) {
    int l1 = s1.length(), l2 = s2.length();
    vector<vector<int>> dp(l1 + 1, vector<int>(l2 + 1, 0));

    for (int i = 0; i <= l1; i++) {
        for (int j = 0; j <= l2; j++) {
            if (i == 0) {
                dp[i][j] = j; // If first string is empty
            } else if (j == 0) {
                dp[i][j] = i; // If second string is empty
            } else if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // No operation needed
            } else {
                dp[i][j] = 1 + min({dp[i - 1][j],    // Deletion
                                   dp[i][j - 1],    // Insertion
                                   dp[i - 1][j - 1]}); // Substitution
            }
        }
    }
    return dp[l1][l2]; // Return the edit distance
}

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file1> <input_file2>" << endl;
        return 1;
    }
    string s1 = inputHandler(argv[1]);
    string s2 = inputHandler(argv[2]);

    // Remove spaces from the strings
    s1.erase(remove(s1.begin(), s1.end(), ' '), s1.end());
    s2.erase(remove(s2.begin(), s2.end(), ' '), s2.end());

    // Remove '\n' from the strings
    s1.erase(remove(s1.begin(), s1.end(), '\n'), s1.end());
    s2.erase(remove(s2.begin(), s2.end(), '\n'), s2.end());
    
    cout << "Edit Distance: " << editDistance(s1, s2) << endl;

    return 0;
}