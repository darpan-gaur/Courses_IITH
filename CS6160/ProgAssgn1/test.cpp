#include <bits/stdc++.h>

using namespace std;

int main() {
    // take 12 ciphertext (many time pad) input form input file and store in vector
    ifstream input("streamciphertexts.txt");
    vector<string> cipher;
    string temp;
    while (getline(input, temp)) {
        cipher.push_back(temp);
    }
    input.close();

    // print all ciphertext
    // for (int i = 0; i < cipher.size(); i++) {
    //     cout << cipher[i] << endl;
    // }

    
    return 0;
}
