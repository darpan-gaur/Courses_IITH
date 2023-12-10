#include <iostream>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#define ll long long

using namespace std;

void takeInput(vector<string>& input) {
    FILE *fp = fopen("input.txt", "r");
    if (fp == NULL) {
        cout << "Error opening file\n";
        exit(0);
    }
    // read file line by line and store in vector
    char c;
    string line = "";
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            input.push_back(line);
            line = "";
        } else {
            line += c;
        }
    }
    if (line != "") {
        input.push_back(line);
    }
    fclose(fp);
}

string hexToBin(string& hex) {
    string binary="";
    for (int j = 0; j < hex.length(); j++) {
        if (hex[j] == '0') binary += "0000";
        else if (hex[j] == '1') binary += "0001";
        else if (hex[j] == '2') binary += "0010";
        else if (hex[j] == '3') binary += "0011";
        else if (hex[j] == '4') binary += "0100";
        else if (hex[j] == '5') binary += "0101";
        else if (hex[j] == '6') binary += "0110";
        else if (hex[j] == '7') binary += "0111";
        else if (hex[j] == '8') binary += "1000";
        else if (hex[j] == '9') binary += "1001";
        else if (hex[j] == 'A' || hex[j] == 'a') binary += "1010";
        else if (hex[j] == 'B' || hex[j] == 'b') binary += "1011";
        else if (hex[j] == 'C' || hex[j] == 'c') binary += "1100";
        else if (hex[j] == 'D' || hex[j] == 'd') binary += "1101";
        else if (hex[j] == 'E' || hex[j] == 'e') binary += "1110";
        else if (hex[j] == 'F' || hex[j] == 'f') binary += "1111";
    }
    return binary;
}

string binaryToHex(string& binary) {
    string hex = "";
    // check is binary is multiple of 4, if not pad with 0s
    // if (binary.length() % 4 != 0) {
    //     int pad = 4 - (binary.length() % 4);
    //     for (int i = 0; i < pad; i++) {
    //         binary = "0" + binary;
    //     }
    // }
    while (binary.size() < 32) {
        binary = "0" + binary;
    }
    for (int i = 0; i < binary.length(); i += 4) {
        string temp = binary.substr(i, 4);
        if (temp == "0000") hex += "0";
        else if (temp == "0001") hex += "1";
        else if (temp == "0010") hex += "2";
        else if (temp == "0011") hex += "3";
        else if (temp == "0100") hex += "4";
        else if (temp == "0101") hex += "5";
        else if (temp == "0110") hex += "6";
        else if (temp == "0111") hex += "7";
        else if (temp == "1000") hex += "8";
        else if (temp == "1001") hex += "9";
        else if (temp == "1010") hex += "A";
        else if (temp == "1011") hex += "B";
        else if (temp == "1100") hex += "C";
        else if (temp == "1101") hex += "D";
        else if (temp == "1110") hex += "E";
        else if (temp == "1111") hex += "F";
    }
    return hex;
}

ll binToDecimal(string& input) {
    ll a = 1,res=0;
    int n=input.size(), i;
    for (i=n-1;i>=0;i--) {
        if (input[i]=='1') {
            res += a;
        }
        a = a*2;
    }
    return res;
}

string decimalToBinary(ll input) {
    string res="";
    while (input > 0) {
        if (input%2) res = "1" + res;
        else res = "0" + res;
        input = input/2;
    }
    return res;
}

double getFraction(string& input) {
    double res=0.0;
    double a = 0.5;
    for (int i=9;i<input.size();i++) {
        if (input[i]=='1') {
            res += a;
        }
        a = a/2;
    }
    return res;
}

void test(vector<string>& input) {
    for (int i = 0; i < input.size(); i++) {
        // hex is starting from 3rd character of input string
        string hex = input[i].substr(0, 8);
        cout << hex << " " << hexToBin(hex) << "\n";
        hex = hexToBin(hex);
        string hex2 = input[i].substr(10, 8);
        cout << hex2 << " " << hexToBin(hex2) << "\n";
        hex2 = hexToBin(hex2);
        // cout << hex.size() << " " << hex2.size() << "\n";
        cout << binToDecimal(hex) << "\n";
        cout << binToDecimal(hex2) << "\n";
        cout << getFraction(hex) << " " << getFraction(hex2) << "\n";
        // ll res = binToDecimal(hex) + binToDecimal(hex2);
        // cout << res << "\n";
    }
}

void seperateInput(vector<string>& input, vector<pair<ll,ll>>& output) {
    for (int i=0;i<input.size();i++) {
        string op1 = input[i].substr(0, 8);
        op1 = hexToBin(op1);
        // cout << op1 << "\n";
        string op2 = input[i].substr(10, 8);
        op2 = hexToBin(op2);
        // cout << op2 << "\n";
        output.push_back({binToDecimal(op1), binToDecimal(op2)});
    }
}

// 0 bit from starting
ll getSign(ll op) {
    return (op & 0x80000000) >> 31;
}

ll getExponent(ll op) {
    return (op & 0x7f800000) >> 23;
}

ll getMantisa(ll op) {
    return (op & 0x7fffff);
}


ll myAdd(ll op1, ll op2) {
    ll res=0, resSign, resExponent, resS;
    // // if either of operand is zero
    if (op1==0) return op2;
    if (op2==0) return op1;
    cout << getSign(op1) << " " << getSign(op2) << "\n";
    cout << getExponent(op1) << " " << getExponent(op2) << "\n";
    cout << getMantisa(op1) << " " << getMantisa(op2) << "\n";
    ll op1Sign = getSign(op1), op2Sign = getSign(op2);
    ll op1Exponent = getExponent(op1), op2Exponent = getExponent(op2);
    ll op1Mantisa = getMantisa(op1), op2Mantisa = getMantisa(op2);

    
    // // if exponent is zero then 0+fraction, else 1+fraction
    ll op1SS=op1Mantisa, op2SS=op2Mantisa;
    if (op1Exponent != 0) op1SS = op1SS | (1<<23);
    if (op2Exponent != 0) op2SS = op2SS | (1<<23);
    // cout << op1S << " " << op2S << "\n";

    // // check if exponent is 0
    if (op1Exponent == 0) op1Exponent = 1;
    if (op2Exponent == 0) op2Exponent = 1;

    // // check for NaN infinity

    // align binary points
    if (op1Exponent >= op2Exponent) {
        ll shift = op1Exponent - op2Exponent;
        if (shift>31) {
            op2SS = (op2SS >> 31);
        }
        else {
            op2SS = (op2SS >> shift);
        }
        resExponent = op1Exponent;
    }
    else {
        ll shift = op2Exponent - op1Exponent;
        if (shift>31) {
            op1SS = (op1SS >> 31);
        }
        else {
            op1SS = (op1SS >> shift);
        }
        resExponent = op2Exponent;
    }
    cout << resExponent << "\n";

    // // add significants
    if (op1Sign == op2Sign) {
        resS = op1SS + op2SS;
        resSign = op1Sign;   
    }
    else {
        if (op1SS > op2SS) {
            resS = op1SS - op2SS;
            resSign = op1Sign;
        }
        else if (op2SS > op1SS) {
            resS = op2SS - op1SS;
            resSign = op2Sign;
        }
        else {
            resS = op1SS-op2SS;
            resSign = 0;
        }
    }
    // cout << "resS:- " <<  decimalToBinary(resS) << "\n";

    int i;
    for (i=31;i>0 && (resS>>i)==0; i--) {;}
    while (i>23) {
        resS>>1;
        i--;
    }

    res = (resSign << 31) | (resExponent<<23) | (resS & 0x07fffff );
    return res;
}

void print(vector<string>& input){
    for (int i=0;i<input.size();i++) {
        cout << input[i] << "\n";
    }
}

int main() {
    vector<string> input;
    takeInput(input);
    print(input);

    vector<pair<ll,ll>> ll_input;
    seperateInput(input, ll_input);
    // for (int i=0;i<ll_input.size();i++) {
    //     cout << ll_input[i].first << " " << ll_input[i].second << "\n";
    // }
    FILE *outFile = fopen("output.txt", "w");
    test(input);
    for (int i=0;i<ll_input.size();i++) {
        ll res;
        res = myAdd(ll_input[i].first, ll_input[i].second);
        string binRes = decimalToBinary(res);
        string hexRes = binaryToHex(binRes);
        // cout << hexRes << "\n";
        fprintf(outFile, "%s\n", hexRes.c_str());
    }

    return 0;
}