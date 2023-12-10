/*
Name    : -     Darpan Gaur 
Roll No.: -     CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace std;

// take input from the input file and store as vector<string>
void takeInput(vector<string> &input, char* fielName) {
    FILE *fp = fopen(fielName, "r");
    if (fp == NULL) {
        cout << "Error opening file" << endl;
        exit(0);
    }
    char c;
    string temp = "";
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            input.push_back(temp);
            temp = "";
        } else {
            temp += c;
        }
    }
    if (temp != "") {
        input.push_back(temp);
    }
    fclose(fp);
}

// map for register with their number
map<string, int> reg = {
    {"x0", 0}, {"x1", 1}, {"x2", 2}, {"x3", 3}, {"x4", 4}, {"x5", 5}, {"x6", 6}, {"x7", 7}, {"x8", 8}, {"x9", 9}, {"x10", 10}, {"x11", 11}, {"x12", 12}, {"x13", 13}, {"x14", 14}, {"x15", 15}, {"x16", 16}, 
    {"x17", 17}, {"x18", 18}, {"x19", 19}, {"x20", 20}, {"x21", 21}, {"x22", 22}, {"x23", 23}, {"x24", 24}, {"x25", 25}, {"x26", 26}, {"x27", 27}, {"x28", 28}, {"x29", 29}, {"x30", 30}, {"x31", 31},
    {"zero", 0}, {"ra", 1}, {"sp", 2}, {"gp", 3}, {"tp", 4}, {"t0", 5}, {"t1", 6}, {"t2", 7}, {"s0", 8}, {"fp", 8}, {"s1", 9}, {"a0", 10}, {"a1", 11}, {"a2", 12}, {"a3", 13}, {"a4", 14}, {"a5", 15}, {"a6", 16}, 
    {"a7", 17}, {"s2", 18}, {"s3", 19}, {"s4", 20}, {"s5", 21}, {"s6", 22}, {"s7", 23}, {"s8", 24}, {"s9", 25}, {"s10", 26}, {"s11", 27}, {"t3", 28}, {"t4", 29}, {"t5", 30}, {"t6", 31}
};

// make map to check if R-type instruction (1) or load instruction (2) or store instruction (3) or I-type instruction (4) or nop (5)
map<string, int> insrtType = {
    {"add", 1}, {"sub", 1}, {"xor", 1}, {"or", 1}, {"and", 1}, {"sll", 1}, {"srl", 1}, {"sra", 1}, {"slt", 1}, {"sltu", 1},
    {"lb", 2}, {"lh", 2}, {"lw", 2}, {"ld", 2}, {"lbu", 2}, {"lhu", 2}, {"lwu", 2},
    {"sb", 3}, {"sh", 3}, {"sw", 3}, {"sd", 3},
    {"addi", 4}, {"xori", 4}, {"ori", 4}, {"andi", 4}, {"slli", 4}, {"srli", 4}, {"srai", 4}, {"slti", 4}, {"sltiu", 4},
    {"nop", 5}
};

// split the input, by comma, space, (, ) and store in vector of vector of strings
// also if register name is valid or not, or instruction is RType, load and store
void splitInput(vector<string> &input, vector<vector<string>> &output) {
    // split if ' ' or ',' or '(' or ')' is found
    for (int i = 0; i < input.size(); i++) {
        string temp = "";
        vector<string> tempVec;
        for (int j = 0; j < input[i].size(); j++) {
            if (input[i][j] == ' ' || input[i][j] == ',' || input[i][j] == '(' || input[i][j] == ')') {
                if (temp != "") {
                    tempVec.push_back(temp);
                }
                temp = "";
            } else {
                temp += input[i][j];
            }
        }
        if (temp != "") {
            tempVec.push_back(temp);
        }
        if (tempVec.size() != 4) continue;
        if (insrtType.find(tempVec[0]) == insrtType.end()) {
            cout << "Instruction is other from R-type, load and store" << endl;
            exit(0);
        }
        if ((reg.find(tempVec[1]) == reg.end())) {
            cout << "Invalid register name" << endl;
            exit(0);
        }
        if ((insrtType[tempVec[0]] == 1 || insrtType[tempVec[0]]==4) && (reg.find(tempVec[2]) == reg.end())) {
            cout << "Invalid register name" << endl;
            exit(0);
        }
        if ((insrtType[tempVec[0]] != 4) && (reg.find(tempVec[3]) == reg.end())) {
            cout << "Invalid register name" << endl;
            exit(0);
        }
        output.push_back(tempVec);
    }
}

// adds nop when there is a data hazard
void addNop(vector<vector<string>> &instructions, int idx) {
    int i = idx;
    string r = instructions[i][1];
    vector<string> nop = {"nop"};
    if (reg[r] == 0) return;
    bool flag = false;
    if ((i+1<instructions.size()) && (instructions[i+1].size()==4)) {
        if ((insrtType[instructions[i+1][0]]!=4) && reg[instructions[i+1][3]] == reg[r]) flag = true;
        if ((insrtType[instructions[i+1][0]]==4 || insrtType[instructions[i+1][0]]==1) && (reg[instructions[i+1][2]] == reg[r])) flag = true;

        if (flag) {
            instructions.insert(instructions.begin()+i+1, nop);
        }
    }
}

// print the output
void print(vector<vector<string>> &output, vector<string> &input) {
    int i=0,j=0;
    for (j=0;j<output.size();j++) {
        if (output[j][0] != "nop") {
            cout << input[i++] << "\n";
        }
        else {
            cout << output[j][0] << "\n";
        }
    }
    cout << "Total: " << output.size()+4 << " cycles" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Invalid number of arguments" << endl;
        exit(0);
    }

    // storing the input file in a vector of strings
    vector<string> input;
    takeInput(input, argv[1]);

    // split the input
    vector<vector<string>> instructions;
    splitInput(input, instructions);

    for (int i=0;i<instructions.size();i++) {
        if (insrtType[instructions[i][0]] == 1) {
            // R-type instruction
            continue;
        }else if (insrtType[instructions[i][0]] == 2) {
            // load instruction
            addNop(instructions, i);
        }else if (insrtType[instructions[i][0]] == 3) {
            // store instruction
            continue;
        }else if (insrtType[instructions[i][0]] == 4) {
            // I type instruction
            continue;
        }
        else if (insrtType[instructions[i][0]] == 5) {
            // nop instruction
            continue;
        }
        else {
            cout << "Instruction is other from R-type, load and store" << endl;
        }
    }

    cout << "Assuming that data forwarding without hazard detection is implemented: -\n";
    print(instructions, input);

    return 0;
}