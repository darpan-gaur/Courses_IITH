/*
Name        :- Darpan Gaur
Roll No.    :- CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <bitset>
#include <map>

using namespace std;

// convert hex to binary
string hexToBin(string &temp) {
    string binStr = "";
    for (auto ch:temp) {
        if (isxdigit(ch)) {
            int hexVal;
            if (isdigit(ch)) hexVal = ch - '0';
            else hexVal = toupper(ch) - 'A' + 10;
            binStr += bitset<4>(hexVal).to_string();
        }
        else {
            cout << "Invalid hex character" << endl;
            exit(1);
        }
    }
    return binStr;
}

// take input from file, print the input 
// return input in binary format
void takeInput(vector<string> &input, char* fileName) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        cout << "Error opening file" << endl;
        exit(1);
    }

    char c;
    string temp = "";
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            cout << temp << "\n";
            input.push_back(hexToBin(temp));
            temp = "";
        } else {
            temp += c;
        }
    }
    if (temp != "") {
        input.push_back(hexToBin(temp));
    }
    fclose(fp);
}

// get opcode from binary string
string getOpcode(string &line) {
    string opcode = line.substr(25, 7);
    if (opcode == "0110011") return "R";
    if (opcode == "0010011") return "I";
    if (opcode == "0000011") return "IL";
    if (opcode == "0100011") return "S";
    if (opcode == "1100011") return "B";
    if (opcode == "1101111") return "J";
    if (opcode == "1100111") return "IJ";
    if (opcode == "0110111") return "U";
    if (opcode == "0010111") return "U";
    if (opcode == "1110011") return "IE";
    return "Invalid opcode";
}

// get funct3 from binary string
string get_funct3(string &line) {
    return line.substr(17, 3);
}

// get rs1 from binary string and return in decimal
string get_rs1(string &line) {
    string rs1 = line.substr(12, 5);
    int dec = 0;
    for (int i = 0; i < rs1.size(); i++) {
        dec += (rs1[i] - '0') * (1 << (rs1.size() - i - 1));
    }
    return to_string(dec);
}

// get rs2 from binary string and return in decimal
string get_rs2(string &line) {
    string rs2 = line.substr(7, 5);
    int dec = 0;
    for (int i = 0; i < rs2.size(); i++) {
        dec += (rs2[i] - '0') * (1 << (rs2.size() - i - 1));
    }
    return to_string(dec);
}

// get rd from binary string and return in decimal
string get_rd(string &line) {
    string rd = line.substr(20, 5);
    int dec = 0;
    for (int i = 0; i < rd.size(); i++) {
        dec += (rd[i] - '0') * (1 << (rd.size() - i - 1));
    }
    return to_string(dec);
}

// RType for R type instructions
// add, sub, sll, slt, sltu, xor, srl, sra, or, and
string RType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rs2 = get_rs2(line);
    string rd = get_rd(line);
    string funct7 = line.substr(0, 7);
    string instr = "";
    if (funct3=="000") {
        if (funct7=="0000000") instr = "add";
        else if (funct7=="0100000") instr = "sub";
        else return "Invalid instruction of R type";
    }
    else if (funct3=="001") instr = "sll";
    else if (funct3=="010") instr = "slt";
    else if (funct3=="011") instr = "sltu";
    else if (funct3=="100") instr = "xor";
    else if (funct3=="101") {
        if (funct7=="0000000") instr = "srl";
        else if (funct7=="0100000") instr = "sra";
        else return "Invalid instruction of R type";
    }
    else if (funct3=="110") instr = "or";
    else if (funct3=="111") instr = "and";
    else return "Invalid instruction of R type";
    return instr + " x" + rd + ", x" + rs1 + ", x" + rs2;
}

// IType for I type instructions
// addi, slti, sltiu, xori, ori, andi, slli, srli, srai
string IType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rd = get_rd(line);
    string imm = line.substr(0, 12);
    int dec = 0;    // signed decimal
    dec = - (imm[0] - '0') * (1 << 11);
    for (int i = 1; i < imm.size(); i++) {
        dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
    }
    string instr = "";
    if (funct3=="000") instr = "addi";
    else if (funct3=="010") instr = "slti";
    else if (funct3=="011") instr = "sltiu";
    else if (funct3=="100") instr = "xori";
    else if (funct3=="110") instr = "ori";
    else if (funct3=="111") instr = "andi";
    else if (funct3=="001" && imm.substr(0,6) == "000000" ) instr = "slli";
    else if (funct3=="101") {
        if (imm.substr(0, 6) == "000000") instr = "srli";
        else if (imm.substr(0, 6) == "010000") instr = "srai";
        else return "Invalid instruction of I type";
        imm = imm.substr(6, 6);
        dec = 0;    // signed decimal
        dec = - (imm[0] - '0') * (1 << 5);
        for (int i = 1; i < imm.size(); i++) {
            dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
        }
    }
    else return "Invalid instruction of I type";

    return instr + " x" + rd + ", x" + rs1 + ", " + to_string(dec);
}

// ILType for I type instructions
// lb, lh, lw, ld, lbu, lhu, lwu
string ILType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rd = get_rd(line);
    string imm = line.substr(0, 12);
    int dec = 0;    // signed decimal
    dec = - (imm[0] - '0') * (1 << 11);
    for (int i = 1; i < imm.size(); i++) {
        dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
    }
    string instr = "";
    if (funct3=="000") instr = "lb";
    else if (funct3=="001") instr = "lh";
    else if (funct3=="010") instr = "lw";
    else if (funct3=="011") instr = "ld";
    else if (funct3=="100") instr = "lbu";
    else if (funct3=="101") instr = "lhu";
    else if (funct3=="110") instr = "lwu";
    else return "Invalid instruction of IL type";

    return instr + " x" + rd + ", " + to_string(dec) + "(x" + rs1 + ")";
}

// SType for S type instructions
// sb, sh, sw, sd
string SType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rs2 = get_rs2(line);
    string imm;
    imm = line.substr(0, 7) + line.substr(20, 5);
    // convert binary to decimal
    int dec = 0;
    dec = - (imm[0] - '0') * (1 << 11);
    for (int i = 1; i < imm.size(); i++) {
        dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
    }
    string instr = "";
    if (funct3=="000") instr = "sb";
    else if (funct3=="001") instr = "sh";
    else if (funct3=="010") instr = "sw";
    else if (funct3=="011") instr = "sd";
    else return "Invalid instruction of S type";

    return instr + " x" + rs2 + ", " + to_string(dec) + "(x" + rs1 + ")";
}

// BType for B type instructions
// beq, bne, blt, bge, bltu, bgeu
string BType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rs2 = get_rs2(line);
    
    string instr = "";
    if (funct3=="000") instr = "beq";
    else if (funct3=="001") instr = "bne";
    else if (funct3=="100") instr = "blt";
    else if (funct3=="101") instr = "bge";
    else if (funct3=="110") instr = "bltu";
    else if (funct3=="111") instr = "bgeu";
    else return "Invalid instruction of B type";

    // return instr + " x" + rs1 + ", x" + rs2 + ", " + to_string(dec);
    return instr + " x" + rs1 + ", x" + rs2 + ", ";

}

// JType for J type instructions
// jal
string JType(string &line) {
    string rd = get_rd(line);
   
    string instr = "jal";
    // return instr + " x" + rd + ", " + to_string(dec);
    return instr + " x" + rd + ", ";
}

// IJType for I type instructions
// jalr
string IJType(string &line) {
    string funct3 = get_funct3(line);
    string rs1 = get_rs1(line);
    string rd = get_rd(line);
    string imm = line.substr(0, 12);
    int dec = 0;    // signed decimal
    dec = - (imm[0] - '0') * (1 << 11);
    for (int i = 1; i < imm.size(); i++) {
        dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
    }
    string instr = "";
    if (funct3=="000") instr = "jalr";
    else return "Invalid instruction of IJ type";

    return instr + " x" + rd + ", " + to_string(dec) + "(x" + rs1 + ")";
}

// UType for U type instructions
// lui, auipc
string UType(string &line) {
    string rd = get_rd(line);
    string imm = line.substr(0, 20);
    // convert binary immm to hexadecimal
    string hex = "";
    for (int i = 0; i < imm.size(); i+=4) {
        int dec = 0;
        for (int j = 0; j < 4; j++) {
            dec += (imm[i+j] - '0') * (1 << (3 - j));
        }
        if (dec < 10) hex += (dec + '0');
        else hex += (dec - 10 + 'A');
    }
    string instr = "";
    if (line.substr(25, 7)=="0110111") instr = "lui";
    else if (line.substr(25, 7)=="0010111") instr = "auipc";
    else return "Invalid instruction of U type";

    return instr + " x" + rd + ", 0x" + hex;
}

// print function for vector of strings
void print(vector<string> &input) {
    for (int i = 0; i < input.size(); i++) {
        cout << input[i] << endl;
    }
}

int main(int argc, char *argv[]) {

    // store input in vector of strings, in binary format
    vector<string> input;
    cout << "Using input file: " << argv[1] << "\n";
    takeInput(input, argv[1]);
    cout << "\n";

    // disassemble
    vector<string> output;
    map<int, string> label;
    int c=0;
    string textLabel = "L";
    for (int i=0;i<input.size();i++) {
        // get opcode
        string line = input[i];
        string opcode = getOpcode(line);
        if (opcode == "R") {
            output.push_back(RType(line));
        }
        else if (opcode == "I") {
            output.push_back(IType(line));
        }
        else if (opcode == "IL") {
            output.push_back(ILType(line));
        }
        else if (opcode == "S") {
            output.push_back(SType(line));
        }
        else if (opcode == "B") {
            // rename line number with lable name using map
            string imm;
            imm = line.substr(0, 1) + line.substr(24, 1) + line.substr(1, 6) + line.substr(20, 4) + "0";
            // convert binary to decimal
            int dec = 0;
            dec = - (imm[0] - '0') * (1 << 12);
            for (int i = 1; i < imm.size(); i++) {
                dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
            }
            if (label.find(dec+i*4) == label.end()) {
                label[dec+i*4] = textLabel + to_string(c);
                c++;
            }
            output.push_back(BType(line) + label[dec+i*4]);
        }
        else if (opcode == "J") {
            string imm;
            imm = line.substr(0, 1) + line.substr(12, 8) + line.substr(11, 1) + line.substr(1, 10) + "0";
            // convert binary to signed decimal
            int dec = 0;
            dec = - (imm[0] - '0') * (1 << 20);
            for (int i = 1; i < imm.size(); i++) {
                dec += (imm[i] - '0') * (1 << (imm.size() - i - 1));
            }
            if (label.find(dec+i*4) == label.end()) {
                label[dec+i*4] = textLabel + to_string(c);
                c++;
            }
            output.push_back(JType(line) + label[dec+i*4]);
        }
        else if (opcode == "IJ") {
            output.push_back(IJType(line));
        }
        else if (opcode == "U") {
            output.push_back(UType(line));
        }
        else {
            print(output);
            cout << "Invalid opcode\n" << endl;
            exit(1);
        }
        // add label in front of instruction if it exists
        if (label.find(i*4) != label.end()) {
            string temp = output.back();
            output.pop_back();
            output.push_back(label[i*4] + " : " + temp);
        }
        
    }

    // print output
    cout << "Disassembled code:\n";
    print(output);
    cout << "\n";
    return 0;
}