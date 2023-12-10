#include <iostream>
#include <vector>
#include <cmath>
#include <random>
// #include <string>    

using namespace std;

int myClk=1;

void configCache(vector<string>& input, char* fileName) {
    FILE *fp = fopen(fileName, "r");
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

// convert hex to binary
void hexToBinary(vector<string>& input, vector<string>& output) {
    for (int i = 0; i < input.size(); i++) {
        // hex is starting from 3rd character of input string
        string hex = input[i].substr(5);
        string binary = "";
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
        // push first 3 bit of input[i] with binary to output
        output.push_back(input[i].substr(0, 3) + binary);
    }
}

// convert binary to hex
string binaryToHex(string& binary) {
    string hex = "0x";
    // check is binary is multiple of 4, if not pad with 0s
    if (binary.length() % 4 != 0) {
        int pad = 4 - (binary.length() % 4);
        for (int i = 0; i < pad; i++) {
            binary = "0" + binary;
        }
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

class Desc {
public:
    int set;
    int set_bits;
    int line;
    int line_bits;
    int offset_bits;
    int policy;
    bool writePolicy;

    Desc() {};
    Desc (int cacheSize, int blockSize, int associativity, string replacementPolicy, string writePolicy) {
        if (associativity) this->set = cacheSize / (blockSize * associativity);
        else set = 1;
        this->set_bits = log2(set);
        if (associativity) this->line = associativity;
        else this->line = cacheSize / blockSize;
        this->line_bits = log2(line);
        this->offset_bits = log2(blockSize);
        if (replacementPolicy == "FIFO") {
            this->policy = 0;
        } else if (replacementPolicy == "LRU") {
            this->policy = 1;
        } else if (replacementPolicy == "RANDOM") {
            this->policy = 2;
        } else {
            cout << "Invalid replacement policy\n";
            exit(0);
        }
        if (writePolicy == "WB") {
            this->writePolicy = true;
        } else if (writePolicy == "WT") {
            this->writePolicy = false;
        } else {
            cout << "Invalid write policy\n";
            exit(0);
        }
    }
};

class Cache {
public:
    vector<vector<string>> tag;
    vector<vector<bool>> valid;
    // vector<vector<bool>> dirty;
    vector<vector<int>> cFIFO;  // counter to implement fifo
    vector<vector<int>> cLRU;   // counter to implement lru

    Cache() {};
    Cache (int set, int line) {
        this->tag.resize(set);
        this->valid.resize(set);
        // this->dirty.resize(set);
        this->cFIFO.resize(set);
        this->cLRU.resize(set);
        for (int i = 0; i < set; i++) {
            this->tag[i].resize(line);
            this->valid[i].resize(line, false);
            // this->dirty[i].resize(line, false);
            this->cFIFO[i].resize(line, 0);
            this->cLRU[i].resize(line, 0);
        }
    }
};

Desc cacheDesc;
Cache cache;

// get line number of tag in set
int getLineNumber(int index, string& tag) {
    for (int line=0;line<cacheDesc.line;line++) {
        if (cache.tag[index][line] == tag) {
            return line;
        }
    }
    return -1;
}

// check if set is full
bool isSetFull(int index) {
    for (int line=0;line<cacheDesc.line;line++) {
        if (cache.valid[index][line] == false) {
            return false;
        }
    }
    return true;
}

// find line to replace by FIFO
int findLineByFIFO(int index) {
    int mVal = cache.cFIFO[index][0];
    int mLine = 0;
    for (int line=1;line<cacheDesc.line;line++) {
        if (cache.cFIFO[index][line] < mVal) {
            mVal = cache.cFIFO[index][line];
            mLine = line;
        }
    }
    return mLine;
}

// get line with minimum LRU counter
int findLineByLRU(int index) {
    int mVal = cache.cLRU[index][0];
    int mLine = 0;
    for (int line=1;line<cacheDesc.line;line++) {
        if (cache.cLRU[index][line] < mVal) {
            mVal = cache.cLRU[index][line];
            mLine = line;
        }
    }
    return mLine;
}

int findLineByRANDOM(int index) {
    // use unfiform distribution to generate random integer between 0 and cacheDesc.line
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, cacheDesc.line-1);
    return dis(gen);
}

int getEmptyLine(int index) {
    for (int line=0;line<cacheDesc.line;line++) {
        if (cache.valid[index][line] == false) {
            return line;
        }
    }
    return -1;
}

void read_access(string& address) {
    
    string offset = address.substr(address.length() - cacheDesc.offset_bits);                                           // offset is last (cacheDesc.offset) bits
    string setBin = address.substr(address.length() - cacheDesc.offset_bits - cacheDesc.set_bits, cacheDesc.set_bits);  // set is next (cacheDesc.set) bits
    // convert string set to int by binary to decimal conversion
    int set=0;
    if (cacheDesc.set_bits) set = stoi(address.substr(address.length() - cacheDesc.offset_bits - cacheDesc.set_bits, cacheDesc.set_bits) , 0, 2);
    string tag = address.substr(3, address.length() - cacheDesc.offset_bits - cacheDesc.set_bits - 3);  // tag is remaining bits, excluding first 3 bits
    
    cout << "Set: " << binaryToHex(setBin) << " ";

    int line = getLineNumber(set, tag);
    // // Read Miss
    if (line == -1) {
        cout << "Miss ";
        int insrtLine;
        // set id full
        if (isSetFull(set)) {
            // check for FIFO
            if (cacheDesc.policy == 0) insrtLine = findLineByFIFO(set);
            // check for LRU
            else if (cacheDesc.policy == 1) insrtLine = findLineByLRU(set);
            // check for RANDOM
            else if (cacheDesc.policy == 2) insrtLine = findLineByRANDOM(set);
        }
        // set is not full
        else {
            insrtLine = getEmptyLine(set);
        }
        cache.tag[set][insrtLine] = tag;
        cache.valid[set][insrtLine] = true;
        // cache.dirty[set][insrtLine] = false;
        cache.cFIFO[set][insrtLine] = myClk;
        cache.cLRU[set][insrtLine] = myClk;
    }
    // Read Hit
    else {
        cout << "Hit ";
        cache.tag[set][line] = tag;
        cache.cLRU[set][line] = myClk;
    }
    cout << "Tag: " << binaryToHex(tag) << "\n";
}

// write through no write allocate
void writeThrough(string& address) {
    string offset = address.substr(address.length() - cacheDesc.offset_bits);                       // offset is last (cacheDesc.offset) bits
    string setBin = address.substr(address.length() - cacheDesc.offset_bits - cacheDesc.set_bits, cacheDesc.set_bits);  // set is next (cacheDesc.set) bits
    // convert string set to int by binary to decimal conversion
    int set = stoi(address.substr(address.length() - cacheDesc.offset_bits - cacheDesc.set_bits, cacheDesc.set_bits) , 0, 2);   // tag is remaining bits, excluding first 3 bits
    if (cacheDesc.set_bits) set = stoi(address.substr(address.length() - cacheDesc.offset_bits - cacheDesc.set_bits, cacheDesc.set_bits) , 0, 2);
    
    string tag = address.substr(3, address.length() - cacheDesc.offset_bits - cacheDesc.set_bits - 3);
    
    cout << "set: " << binaryToHex(setBin) << " ";

    int line = getLineNumber(set, tag);
    // // Read Miss
    if (line == -1) {
        cout << "Miss ";
    }
    // Read Hit
    else {
        cout << "Hit ";
        cache.tag[set][line] = tag;
        cache.cLRU[set][line] = myClk;
    }
    cout << "tag: " << binaryToHex(tag) << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Invalid number of arguments\n";
        exit(0);
    }

    // take cahce config input from config.txt
    vector<string> cacheConfig; // input of form <SizeOfCache> <BlockSize> <Associativity> <ReplacementPolicy (FIFO/LRU/RANDOM)> <WritePolicy (WB/WT)>
    configCache(cacheConfig, argv[1]);

    // storing description of cache
    cacheDesc = Desc(stoi(cacheConfig[0]), stoi(cacheConfig[1]), stoi(cacheConfig[2]), cacheConfig[3], cacheConfig[4]);

    // creating cache
    cache = Cache(cacheDesc.set, cacheDesc.line);

    // take cahche access input from access.txt
    vector<string> cacheAccess; // input of form <Read/Write> <Address>
    configCache(cacheAccess, argv[2]); 

    vector<string> cacheAccessBinary;
    hexToBinary(cacheAccess, cacheAccessBinary);

    for (int i=0;i<cacheAccessBinary.size();i++) {
        // cout << "Address: " << cacheAccessBinary[i].substr(3) << " ";
        // cout << cacheAccessBinary[i] << "\n";
        string address = cacheAccessBinary[i].substr(3);
        cout << "Address: " << binaryToHex(address) << " ";
        if (cacheAccessBinary[i][0]=='R' || cacheDesc.writePolicy == true) {
            read_access(cacheAccessBinary[i]);
        }
        else if (cacheAccessBinary[i][0]=='W') {
            writeThrough(cacheAccessBinary[i]);
        }
        else {
            cout << "Invalid operation\n";
            exit(0);
        }
        myClk++;
    }
    return 0;
}