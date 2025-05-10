/*
Name    :- Darpan Gaur
Roll No :- CO21BTECH11004
*/
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <queue>

using namespace std;

// shortest path from source to destination

vector<int> shortestPath(int source, int destination, const map<int, vector<pair<int, float>>>& graph) {
    float INF = 1e9;
    map<int, float> dist;
    map<int, int> prev;
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    for (const auto& node : graph) {
        dist[node.first] = INF;
        prev[node.first] = -1;
    }
    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (u == destination) {
            break; // Found the destination
        }

        for (const auto& neighbor : graph.at(u)) {
            int v = neighbor.first;
            float weight = neighbor.second;

            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                prev[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    vector<int> path;
    for (int at = destination; at != -1; at = prev[at]) {
        path.push_back(at);
    }
    reverse(path.begin(), path.end());
    if (path.size() == 1 && path[0] != source) {
        path.clear(); // No path found
    }
    return path;
}

int main(){
    // read the input file u, v, w and make a graph
    // ifstream inputFile("example1.txt");
    ifstream inputFile("airline_distances.txt");
    if (!inputFile) {
        cerr << "Error opening file." << endl;
        return 1;
    }
    map<int, vector<pair<int, float>>> graph;
    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string uStr, vStr, eStr;
        
        if (getline(ss, uStr, ',') && getline(ss, vStr, ',') && getline(ss, eStr)) {
            int u = stoi(uStr);
            int v = stoi(vStr);
            float e = stof(eStr); // <--- Changed to stof for float parsing
            
            graph[u].push_back({v, e});
            // If undirected graph, also add reverse edge:
            // graph[v].push_back({u, e});
        }
    }
    inputFile.close();
    
    // Find the shortest path from source to destination
    int source = 12087; // Example source node
    int destination = 3469; // Example destination node
    vector<int> path = shortestPath(source, destination, graph);
    // calculate the total weight of the path
    float totalWeight = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        int u = path[i];
        int v = path[i + 1];
        auto it = find_if(graph[u].begin(), graph[u].end(), [v](const pair<int, float>& p) { return p.first == v; });
        if (it != graph[u].end()) {
            totalWeight += it->second;
        }
    }
    // Print the shortest path and its total weight
    cout << "Shortest path from " << source << " to " << destination << ": ";
    for (size_t i = 0; i < path.size(); ++i) {
        cout << path[i];
        if (i < path.size() - 1) {
            cout << " -> ";
        }
    }
    cout << endl;
    cout << "Total weight: " << totalWeight << endl;

    return 0;
}