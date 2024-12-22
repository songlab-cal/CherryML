#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <assert.h>
#include <math.h>
#include <random>

#define forn(i, n) for(int i = 0; i < int(n); i++)
#define pb push_back

using namespace std;

const int ROOT = 0;
int N, L;
const int maxN = 100000;
const int maxS = 30;
string sequences[maxN];
vector<int> G[maxN];
int dp[maxN][maxS];

std::mt19937 rng;

void read_tree(string tree_filepath){
    // cerr << "Reading graph ... " << endl;
    ifstream fin(tree_filepath);
    fin >> N;
    forn(i, N - 1){
        int u, v;
        fin >> u >> v;
        G[u].pb(v);
    }
}

void read_msa(string msa_filepath){
    // cerr << "Reading sequences ... " << endl;
    ifstream fin(msa_filepath);
    int nleaves;
    fin >> nleaves;
    forn(i, nleaves){
        int v;
        string s;
        fin >> v >> s;
        sequences[v] = s;
        if(i) assert(int(s.size()) == L);
        else L = s.size();
    }
}

const int INF = 100000000;

int aa_to_int(char aa){
    forn(i, maxS - 1){
        if(char(i + 'A') == aa)
            return i;
    }
    assert(aa == '-');
    return maxS - 1;
}

char int_to_aa(int i){
    if(i == maxS - 1)
        return '-';
    return char(i + 'A');
}

void dfs(int p, int v, int site_id){
    // cerr << p << " -> " << v << endl;
    for(auto u : G[v])
        dfs(v, u, site_id);
    forn(i, maxS){
        // Compute dp[v][i]
        dp[v][i] = 0;
        if(G[v].size() == 0){
            // Leaf! Base case!
            if(sequences[v] == ""){
                cerr << "Leaf " << v << " has empty sequence!" << endl;
            }
            assert(sequences[v] != "");
            if(aa_to_int(sequences[v][site_id]) != i)
                dp[v][i] = INF;
        }
        else{
            // Recursive case
            for(auto u : G[v]){
                // Find optimal assignment for child u
                int optimal_cost_for_u = INF;
                forn(j, maxS){
                    // Assign j to u
                    int cost_u_j = dp[u][j] + (j != i);
                    optimal_cost_for_u = min(optimal_cost_for_u, cost_u_j);
                }
                dp[v][i] += optimal_cost_for_u;
            }
        }
    }
}

void reconstruct_solution(int v, int i, int site_id){
    for(auto u : G[v]){
        // Find optimal assignment for child u
        vector<int> optimal_assignments;
        int optimal_cost_for_u = INF;
        forn(j, maxS){
            // Assign j to u
            int cost_u_j = dp[u][j] + (j != i);
            optimal_cost_for_u = min(optimal_cost_for_u, cost_u_j);
        }
        forn(j, maxS){
            // Assign j to u
            int cost_u_j = dp[u][j] + (j != i);
            if(optimal_cost_for_u == cost_u_j)
                optimal_assignments.pb(j);
        }
        // Choose the state at random from all valid states, and proceed.
        int chosen_state = optimal_assignments[rng() % optimal_assignments.size()];
        if(G[u].size() == 0){
            // Nothing to do really, just sanity check
            assert(optimal_assignments.size() == 1);
            assert(sequences[u][site_id] == int_to_aa(optimal_assignments[0]));
        } else {
            sequences[u] += int_to_aa(chosen_state);
            reconstruct_solution(u, chosen_state, site_id);
        }
    }
}

void show(vector<int> & v){
    for(auto x: v) cerr << x << " ";
    cerr << endl;
}

void solve_maximum_parsimony_for_site(int site_id){
    // cerr << "Solving maximum parsimony problem for site " << site_id << " ... " << endl;
    dfs(-1, 0, site_id);
    int maximum_parsimony = INF;
    forn(i, maxS) maximum_parsimony = min(maximum_parsimony, dp[ROOT][i]);
    // cerr << "Maximum Parsimony = " << maximum_parsimony << endl;
    vector<int> optimal_root_states;
    forn(i, maxS) if(dp[ROOT][i] == maximum_parsimony) optimal_root_states.pb(i);
    // show(optimal_root_states);
    int chosen_root_state = optimal_root_states[rng() % (optimal_root_states.size())];
    sequences[ROOT] += int_to_aa(chosen_root_state);
    reconstruct_solution(ROOT, chosen_root_state, site_id);
}

void solve_maximum_parsimony(){
    // cerr << "Solving maximum parsimony problem ... " << endl;
    forn(i, L){
        solve_maximum_parsimony_for_site(i);
    }
}

void write_out_solution(string solution_filepath){
    ofstream fout(solution_filepath);
    fout << N << endl;
    forn(v, N){
        fout << v << " " << sequences[v] << endl;
    }
}

void test(){
    assert(aa_to_int('A') == 0);
    assert(aa_to_int('-') == maxS - 1);
    assert(int_to_aa(maxS - 1) == '-');
}

int main(int argc, char* argv[]){
    // cerr << "Running maximum parsimony C++ script ... " << endl;
    if(argc != 4){
        cerr << "ERROR: The tree_filepath, msa_filepath, and solution_filepath should be provided!" << endl;
        return 1;
    }
    test();
    string tree_filepath = argv[1];
    string msa_filepath = argv[2];
    string solution_filepath = argv[3];
    read_tree(tree_filepath);
    read_msa(msa_filepath);
    solve_maximum_parsimony();
    write_out_solution(solution_filepath);
    return 0;
}