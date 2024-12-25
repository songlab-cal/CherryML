#pragma once
#include<string>
#include<vector>
#include<unordered_map>
#include<iostream>

const std::unordered_map<char, int> default_alphabet = {
    {'A',0},
    {'R',1},
    {'N',2},
    {'D',3},
    {'C',4},
    {'Q',5},
    {'E',6},
    {'G',7},
    {'H',8},
    {'I',9},
    {'L',10},
    {'K',11},
    {'M',12},
    {'F',13}, 
    {'P',14},
    {'S',15},
    {'T',16},
    {'W',17},
    {'Y',18},
    {'V',19}
};

const std::string TEST_MSA_PATH = "tests/Aln0000_txt-gb_phyml.txt";

struct length_and_rates {
    std::vector<double> lengths;
    std::vector<double> rates;
};

struct transition_matrices {
    int t;
    int r; 
    int n; 
    int m; 
    std::vector<double> matrix;

    transition_matrices(int _t, int _r, int _n, int _m) : t(_t), r(_r), n(_n), m(_m) {
        matrix.resize(t * r * n * m);
    }

    double& operator()(int time_idx, int rate_idx, int n_idx, int m_idx) {
        int index = time_idx * r * n * m + rate_idx * n * m + n_idx * m + m_idx;
        return matrix[index];
    }
    
    const double& operator()(int time_idx, int rate_idx, int n_idx, int m_idx) const{
        int index = time_idx * r * n * m + rate_idx * n * m + n_idx * m + m_idx;
        return matrix[index];
    }

    int size() const {
        return matrix.size();
    }
};


struct msa_type{
    std::vector<std::string> all_names;
    std::unordered_map<std::string, std::vector<int> > names_to_sequence;
    std::vector<std::vector<int> > all_sequences;
};


struct matrix {
    int n; 
    int m;  
    std::vector<double> data;  // Flattened representation

    matrix(int n_size, int m_size)
        : n(n_size), m(m_size), data(n * m, 0.0) {
    }

    double& operator()(int ni, int mi) {
        return data[ni * m + mi];
    }

    const double& operator()(int ni, int mi) const {
        return data[ni * m + mi];
    }
};
