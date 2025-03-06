#include "pairing_algorithms.h"
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <float.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <random>


inline double hamming_distance(
    const std::vector<int>& x, 
    const std::vector<int>& y
) {
    int dist = 0;
    int n = x.size();
    int int_x;
    int int_y;
    int count = 0;
    for(int i = 0; i < n; i++) {

        int_x = x[i];
        int_y = y[i];

        if(int_x != -1 && int_y != -1) {
            count += 1;
            dist += int_x != int_y;
        }
    }
    if(count == 0) {
        return 0;
    }
    
    // original code used log likelihood of a cherry as the distance, so hamming distance is negated to keep conistent
    return dist*-1.0/count;
}

inline std::pair<std::string, std::vector<double> > find_farthest(
    const std::vector<std::string>& msa_list,
    const std::unordered_map<std::string, std::vector<int> >& msa_map,
    const std::string& x
){
    double farthest = DBL_MAX;
    std::string y;
    const std::vector<int> &x_seq = msa_map.at(x);
    std::vector<double> distances;
    distances.reserve(msa_list.size());
    for(const std::string &seq:msa_list) {
        double d = hamming_distance(msa_map.at(seq), x_seq);
        distances.push_back(d);
        if(d < farthest) {
            farthest = d;
            y = seq;
        }  
    }
    return {y, distances};
}

inline std::vector<bool> partition_subset_distance(
    const std::vector<std::string>& msa_list,
    const std::unordered_map<std::string, std::vector<int> >& msa_map,
    const std::vector<double> dist_x,
    const std::string& x,
    const std::string& y
) {
    int n = msa_list.size();
    const std::vector<int> &y_seq = msa_map.at(y);
    std::vector<bool> closer_left(n, false); 
    for(int i = 0; i < n; i++) {
        closer_left[i] = (dist_x[i] >= hamming_distance(msa_map.at(msa_list[i]), y_seq));
    }
    return closer_left;
}

inline std::pair<std::string, std::vector<std::pair<std::string, std::string> > > divide(
    const std::vector<std::string>& msa_list,
    const std::unordered_map<std::string, std::vector<int> >& msa_map,
    std::mt19937& rng
) {
    // base cases
    if(msa_list.size() == 2) {
        return {"", {{msa_list[0],msa_list[1]}}};
    }
    else if(msa_list.size() == 1) {
        return {msa_list[0],{}};
    }
    else if(msa_list.size() == 0) {
        return {"", {}};
    }

    // only source of non-determinism, seed the rng in main file
    // Use Mersenne Twister instead of rand()
    std::uniform_int_distribution<size_t> dist(0, msa_list.size() - 1);
    std::string x = msa_list[dist(rng)];  // Randomly select from msa_list
    // std::string x = msa_list[rand()%msa_list.size()];  // Old code with rand() - not reproducible accross different machine architectures.
    std::pair<std::string, std::vector<double> > node_and_dist;
    node_and_dist = find_farthest(
        msa_list,
        msa_map,
        x
    );
    x = node_and_dist.first;
    
    node_and_dist = find_farthest(
        msa_list,
        msa_map,
        x
    );
    std::string y = node_and_dist.first;

    std::vector<bool> partition;
    partition = partition_subset_distance(
        msa_list,
        msa_map,
        node_and_dist.second,
        x,
        y
    );

    std::vector<std::string> close_x, close_y;
    for(int i = 0; i < msa_list.size(); i++) {
        if(partition[i] && msa_list[i]!=y) {
            close_x.push_back(msa_list[i]);
        } else {
            close_y.push_back(msa_list[i]);
        }
    }
    std::pair<std::string, std::vector<std::pair<std::string, std::string> > > unpaired_and_cherries_x = divide(
        close_x,
        msa_map,
        rng
    );
    std::pair<std::string, std::vector<std::pair<std::string, std::string> > > unpaired_and_cherries_y = divide(
        close_y,
        msa_map,
        rng
    );

    std::vector<std::pair<std::string, std::string> > cherries;
    cherries.reserve(unpaired_and_cherries_x.second.size() + unpaired_and_cherries_y.second.size());
    cherries.insert(
        cherries.end(), 
        unpaired_and_cherries_x.second.begin(), 
        unpaired_and_cherries_x.second.end()
    );
    cherries.insert(
        cherries.end(), 
        unpaired_and_cherries_y.second.begin(), 
        unpaired_and_cherries_y.second.end()
    );

    std::string unpaired = "";
    if(unpaired_and_cherries_x.first.length() > 0 && unpaired_and_cherries_y.first.length() > 0) {
        cherries.push_back({unpaired_and_cherries_x.first, unpaired_and_cherries_y.first});
    } else {
        unpaired = unpaired_and_cherries_x.first + unpaired_and_cherries_y.first;
    }

    return {unpaired, cherries};
}

std::vector<std::pair<std::string, std::string> > divide_and_pair(
    const std::vector<std::string>& msa_list,
    const std::unordered_map<std::string, std::vector<int> >& msa_map,
    std::mt19937& rng
) {
    return divide(
        msa_list,
        msa_map,
        rng
    ).second;
}
