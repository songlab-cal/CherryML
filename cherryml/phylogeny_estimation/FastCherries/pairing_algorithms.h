/*
strategies to pair sequences
*/

#pragma once
#include "types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

// divide and conquer algorithm to find cherries based on hamming distance
std::vector<std::pair<std::string, std::string> > divide_and_pair(
    const std::vector<std::string>& msa_list,
    const std::unordered_map<std::string, std::vector<int> >& msa_map,
    std::mt19937& rng
);
