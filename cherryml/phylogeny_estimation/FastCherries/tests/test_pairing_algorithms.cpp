#include "../pairing_algorithms.h"
#include "../types.h"
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

inline int get_char_index(char c) {
    auto it = default_alphabet.find(c);
    return (it != default_alphabet.end()) ? it->second : -1;
}
std::vector<int> string_to_ints(const std::string& x) {
    std::vector<int> res;
    res.reserve(x.length());
    for(int i = 0; i < x.length(); i++) {
        res.push_back(get_char_index(x[i]));
    }
    return res;
}

void test_dc_pair_with_hamming_distance() {
    srand(1);
    std::vector<std::string> msa_list = {"a", "b", "c", "d", "e", "f", "a1", "b1", "c1", "d1", "e1", "f1"};
    std::unordered_map<std::string, std::vector<int> > msa_map;  
    msa_map["a"] = string_to_ints("AAAAAAAAAA");
    msa_map["b"] = string_to_ints("AAAAAAAAAA");
    msa_map["c"] = string_to_ints("NNNNNNNNNN");
    msa_map["d"] = string_to_ints("NNNNRRRRTT");
    msa_map["e"] = string_to_ints("CDCDCDCDCD");
    msa_map["f"] = string_to_ints("DCDCDCDCDC");
    msa_map["a1"] = string_to_ints("AAAAAAAANN");
    msa_map["b1"] = string_to_ints("AAAAAAAAAA");
    msa_map["c1"] = string_to_ints("NNNNNNNNRR");
    msa_map["d1"] = string_to_ints("NNNNRRRRNN");
    msa_map["e1"] = string_to_ints("CDCDCDCDCD");
    msa_map["f1"] = string_to_ints("DCDCDCDCDC");

    std::vector<std::pair<std::string, std::string> > expected = {{"a", "b"},{"c", "d"}, {"e", "f"},{"a1", "b1"},{"c1", "d1"}, {"e1", "f1"}};

    std::vector<std::pair<std::string, std::string> > res = divide_and_pair(msa_list, msa_map);
    
    srand(1); 
    std::vector<std::pair<std::string, std::string> > res1 = divide_and_pair(msa_list, msa_map);
    
    for(int i = 0; i < res.size(); i++) {
        bool match = (res[0] == res1[0] && res[1] == res1[1]) ||  (res[0] == res1[1] && res[1] == res1[0]);
        assert(match);
    }
    assert(res.size() == expected.size());

    std::cout << "dc with hamming distance sanity passed" << std::endl;
}
int main() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    test_dc_pair_with_hamming_distance();
    return 0;
}
 