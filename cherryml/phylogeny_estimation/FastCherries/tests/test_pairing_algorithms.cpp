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

void test_dc_pairer(
    const std::vector<std::string> &msa_list,
    const std::unordered_map<std::string, std::vector<int> > &msa_map,
    const std::vector<std::pair<std::string, std::string> > &expected,
    const std::string success_message
) {
    std::vector<std::pair<std::string, std::string> > res = divide_and_pair(msa_list, msa_map);

    assert(res.size() == expected.size());
    for(int i = 0; i < res.size(); i++) {
        std::cout << res[i].first << " " << res[i].second  << ", ";
    }
    for(int i = 0; i < res.size(); i++) {
        
        assert(expected[i] == res[i]);
    }

    std::cout << success_message << std::endl;
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

    std::vector<std::pair<std::string, std::string> > expected = {{"e", "e1"}, {"f", "f1"}, {"d", "d1"}, {"c", "c1"}, {"b", "b1"}, {"a", "a1"} };

    test_dc_pairer(msa_list, msa_map, expected, "sanity 1 passed");
}

void test_dc_pair_with_hamming_distance2() {
    srand(1);
    std::vector<std::string> msa_list = {"a", "b", "c", "d", "e", "f", "a1", "b1", "c1", "d1", "e1", "f1"};
    std::unordered_map<std::string, std::vector<int> > msa_map;  
    msa_map["a"] = string_to_ints("QQQEEECQ");
    msa_map["b"] = string_to_ints("EEEQQQCC");
    msa_map["c"] = string_to_ints("EEQQCCQC");
    msa_map["d"] = string_to_ints("CEEQECCE");
    msa_map["e"] = string_to_ints("EEQQEEQE");
    msa_map["f"] = string_to_ints("EEQQCCCE");
    msa_map["a1"] = string_to_ints("EEQQEECE");
    msa_map["b1"] = string_to_ints("EEEQQQCE");
    msa_map["c1"] = string_to_ints("CCEEQQQQ");
    msa_map["d1"] = string_to_ints("CEEQECQE");
    msa_map["e1"] = string_to_ints("QQQEEEEC");
    msa_map["f1"] = string_to_ints("EEQQCCQC");

    std::vector<std::pair<std::string, std::string> > expected = {{"a", "e1"}, {"b", "b1"}, {"c", "f1"}, {"d1", "f"}, {"e", "a1"}, {"c1", "d"}};

    test_dc_pairer(msa_list, msa_map, expected, "sanity 2 passed");
}

void test_dc_pair_with_hamming_distance3() {
    srand(1);
    std::vector<std::string> msa_list = {"a", "b", "c", "d", "e", "f", "a1", "b1", "c1", "d1", "e1", "f1"};
    std::unordered_map<std::string, std::vector<int> > msa_map;  
    msa_map["a"] = string_to_ints("TWYWYTWYWYWTYWYW");
    msa_map["b"] = string_to_ints("TYWYWTYTWWYTWYWW");
    msa_map["c"] = string_to_ints("YWYWTYWYTWYTWYWY");
    msa_map["d"] = string_to_ints("WYTWYWYTTWYWTYTT");
    msa_map["e"] = string_to_ints("YTWYTWTWTWTTTWWY");
    msa_map["f"] = string_to_ints("YYYYTWTWTTWTWWYW");
    msa_map["a1"] = string_to_ints("WYTYWTWTTWYYWYWY");
    msa_map["b1"] = string_to_ints("WTYYYWYWWWYTWTTT");
    msa_map["c1"] = string_to_ints("YTYYTYWWYTWTWWWW");
    msa_map["d1"] = string_to_ints("WYYTWYWYTWTWWTTT");
    msa_map["e1"] = string_to_ints("YTWYTYWWTYTWYTWY");
    msa_map["f1"] = string_to_ints("TYTYTYYTWTWWWTWW");

    std::vector<std::pair<std::string, std::string> > expected = {{"a", "f"}, {"c", "c1"}, {"b", "e"}, {"d", "a1"}, {"d1", "b1"}, {"e1", "f1"}};

    test_dc_pairer(msa_list, msa_map, expected, "sanity 3 passed");
}
int main() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    test_dc_pair_with_hamming_distance();
    test_dc_pair_with_hamming_distance2();
    test_dc_pair_with_hamming_distance3();
    return 0;
}
 