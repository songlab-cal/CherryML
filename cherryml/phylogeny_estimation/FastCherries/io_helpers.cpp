#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include "io_helpers.h"
#include "matrix_exponential/matrix_exponential.hpp"
#include <limits>


inline int get_char_index(char c, const unordered_map<char, int>& alphabet) {
    auto it = alphabet.find(c);
    return (it != alphabet.end()) ? it->second : -1;
}

std::unordered_map<char, int> read_alphabet(std::string alphabet_path) {
    ifstream alphabet_file(alphabet_path);
    int size;
    alphabet_file >> size;
    std::unordered_map<char, int> alphabet;
    for(int i = 0; i < size; i++) {
        char c;
        alphabet_file >> c;
        alphabet[c] = i;
    }
    alphabet_file.close();
    return alphabet;
}

msa_type read_msa(const std::string& msa_path, const unordered_map<char, int>& alphabet) {
    std::vector<std::string> names;
    unordered_map<std::string, std::vector<int> > nameValueMap;
    std::vector<std::vector<int> > all_sequences;

    std::ifstream inputFile(msa_path);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << msa_path << std::endl;
        return {names, nameValueMap, all_sequences}; 
    }

    std::string line;
    std::string currentName;

    while (std::getline(inputFile, line)) {
        if (!line.empty() && line[0] == '>') {
            // Line starts with '>', it is a name line
            currentName = line.substr(1); // Exclude '>'
            names.push_back(currentName);

            // Read the next line as the corresponding value
            if (std::getline(inputFile, line)) {
                std::vector<int> to_int;
                to_int.reserve(line.length());
                for(int i = 0 ; i < line.length(); i++) {
                    to_int.push_back(get_char_index(line[i], alphabet));
                }
                all_sequences.push_back(to_int);
                nameValueMap[currentName] = to_int;
            } else {
                std::cerr << "Error: Incomplete record for name " << currentName << std::endl;
                names.pop_back(); // Remove the incomplete name
                break; // Stop reading further
            }
        }
    }
    inputFile.close();
    return {names, nameValueMap, all_sequences};
}

void write_cherries_and_distances(const std::vector<std::pair<std::string, std::string>>& cherries,
                     const std::vector<double>& distances,
                     const std::string& output_path) {
    std::ofstream outputFile(output_path);
    outputFile << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (size_t i = 0; i < cherries.size(); ++i) {
        outputFile << cherries[i].first << "\n";
        outputFile << cherries[i].second << "\n";
        outputFile << distances[i] << "\n";
    }

    outputFile.close();
}

void write_site_rates(
    const std::vector<double>& rates,
    const std::string& output_path
) {
    std::ofstream outputFile(output_path);
    outputFile << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    outputFile << rates.size() << " sites\n";
    for (size_t i = 0; i < rates.size(); ++i) {
        outputFile << rates[i] << " ";
    }
    outputFile.close();
}

std::vector<std::string> read_file_paths_from_list(const std::string& file_list_path) {
    std::ifstream file_list(file_list_path);
    if (!file_list.is_open()) {
        std::cerr << "Error: Unable to open file list." << std::endl;
        return {};
    }

    int n;
    file_list >> n;
    file_list.ignore(); 

    std::vector<std::string> file_paths(n);
    std::string path;
        
    for(int i = 0; i < n; i++) {
        std::getline(file_list, path);
        file_paths[i] = path;
    }
    file_list.close();
    return file_paths;
}

std::vector<double> read_rate_matrix_from_file(
    const std::string& matrix_path,
    int size
    
) {
    std::vector<double> rate_matrix(size * size, 0);
    std::ifstream fin(matrix_path);

    if (!fin.is_open()) {
        std::cerr << "Error: Unable to open file "  << std::endl;
        return rate_matrix;
    }
    double x;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if (fin >> x) {
                rate_matrix[i * size + j] = x;
            }
            else {
                throw std::runtime_error("Error: not enough elements in rate matrix file  " + matrix_path + " for a matrix with dimensions " + std::to_string(size));
            }
        }   
    }    
    fin.close();
    return rate_matrix;
}

transition_matrices read_rate_compute_log_transition_matrices(
    const std::string& matrix_path,
    const std::vector<double>& quantization_points,
    const std::vector<double>& rates,
    int size
) {
    const std::vector<double>& rate_matrix = read_rate_matrix_from_file(matrix_path, size);
    transition_matrices log_transition_matrices(quantization_points.size(), rates.size(), size, size);
    double rt[size * size];
    for(int i = 0; i < quantization_points.size(); i++) {
        for(int r = 0; r < rates.size(); r++) {
            for(int j = 0; j < size * size; j++) {
                rt[j] = quantization_points[i] * rates[r] * rate_matrix[j];
            }
            double* ert = r8mat_expm1(size, rt);
            
            for(int j = 0; j < size; j++) {
                for(int k = 0; k < size; k++) {
                    log_transition_matrices(i,r,j,k) = log(ert[j*size + k]);
                }
            }
        }
    }
    return log_transition_matrices;
}

std::vector<double> compute_quantization_points(
    double quantization_grid_center,
    double quantization_grid_step,
    int quantization_grid_num_steps
) {
    std::vector<long double> quantization_points(quantization_grid_num_steps * 2 + 1, 0);
    quantization_points[quantization_grid_num_steps] = (long double)quantization_grid_center;

    for(int i = 1; i <= quantization_grid_num_steps; i++) {
        quantization_points[quantization_grid_num_steps + i] = quantization_points[quantization_grid_num_steps + i - 1]*quantization_grid_step;
        quantization_points[quantization_grid_num_steps - i] = quantization_points[quantization_grid_num_steps - i + 1]/quantization_grid_step;
    }

    std::vector<double> res(quantization_grid_num_steps * 2 + 1, 0);
    for(int i = 0; i < 2 * quantization_grid_num_steps + 1; i++) {
        res[i] = (double)quantization_points[i];
    }
    return res;
}
