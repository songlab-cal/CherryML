# pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "types.h"


std::unordered_map<char, int> read_alphabet(std::string alphabet_path);

msa_type read_msa(const std::string& msa_path, const std::unordered_map<char, int>& alphabet);

void write_cherries_and_distances(const std::vector<std::pair<std::string, std::string>>& cherries,
                     const std::vector<double>& distances,
                     const std::string& output_path);

void write_site_rates(const std::vector<double>& rates,
                     const std::string& output_path);

std::vector<std::string> read_file_paths_from_list(const std::string& file_list_path);

std::vector<double> read_rate_matrix_from_file(
    const std::string& matrix_path,
    int size
    
);

transition_matrices read_rate_compute_log_transition_matrices(
    const std::string& matrix_path,
    const std::vector<double>& quantization_points,
    const std::vector<double>& rates,
    int size
);

std::vector<double> compute_quantization_points(
    double quantization_grid_center,
    double quantization_grid_step,
    int quantization_grid_num_steps
);