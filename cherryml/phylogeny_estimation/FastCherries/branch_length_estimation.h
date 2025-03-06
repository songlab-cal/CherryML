#pragma once
#include "types.h"
#include <string>
#include <vector>

std::vector<int> get_branch_lengths(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points,
    const std::vector<int>& site_to_rate_index,
    const std::vector<std::vector<int>>& valid_indices
);
std::vector<int> get_site_rates(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<int>& lengths_index,
    const std::vector<double>& priors,
    const std::vector<std::vector<int>>& valid_indices
); 
/*
given a list of cherries, computes the MLE distances of each cherry under the WAG model using the provided optimization_algorithm
and likelihood_from_time function
*/
length_and_rates ble(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const std::vector<std::vector<int> >& all_sequences, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points, 
    const std::vector<double>& rate_categories,
    const std::vector<double>& weights_for_initial_site_rates, 
    int max_iters
) ;

