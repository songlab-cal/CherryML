#include "branch_length_estimation.h"
#include "types.h"
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <float.h>
#include <algorithm> // Required for std::sort

void site_rates_gamma_bins_all_pairs_inplace(
    const std::vector<std::vector<int> >& all_sequences,
    const std::vector<double>& weights_for_initial_site_rates,
    std::vector<int>& site_to_rate_index,
    int s
) {
    int l = all_sequences[0].size();
    std::vector<std::pair<int, int> > counts_and_index;
    counts_and_index.reserve(l);
    for(int j = 0; j < l; j++) {
        counts_and_index.push_back({0, j});
    }
    matrix counts(l,s);

    for(int i = 0; i < all_sequences.size(); i++) {
        for(int j = 0; j < l; j++) {
            if(all_sequences[i][j] != -1) {
                counts(j, all_sequences[i][j]) += 1;
            }
        }
    }
    for(int j = 0; j < l; j++) {
        int total = 0;
        int non_missing = 0;
        for(int k = 0; k < s; k++) {
            non_missing += counts(j,k);
        }
        for(int k = 0; k < s; k++) {
            total += (non_missing - counts(j,k)) * counts(j,k);
        }
        counts_and_index[j].first = total;
    }
    std::sort(counts_and_index.begin(), counts_and_index.end());
    
    // get the cuttoffs for each bin based on the weight for that rate category. 
    // the weight of the curve is the cdf of gamma(shape=3, scale=1/3) from the midpoint of the previous
    // category to the midpoint of the next
    std::vector<double> weights;
    weights.reserve(weights_for_initial_site_rates.size());
    for(int r = 0; r < weights_for_initial_site_rates.size(); r++) {
        weights[r] = (int)std::round(weights_for_initial_site_rates[r] * l);
    }

    int rc = 0;
    for(int i = 0; i < l; i++) {
        rc += (i >= weights[rc]);
        site_to_rate_index[counts_and_index[i].second] = rc;
    }
}

std::vector<int> get_branch_lengths(
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points,
    const std::vector<int>& site_to_rate_index,
    const std::vector<std::vector<int>>& valid_indices
) {
    std::vector<int> branch_lengths;
    branch_lengths.reserve(cherries.size());

    for (int cherry_index = 0; cherry_index < cherries.size(); cherry_index++) {
        const std::vector<int>& x = cherries[cherry_index].first;
        const std::vector<int>& y = cherries[cherry_index].second;
        std::vector<double> cache(quantization_points.size(), 1.0); 

        int low = 0;
        int high = quantization_points.size() - 1;

        while (low < high) {
            int mid = low + (high - low) / 2;
            double ll_m = 0.0;
            double ll_m1 = 0.0;
            #pragma GCC unroll 4
            for (size_t idx = 0; idx < valid_indices[cherry_index].size(); idx++) {
                const int i = valid_indices[cherry_index][idx];
                const int xi = x[i];
                const int yi = y[i];
                ll_m += log_transition_matrices(mid, site_to_rate_index[i], xi, yi) + 
                        log_transition_matrices(mid, site_to_rate_index[i], yi, xi);
                ll_m1 += log_transition_matrices(mid + 1, site_to_rate_index[i], xi, yi) + 
                        log_transition_matrices(mid + 1, site_to_rate_index[i], yi, xi);
            }

            // Update the search range
            if (ll_m > ll_m1) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        branch_lengths.push_back(low);
    }

    return branch_lengths;
}

std::vector<int> get_site_rates(
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<int>& lengths_index,
    const std::vector<double>& priors,
    const std::vector<std::vector<int>>& valid_indices
) {
    std::vector<int> site_rates;
    site_rates.reserve(cherries[0].first.size());
    for (int site_index = 0; site_index < cherries[0].first.size(); site_index++) {
        int low = 0;
        int high = priors.size() - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            double ll_m = priors[mid];
            double ll_m1 = priors[mid + 1];
            #pragma GCC unroll 4
            for (size_t idx = 0; idx < valid_indices[site_index].size(); idx++) {
                const int i = valid_indices[site_index][idx];
                const int xi = cherries[i].first[site_index];
                const int yi = cherries[i].second[site_index];
                ll_m += log_transition_matrices(lengths_index[i], mid, xi, yi) + 
                        log_transition_matrices(lengths_index[i], mid, yi, xi);
                ll_m1 += log_transition_matrices(lengths_index[i], mid + 1, xi, yi) + 
                        log_transition_matrices(lengths_index[i], mid + 1, yi, xi);
            }
            // Update the search range
            if (ll_m > ll_m1) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        site_rates.push_back(low);
    }

    return site_rates;
}

length_and_rates ble(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const std::vector<std::vector<int> >& all_sequences, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points, 
    const std::vector<double>& rate_categories,
    const std::vector<double>& weights_for_initial_site_rates, 
    int max_iters
) {
    int l = cherries[0].first.size();
    std::vector<int> site_to_rate_index(l, 0);
    int s = log_transition_matrices.n;
    site_rates_gamma_bins_all_pairs_inplace(
        all_sequences,
        weights_for_initial_site_rates,
        site_to_rate_index,
        s
    );
    // Precompute valid indices for each site across all cherries
    std::vector<std::vector<int>> valid_indices_for_site_rate(cherries[0].first.size());
    for (int site_index = 0; site_index < cherries[0].first.size(); site_index++) {
        for (int i = 0; i < cherries.size(); i++) {
            int xi = cherries[i].first[site_index];
            int yi = cherries[i].second[site_index];
            if (xi != -1 && yi != -1) {
                valid_indices_for_site_rate[site_index].push_back(i);  // Store valid index for this site
            }
        }
    }

    std::vector<std::vector<int>> valid_indices_for_branch_length(cherries.size());
    for (int cherry_index = 0; cherry_index < cherries.size(); cherry_index++) {
        const std::vector<int>& x = cherries[cherry_index].first;
        const std::vector<int>& y = cherries[cherry_index].second;
        for (int i = 0; i < x.size(); i++) {
            if (x[i] != -1 && y[i] != -1) {
                valid_indices_for_branch_length[cherry_index].push_back(i);  // Store index of valid pair
            }
        }
    }

    // begin coordinate ascent
    std::vector<int> lengths_index = get_branch_lengths(
        cherries,
        log_transition_matrices, 
        quantization_points, 
        site_to_rate_index,
        valid_indices_for_branch_length
    );
    
    std::vector<double> priors;
    priors.reserve(rate_categories.size());
    for(double rate:rate_categories) {
        priors.push_back(2*std::log(rate) - 3*rate);
    }

    bool match = false;
    while(!match && max_iters) {
        //std::cout << max_iters << std::endl;
        site_to_rate_index = get_site_rates(
            cherries, 
            log_transition_matrices,
            lengths_index,
            priors,
            valid_indices_for_site_rate
        );
        std::vector<int> new_lengths_index = get_branch_lengths(
            cherries,
            log_transition_matrices, 
            quantization_points, 
            site_to_rate_index,
            valid_indices_for_branch_length
        );
        
        match = true;
        for(int i = 0; match && i < lengths_index.size(); i++) {
            match = match && (lengths_index[i] == new_lengths_index[i]);
        }
        lengths_index = new_lengths_index;
        max_iters--;
    }
    //std::cout << std::endl;
    std::vector<double> lengths;
    lengths.reserve(cherries.size());
    for(int l:lengths_index) {
        lengths.push_back(quantization_points[l]);
    }
    
    std::vector<double> site_to_rate;
    site_to_rate.reserve(l);
    for(int r:site_to_rate_index) {
        site_to_rate.push_back(rate_categories[r]);
    }
    return {lengths, site_to_rate};
}

