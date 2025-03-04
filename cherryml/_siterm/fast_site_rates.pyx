from time import time
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.math cimport log
import numpy as np
cimport numpy as np

def compute_optimal_site_rates(
    int num_sites,
    list cherries,
    np.ndarray[double, ndim=4] log_mexps_tensor_w_gaps,
    list site_rate_grid,
    list site_rate_prior
) -> list:
    cdef vector[double] optimal_site_rates
    cdef int site_id, cherry_id, site_rate_id
    cdef int num_rates = len(site_rate_grid)
    cdef int num_cherries = len(cherries)
    cdef double log_likelihood
    cdef vector[pair[double, double]] log_likelihoods_and_site_rates  # Use std::pair

    cdef double[:, :, :, :] log_mexps_tensor_w_gaps_mv = log_mexps_tensor_w_gaps

    start_time = time()
    for site_id in range(num_sites):
        # Extract cherries at this site
        cherries_at_site = [
            (x[site_id], y[site_id], t) for (x, y, t) in cherries
        ]

        log_likelihoods_and_site_rates.clear()

        for site_rate_id, site_rate in enumerate(site_rate_grid):
            log_likelihood = log(site_rate_prior[site_rate_id])

            for cherry_id, (x_i, y_i, t) in enumerate(cherries_at_site):
                log_likelihood += log_mexps_tensor_w_gaps_mv[site_rate_id, cherry_id, x_i, y_i]

            log_likelihoods_and_site_rates.push_back(pair[double, double](log_likelihood, site_rate))

        # Find the optimal site rate (the one with the max log-likelihood)
        optimal_site_rate = max(log_likelihoods_and_site_rates, key=lambda x: x[0])[1]
        optimal_site_rates.push_back(optimal_site_rate)

    # print(f"I'm cython. Time for main loop: {time() - start_time:.3f} seconds")

    return list(optimal_site_rates)
