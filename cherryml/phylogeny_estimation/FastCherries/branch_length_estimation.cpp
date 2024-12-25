#include "branch_length_estimation.h"
#include "types.h"
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <float.h>

// copied from FastTree 2.1
double LnGamma (double alpha)
{
/* returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.
   Stirling's formula is used for the central polynomial part of the procedure.
   Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
   Communications of the Association for Computing Machinery, 9:684
*/
   double x=alpha, f=0, z;
   if (x<7) {
      f=1;  z=x-1;
      while (++z<7)  f*=z;
      x=z;   f=-(double)log(f);
   }
   z = 1/(x*x);
   return  f + (x-0.5)*(double)log(x) - x + .918938533204673
	  + (((-.000595238095238*z+.000793650793651)*z-.002777777777778)*z
	       +.083333333333333)/x;
}

// copied from FastTree 2.1
double IncompleteGamma(double x, double alpha, double ln_gamma_alpha)
{
/* returns the incomplete gamma ratio I(x,alpha) where x is the upper
	   limit of the integration and alpha is the shape parameter.
   returns (-1) if in error
   ln_gamma_alpha = ln(Gamma(alpha)), is almost redundant.
   (1) series expansion     if (alpha>x || x<=1)
   (2) continued fraction   otherwise
   RATNEST FORTRAN by
   Bhattacharjee GP (1970) The incomplete gamma integral.  Applied Statistics,
   19: 285-287 (AS32)
*/
   int i;
   double p=alpha, g=ln_gamma_alpha;
   double accurate=1e-8, overflow=1e30;
   double factor, gin=0, rn=0, a=0,b=0,an=0,dif=0, term=0, pn[6];

   if (x==0) return (0);
   if (x<0 || p<=0) return (-1);

   factor=(double)exp(p*(double)log(x)-x-g);
   if (x>1 && x>=p) goto l30;
   /* (1) series expansion */
   gin=1;  term=1;  rn=p;
 l20:
   rn++;
   term*=x/rn;   gin+=term;

   if (term > accurate) goto l20;
   gin*=factor/p;
   goto l50;
 l30:
   /* (2) continued fraction */
   a=1-p;   b=a+x+1;  term=0;
   pn[0]=1;  pn[1]=x;  pn[2]=x+1;  pn[3]=x*b;
   gin=pn[2]/pn[3];
 l32:
   a++;  b+=2;  term++;   an=a*term;
   for (i=0; i<2; i++) pn[i+4]=b*pn[i+2]-an*pn[i];
   if (pn[5] == 0) goto l35;
   rn=pn[4]/pn[5];   dif=fabs(gin-rn);
   if (dif>accurate) goto l34;
   if (dif<=accurate*rn) goto l42;
 l34:
   gin=rn;
 l35:
   for (i=0; i<4; i++) pn[i]=pn[i+2];
   if (fabs(pn[4]) < overflow) goto l32;
   for (i=0; i<4; i++) pn[i]/=overflow;
   goto l32;
 l42:
   gin=1-factor*gin;

 l50:
   return (gin);
}

// copied from FastTree 2.1
double PGamma(double x, double alpha)
{
  /* scale = 1/alpha */
  return IncompleteGamma(x*alpha,alpha,LnGamma(alpha));
}

void site_rates_gamma_bins_all_pairs_inplace(
    const std::vector<std::vector<int> >& all_sequences,
    const std::vector<double>& rate_categories,
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
    std::vector<double> midpoints;
    midpoints.reserve(rate_categories.size() - 1);
    for(int i = 1; i < rate_categories.size(); i++) {
        midpoints.push_back(sqrt(rate_categories[i-1]*rate_categories[i]));
    }
    //std::cout << "midpoints ";
    //for(double m:midpoints) {
    //    std::cout << m << " ";
    //}
    //std::cout << std::endl;
    std::vector<double> weights;
    double shape = 3.0;
    weights.reserve(rate_categories.size());
    for(int i = 0; i < rate_categories.size() - 1; i++) {
        weights.push_back(PGamma(midpoints[i], shape));
    }
    weights.push_back(1.0);
    
    weights[0] = (int)std::round(weights[0] * l);
    for(int r = 1; r < rate_categories.size(); r++) {
        weights[r] = (int)std::round(weights[r] * l);
    }

    int rc = 0;
    for(int i = 0; i < l; i++) {
        rc += (i >= weights[rc]);
        site_to_rate_index[counts_and_index[i].second] = rc;
    }
}


std::vector<int> get_branch_lengths(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points,
    const std::vector<int>& site_to_rate_index
) {
    std::vector<int> branch_lengths;
    branch_lengths.reserve(cherries.size());
    for(int cherry_index = 0; cherry_index < cherries.size(); cherry_index++) {
        const std::vector<int>& x = cherries[cherry_index].first;
        const std::vector<int>& y = cherries[cherry_index].second;
        // Binary search-based peak finding algorithm
        double max_ll = -DBL_MAX;

        // Apply binary search for maximum likelihood estimation
        int low = 0;
        int high = quantization_points.size() - 1;

        while (low < high) {
            int mid = low + (high - low) / 2;
            double ll_m = 0.0;
            double ll_m1 = 0.0;
            // Calculate log likelihood for current quantization point
            for(int i = 0; i < x.size(); i++){
                int xi = x[i];
                int yi = y[i];
                if(xi != -1 && yi != -1) {
                    ll_m += log_transition_matrices(mid, site_to_rate_index[i], xi, yi) + 
                        log_transition_matrices(mid, site_to_rate_index[i], yi, xi);
                    ll_m1 += log_transition_matrices(mid + 1, site_to_rate_index[i], xi, yi) + 
                        log_transition_matrices(mid + 1, site_to_rate_index[i], yi, xi);
                }
            }
            max_ll = std::max(max_ll, ll_m);
            max_ll = std::max(max_ll, ll_m1);
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
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points,
    const std::vector<int>& lengths_index,
    const std::vector<double>& priors
) {
    std::vector<int> site_rates;
    site_rates.reserve(cherries[0].first.size());
    for(int site_index = 0; site_index < cherries[0].first.size(); site_index++) {
        int low = 0, high = priors.size() - 1;
        int best_rate = 0;
        double best_ll = -DBL_MAX;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            // Calculate the log-likelihood of rate `mid`
            double ll_of_rate = priors[mid];
            for (int i = 0; i < cherries.size(); i++) {
                int xi = cherries[i].first[site_index];
                int yi = cherries[i].second[site_index];
                if (xi != -1 && yi != -1) {
                    ll_of_rate += log_transition_matrices(lengths_index[i], mid, xi, yi) +
                                log_transition_matrices(lengths_index[i], mid, yi, xi);
                }
            }

            // If we found a better log-likelihood, store it and adjust the search bounds.
            if (ll_of_rate > best_ll) {
                best_ll = ll_of_rate;
                best_rate = mid;
                low = mid + 1;  // Try to find a better rate by going to the higher half
            } else {
                high = mid - 1;  // Otherwise, search the lower half
            }
        }
        //std::cout << std::endl;
        site_rates.push_back(best_rate);
    }
    return site_rates;
}

length_and_rates ble(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries, 
    const std::vector<std::vector<int> >& all_sequences, 
    const transition_matrices& log_transition_matrices,
    const std::vector<double>& quantization_points, 
    const std::vector<double>& rate_categories,
    int max_iters
) {
    int l = cherries[0].first.size();
    std::vector<int> site_to_rate_index(l, 0);
    int s = log_transition_matrices.n;
    site_rates_gamma_bins_all_pairs_inplace(
        all_sequences,
        rate_categories,
        site_to_rate_index,
        s
    );
   //for(int i = 0; i < l; i++) {
   //    std::cout << site_to_rate_index[i] << " ";
   //}
   //std::cout << std::endl;
   //std::cout<<"getting initial lengths" <<std::endl;
    std::vector<int> lengths_index = get_branch_lengths(
        cherries,
        log_transition_matrices, 
        quantization_points, 
        site_to_rate_index
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
            quantization_points,
            lengths_index,
            priors
        );
        std::vector<int> new_lengths_index = get_branch_lengths(
            cherries,
            log_transition_matrices, 
            quantization_points, 
            site_to_rate_index
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