/**
 * @brief Processes a list of MSA's in parallel and outputs a set of cherries for each MSA.
 *
 * @param rate_matrix_path [string] 
 *        A file path to the rate matrix. The i-th column and j-th row should 
 *        correspond to the rate of transitioning from alphabet[i] to alphabet[j]. 
 *
 * @param msa_list_path [string] 
 *        A file path containing a list of file paths (one per line) pointing to MSA files.
 *
 * @param output_list_path [string] 
 *        A file path containing a list of file paths (one per line) where the output will be written.
 *
 * @param profiling_list_path [string] 
 *        A file path containing a list of file paths (one per line) where the pairing and BLE times will be written.
 *
 * @param seed [integer] 
 *        A seed used by the divide and conquer pairer.
 *
 * @param quantization_grid_center [double] 
 *        The center of the quantization grid.
 *
 * @param quantization_grid_step [double] 
 *        The multiplicative step size for the quantization grid.
 *
 * @param quantization_grid_num_steps [integer] 
 *        The number of steps in the quantization grid.
 *
 * @param num_rate_categories_ble [integer] 
 *        The number of rate categories used in BLE. Default value is 1 (i.e., the WAG model).
 *
 * @param max_iters_ble [integer] 
 *        The maximum number of iterations for BLE. Default value is 50.
 * @param alphabet_path [string] optional
 *        Path to a file containing the alphabet used. The alphabet should be in the same order as the 
 *        rows and columns of the rate matrix if not provided, default_alphabet will be assumed.
 */


#include "io_helpers.h"
#include "types.h"
#include "branch_length_estimation.h"
#include "pairing_algorithms.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <assert.h>
#include <random>


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

std::vector<double> get_weights_for_initial_site_rates(
    const std::vector<double>& rate_categories
) {
    // get the cuttoffs for each bin based on the weight for that rate category. 
    // the weight of the curve is the cdf of gamma(shape=3, scale=1/3) from the midpoint of the previous
    // category to the midpoint of the next
    std::vector<double> midpoints;
    midpoints.reserve(rate_categories.size() - 1);
    for(int i = 1; i < rate_categories.size(); i++) {
        midpoints.push_back(sqrt(rate_categories[i-1]*rate_categories[i]));
    }
    std::vector<double> weights;
    double shape = 3.0;
    weights.reserve(rate_categories.size());
    for(int i = 0; i < rate_categories.size() - 1; i++) {
        weights.push_back(PGamma(midpoints[i], shape));
    }
    weights.push_back(1.0);
    
    weights[0] = weights[0];
    for(int r = 1; r < rate_categories.size(); r++) {
        weights[r] = weights[r];
    }
    return weights;
}


int main(int argc, char *argv[]) {
    std::unordered_map<std::string, std::string> arguments;

    // Iterate through command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];

        // Check if the key starts with a dash (indicating a named argument)
        if (key[0] == '-' && i + 1 < argc) {
            std::string value = argv[i + 1];
            arguments[key] = value;
        } else {
            std::cerr << "Invalid command line arguments." << std::endl;
            return 1; 
        }
    }

    if (arguments.count("-rate_matrix_path") == 0 ||
        arguments.count("-msa_list_path") == 0 ||
        arguments.count("-output_list_path") == 0 || 
        arguments.count("-site_rate_list_path") == 0 || 
        arguments.count("-profiling_list_path") == 0 ||
        arguments.count("-seed") == 0 ||
        arguments.count("-quantization_grid_center") == 0||
        arguments.count("-quantization_grid_step") == 0||
        arguments.count("-quantization_grid_num_steps") == 0||
        arguments.count("-num_rate_categories_ble") == 0||
        arguments.count("-max_iters_ble") == 0) {
        std::cerr << "missing args"<< std::endl;
        return 1;
    }

    // Extract named arguments
    const std::string& rate_matrix_path = arguments["-rate_matrix_path"];
    const double quantization_grid_center = stod(arguments["-quantization_grid_center"]);
    const double quantization_grid_step = stod(arguments["-quantization_grid_step"]);
    const int quantization_grid_num_steps = stoi(arguments["-quantization_grid_num_steps"]);
    
    const std::vector<std::string>& msa_paths = read_file_paths_from_list(arguments["-msa_list_path"]);
    const std::vector<std::string>& output_paths = read_file_paths_from_list(arguments["-output_list_path"]);
    const std::vector<std::string>& site_rate_paths = read_file_paths_from_list(arguments["-site_rate_list_path"]);
    const std::vector<std::string>& profiling_paths = read_file_paths_from_list(arguments["-profiling_list_path"]);
    assert(msa_paths.size() == output_paths.size());
    assert(output_paths.size() == profiling_paths.size());
    const int seed = std::stoi(arguments["-seed"]);

    const int max_iters_ble = stoi(arguments["-max_iters_ble"]);

    int num_rate_categories_ble = 1;
    if(arguments.count("-num_rate_categories_ble") > 0) {
        num_rate_categories_ble = stoi(arguments["-num_rate_categories_ble"]);
    }

    std::unordered_map<char, int> alphabet = default_alphabet;
    if(arguments.count("-alphabet_path") > 0) {
        alphabet = read_alphabet(arguments["-alphabet_path"]);
    }

    const std::vector<double>& quantization_points = compute_quantization_points(
        quantization_grid_center, 
        quantization_grid_step,
        quantization_grid_num_steps
    );

    std::vector<double> rate_categories;
    rate_categories.reserve(num_rate_categories_ble);
    double start = 1.0/num_rate_categories_ble;
    double ratio = std::pow(num_rate_categories_ble/start, 1.0/(num_rate_categories_ble - 1));
    rate_categories.push_back(start);
    for(int i = 1; i < num_rate_categories_ble; i++) {
        rate_categories.push_back(rate_categories[i-1] * ratio);
    }
    
    std::vector<double> weights_for_initial_site_rates = get_weights_for_initial_site_rates(rate_categories);

    const transition_matrices& log_transition_matrices = read_rate_compute_log_transition_matrices(rate_matrix_path, quantization_points, rate_categories, alphabet.size());

    for(int i = 0; i < msa_paths.size(); i++) {
        // Set up Mersenne Twister random number generator
        std::mt19937 rng(seed);
        // srand(seed);  // Old code with rand() - not reproducible accross different machine architectures.
        auto start_cpp = std::chrono::high_resolution_clock::now();
        std::string msa_path = msa_paths[i];
        std::string output_path = output_paths[i];
        std::string profiling_path = profiling_paths[i];
        std::string site_rate_path = site_rate_paths[i];
        msa_type names_and_map = read_msa(msa_path, alphabet);

        std::ofstream profiling_fout(profiling_path);
        std::ofstream site_rate_fout(site_rate_path);

        auto start = std::chrono::high_resolution_clock::now();

        // perform pairing using the specified likelihood function
        std::vector<std::pair<std::string, std::string> > cherries_names = divide_and_pair(
            names_and_map.all_names,
            names_and_map.names_to_sequence,
            rng
        );
        
        std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries(cherries_names.size());
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    
        profiling_fout << "pairing_time: " << duration.count()*1.0e-9 << "\n";

        start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < cherries_names.size(); i++) {
            cherries[i].first = names_and_map.names_to_sequence.at(cherries_names[i].first);
            cherries[i].second = names_and_map.names_to_sequence.at(cherries_names[i].second);
        }

        length_and_rates lr = ble(
            cherries, 
            names_and_map.all_sequences, 
            log_transition_matrices,
            quantization_points, 
            rate_categories,
            weights_for_initial_site_rates,
            max_iters_ble
        );

        // normalize the rates and the lengths
        double mean_rates = 0;
        for(double r:lr.rates) {
            mean_rates += r;
        }
        mean_rates /= lr.rates.size();
        for(int i = 0; i < lr.lengths.size(); i++) {
            lr.lengths[i] *= mean_rates;
        }
        for(int i = 0; i < lr.rates.size(); i++) {
            lr.rates[i] /= mean_rates;
        }

        write_cherries_and_distances(
            cherries_names,
            lr.lengths,
            output_path
        );
        write_site_rates(lr.rates, site_rate_path);

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        profiling_fout << "ble_time: " << duration.count()*1.0e-9 << std::endl;

        auto stop_cpp = std::chrono::high_resolution_clock::now();
        auto duration_cpp = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_cpp - start_cpp);
    
        profiling_fout << "cpp_time: " << duration_cpp.count()*1.0e-9 << "\n";
        profiling_fout.close();
    }
    return 0; 
}
