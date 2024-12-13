/**
 * @brief Processes a list of MSA's in parallel and outputs a set of cherries for each MSA.
 *
 * @param rate_matrix_path [string] 
 *        A file path to a 20x20 rate matrix.
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
 * @param num_threads [integer] 
 *        The number of threads to be used for parallel processing of the MSA files.
 *
 * @param num_rate_categories_ble [integer] 
 *        The number of rate categories used in BLE. Default value is 1 (i.e., the WAG model).
 *
 * @param max_iters_ble [integer] 
 *        The maximum number of iterations for BLE. Default value is 50.
 */

