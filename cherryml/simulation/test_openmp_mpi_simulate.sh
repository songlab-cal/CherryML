#!/bin/bash

# Load MPI module
module load openmpi

# Compile the code
mpicxx -fopenmp -o simulate simulate.cpp

# Set OpenMP settings
# export OMP_PLACES=cores
# export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=4

# Run the testing arguments for this cpp implementation with mpi
srun -N 1 --ntasks-per-node=3 ./simulate ./../../tests/simulation_tests/test_input_data/tree_dir ./../../tests/simulation_tests/test_input_data/synthetic_site_rates_dir ./../../tests/simulation_tests/test_input_data/synthetic_contact_map_dir 3 2 ./../../tests/simulation_tests/test_input_data/normal_model/pi_1.txt ./../../tests/simulation_tests/test_input_data/normal_model/Q_1.txt ./../../tests/simulation_tests/test_input_data/normal_model/pi_2.txt ./../../tests/simulation_tests/test_input_data/normal_model/Q_2.txt all_transitions ./../../tests/simulation_tests/test_input_data/simulated_msa_dir 0 fam1 fam2 fam3 S T
