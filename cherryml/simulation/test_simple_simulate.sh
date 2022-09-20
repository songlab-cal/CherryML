#!/bin/bash

# load MPI module
module load openmpi

# Compile the code
mpicxx -o simulate simulate.cpp

# Run the testing arguments for this cpp implementation
./simulate ./../../tests/simulation_tests/test_input_data/tree_dir ./../../tests/simulation_tests/test_input_data/synthetic_site_rates_dir ./../../tests/simulation_tests/test_input_data/synthetic_contact_map_dir 1 2 ./../../tests/simulation_tests/test_input_data/normal_model/pi_1.txt ./../../tests/simulation_tests/test_input_data/normal_model/Q_1.txt ./../../tests/simulation_tests/test_input_data/normal_model/pi_2.txt ./../../tests/simulation_tests/test_input_data/normal_model/Q_2.txt all_transitions ./../../tests/simulation_tests/test_input_data/simulated_msa_dir 0 fam1 S T

