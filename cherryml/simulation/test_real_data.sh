#!/bin/bash

# Load MPI module
module load openmpi

# Compile the code
mpicxx -fopenmp -O3 -o simulate simulate.cpp

# Set OpenMP settings
# export OMP_PLACES=cores
# export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=4

# Run the testing arguments for this cpp implementation with mpi
srun -N 1 --ntasks-per-node=64 ./simulate \
    /global/cscratch1/sd/sprillo/cs267_data/trees_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats \
    /global/cscratch1/sd/sprillo/cs267_data/site_rates_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats \
    /global/cscratch1/sd/sprillo/cs267_data/contact_maps_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats_maximal_matching \
    4 \
    20 \
    ../../data/rate_matrices/wag_stationary.txt \
    ../../data/rate_matrices/wag.txt \
    ../../data/rate_matrices/wag_x_wag_stationary.txt \
    ../../data/rate_matrices/wag_x_wag.txt \
    all_transitions \
    /global/cscratch1/sd/sprillo/xingyu_sim_out \
    0 \
    13gs_1_A 1iah_1_A 1pvh_1_B 1wa5_1_C \
    A R N D C Q E G H I L K M F P S T W Y V
