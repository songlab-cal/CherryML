# CherryML: Scalable Maximum Likelihood Estimation of Phylogenetic Models

This package implements the CherryML method as applied to:
1. The classical LG model of amino acid evolution (involving a $20 \times 20$ rate matrix), as well as
2. A model of co-evolution at protein contact sites (involving a $400 \times 400$ rate matrix).

We expect that the CherryML method will be applied to enable scalable estimation of many models in the future.

This package also enables seamless reproduction of all results in our paper.

# Demo: CherryML applied to the LG model

The following command learns a rate matrix from a set of MSAs, trees, and site rates (try it out!):

```
python -m cherryml \
    --output_path learned_rate_matrix.txt \
    --model_name LG \
    --msa_dir demo_data/msas \
    --tree_dir demo_data/trees \
    --site_rates_dir demo_data/site_rates \
    --cache_dir _cache_demo
```

The learned rate matrix is written to the `output_path`, in this case `learned_rate_matrix.txt`. The directories `msa_dir`, `tree_dir`, `site_rates_dir` should contain one file per family, named `[family_name].txt`, where `[family_name]` is the name of the family. Check out the contents of the directories in `demo_data/` for an example; this demo data is based on real data obtained from Pfam. The format of the files in these directories should be as follows:

Each file in `msa_dir` should list the protein sequences in a family following the format in the following toy example:
```
>seq1
TTLLS
>seq2
TTIIS
>seq3
SSIIS
```
All sequences should have the same length.

Each file in `tree_dir` should list the tree for a family following the format in the following toy example:
```
6 nodes
internal-0
internal-1
internal-2
seq1
seq2
seq3
5 edges
internal-0 internal-1 1.0
internal-1 internal-2 2.0
internal-2 seq1 3.0
internal-2 seq2 4.0
internal-1 seq3 5.0
```
This format is intended to be easier to parse than the newick format. It first lists the nodes in the tree, and then the edges in the tree with their lengths.

Each file in `site_rates_dir` should list the site rates for a family following the format in the following toy example:
```
5 sites
1.0 0.8 1.2 0.7 1.05
```

The `cache_dir` is used to store intermediate computations for future runs. Caching is transparent to the user; just make sure to use a different `cache_dir` for different datasets. If not provided, a temporary directory will be used as the caching directory (so, all intermediate data will be lost).

If you have not estimated trees and site rates already, cherryml will do that for you using FastTree. You can simply run:

```
python -m cherryml \
    --output_path learned_rate_matrix.txt \
    --model_name LG \
    --msa_dir demo_data/msas \
    --cache_dir _cache_demo
```

FastTree will be run with 20 rate categories and with the LG rate matrix. Briefly, all intermediate data, such as the trees estimated with FastTree will be saved in the `cache_dir`, and will be re-used by CherryML if they are needed in the future.

The cherryml API provides control over many aspects of the rate estimation process, such as the number of processes used to parallelize tree estimation, the number of rounds used to iterate tree estiation and rate estimation, among others. These options are all described below or by running `python -m cherryml --help`.

# Demo: CherryML applied to the co-evolution model

To learn a coevolution model, you just need to set `--model_name co-evolution` and provide the directory with the contact maps:

```
python -m cherryml \
    --output_path learned_rate_matrix.txt \
    --model_name co-evolution \
    --msa_dir demo_data/msas \
    --contact_map_dir demo_data/contact_maps \
    --tree_dir demo_data/trees \
    --site_rates_dir demo_data/site_rates \
    --cache_dir _cache_demo
```

Each file in `contact_map_dir` should list the contact map for a family following the format in the following toy example:
```
5 sites
10101
01110
11110
01111
10011
```

As before, if you have not estimated trees already, you can omit the `tree_dir` and cherryml will estimate these for you. (In this case, we recommend using `--num_rate_categories 1` since the coevolution model does not model site rate variation.)

# Full API

The CherryML API provides extensive functionality through additional flags, which we describe below (this is shown when running `python -m cherryml --help`):

```
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Filepath where to write the learned rate matrix (default: None)
  --model_name MODEL_NAME
                        Either "LG" or "co-evolution". If "LG", a 20x20 rate matrix will be learned. If "co-evolution", a 400x400 rate matrix will be learned.
                        (default: None)
  --msa_dir MSA_DIR     Directory where the training multiple sequence alignments (MSAs) are stored. See README at https://github.com/songlab-cal/CherryML for
                        the expected format of these files. (default: None)
  --contact_map_dir CONTACT_MAP_DIR
                        Directory where the contact maps are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these
                        files. (default: None)
  --tree_dir TREE_DIR   Directory where the trees are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files. If
                        not provided, trees will be estimated with FastTree. (default: None)
  --site_rates_dir SITE_RATES_DIR
                        Directory where the site rates are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files.
                        If not provided, site rates will be estimated with FastTree. (default: None)
  --cache_dir CACHE_DIR
                        Directory to use to cache intermediate computations for re-use in future runs of cherryml. Use a different cache directory for different
                        input datasets. If not provided, a temporary directory will be used. (default: None)
  --num_processes_tree_estimation NUM_PROCESSES_TREE_ESTIMATION
                        Number of processes to parallelize tree estimation (with FastTree). (default: 1)
  --num_processes_counting NUM_PROCESSES_COUNTING
                        Number of processes to parallelize counting transitions. (default: 1)
  --num_processes_optimization NUM_PROCESSES_OPTIMIZATION
                        Number of processes to parallelize optimization (if using cpu). (default: 1)
  --num_rate_categories NUM_RATE_CATEGORIES
                        Number of rate categories to use in FastTree to estimate trees and site rates (if trees are not provided). (default: 20)
  --initial_tree_estimator_rate_matrix_path INITIAL_TREE_ESTIMATOR_RATE_MATRIX_PATH
                        Rate matrix to use in FastTree to estimate trees and site rates (the first time around, and only if trees and site rates are not
                        provided) (default: data/rate_matrices/lg.txt)
  --num_iterations NUM_ITERATIONS
                        Number of times to iterate tree estimation and rate matrix estimation. For highly accurate rate matrix estimation this is a good idea,
                        although tree reconstruction becomes the bottleneck. (default: 1)
  --quantization_grid_center QUANTIZATION_GRID_CENTER
                        The center value used for time quantization. (default: 0.03)
  --quantization_grid_step QUANTIZATION_GRID_STEP
                        The geometric spacing between time quantization points. (default: 1.1)
  --quantization_grid_num_steps QUANTIZATION_GRID_NUM_STEPS
                        The number of quantization points to the left and right of the center. (default: 64)
  --use_cpp_counting_implementation USE_CPP_COUNTING_IMPLEMENTATION
                        Whether to use C++ MPI implementation to count transitions. This requires mpirun to be installed. If you do not have mpirun installed,
                        set this argument to False to use a Python implementation (but it will be much slower). (default: True)
  --optimizer_device OPTIMIZER_DEVICE
                        Either "cpu" or "cuda". Device to use in PyTorch. "cpu" is fast enough for applications, but if you have a GPU using "cuda" might
                        provide faster runtime. (default: cpu)
  --learning_rate LEARNING_RATE
                        The learning rate in the PyTorch optimizer. (default: 0.1)
  --num_epochs NUM_EPOCHS
                        The number of epochs of the PyTorch optimizer. (default: 500)
  --minimum_distance_for_nontrivial_contact MINIMUM_DISTANCE_FOR_NONTRIVIAL_CONTACT
                        Minimum distance in primary structure used to determine if two site are in non-trivial contact. (default: 7)
```

# Reproducing all figures in our paper

To reproduce all figures in our paper, proceed as described below. Please note that this will not work in the compute capsule associated with this work since memory and compute are limited in the capsule. To reproduce all figures, you will need a machine with 32 CPU cores and TODO gigs of memory. Indeed, the Pfam dataset is very large and we are in the realm of high-performance computing, which is not feasible in the capsule.

## Install requirements

First, install all required Python libraries:

```
pip install -r requirements.txt
```

To be able to use Historian (required for the results on EM), you must make sure to have these installed:

On a Mac:
```
brew install boost gsl pkg-config zlib
```

On Linux:
```
sudo yum -y install boost-devel gsl-devel zlib
```

(Generally, the requirements for Historian are specified in https://github.com/evoldoers/historian .)

All third-party software, including FastTree, PhyML, and Historian, will be automatically installed locally into this repository by our code. Just run all tests to make sure that they are passing:

```
python -m pytest tests
```

## Download data

Once all tests are passing, you will need to download the data from the trRosetta paper into this repository, which is available at the following link:

https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz

After downloading and untarring the data into this repository, rename the `training_set` directory to `input_data`.

You do not need to worry about downloading the data from the LG paper - we will download this automatically for you.

## Run code to reproduce figures

You are now ready to reproduce all figures in our paper. Just run `main.py` to reproduce all figures in our paper. The approximate runtime needed to reproduce each figure this way is commented in `main.py`. To reproduce a specific figure, comment out the figures you do not want in `main.py`. The code is written in a functional style, so the functions can be run in any order at any time and will reproduce the results. All the intermediate computations are cached, so re-running the code will be very fast the second time around. The output figures will be found in the `images` folder.

Tree estimation is parallelized, so by default you will need a machine with at least 32 cores. If you would like to use more (or less) cores, modify the default values of `num_processes_tree_estimation` in the signatures of `figures.py`. (However, note that the bottleneck is not tree estimation but performing EM with Historian (Fig. 1b), which will take around 60 hours regardless.)
