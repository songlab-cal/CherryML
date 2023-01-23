# CherryML: Scalable Maximum Likelihood Estimation of Phylogenetic Models

This package implements the CherryML method as applied to:
1. The classical LG model of amino acid evolution (involving a $20 \times 20$ rate matrix), as well as
2. A model of co-evolution at protein contact sites (involving a $400 \times 400$ rate matrix).

We expect that the CherryML method will be applied to enable scalable estimation of many models in the future.

This package also enables seamless reproduction of all results in our paper.

For a quick demonstration of an end-to-end application of CherryML to real data, please check out the section "[End-to-end worked-out application: plant dataset](#end-to-end-worked-out-application:-plant-dataset)".

# Demo: CherryML applied to the LG model (runtime on a normal computer: 1 - 5 minutes)

The following command learns a rate matrix from a set of MSAs, trees, and site rates (try it out!):

```
python -m cherryml \
    --output_path learned_rate_matrix_LG.txt \
    --model_name LG \
    --msa_dir demo_data/msas \
    --tree_dir demo_data/trees \
    --site_rates_dir demo_data/site_rates \
    --cache_dir _cache_demo
```

Expected output: `learned_rate_matrix_LG.txt` contains the learned rate matrix.

Generally, the learned rate matrix is written to the `output_path`, in this case `learned_rate_matrix_LG.txt`. The directories `msa_dir`, `tree_dir`, `site_rates_dir` should contain one file per family, named `[family_name].txt`, where `[family_name]` is the name of the family. Check out the contents of the directories in `demo_data/` for an example; this demo data is based on real data obtained from Pfam. The format of the files in these directories should be as follows:

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

The `cache_dir` is used to store intermediate computations for future runs. Caching is transparent to the user; just make sure to use a different `cache_dir` for different datasets. If not provided, a temporary directory will be used as the caching directory (in which case all intermediate data will be lost).

If you have _not_ estimated trees and site rates already, CherryML will do that for you using FastTree. You can simply omit the `tree_dir` and `site_rates_dir` arguments and run:

```
python -m cherryml \
    --output_path learned_rate_matrix_LG.txt \
    --model_name LG \
    --msa_dir demo_data/msas \
    --cache_dir _cache_demo \
    --num_processes_tree_estimation 32
```

Expected output: `learned_rate_matrix_LG.txt` contains the learned rate matrix.

FastTree will by default be run with 20 rate categories and with the LG rate matrix (this can be changed by using the full API described later). The trees estimated with FastTree will be saved in the `cache_dir`, and will be re-used by CherryML if they are needed in future runs with the same data (for example when learning a rate matrix on a subset of families, as done via the `--families` argument, or when learning a rate matrix on a subset of sites, as done via the `--sites_subset_dir` argument; clearly trees do not need to be re-estimated in this case!). The argument `--num_processes_tree_estimation` is used to parallelize tree estimation. In the example above, 32 processes are used.

To learn a rate matrix on only a subset of sites from each family (for example, when learning a domain-specific or structure-specific rate matrix), you can provide the indices of the sites used for each family with the `--sites_subset_dir` argument. Each file in `sites_subset_dir` should list the sites (0-based) for a family following the format in the following toy example:
```
3 sites
0 1 4
```

The CherryML API provides control over many aspects of the rate estimation process, such as the number of processes used to parallelize tree estimation, the number of rounds used to iterate between tree estimation and rate estimation, among others. These options are all described below or by running `python -m cherryml --help`.

# Demo: CherryML applied to the co-evolution model (runtime on a normal computer: 1 - 5 minutes)

To learn a coevolution model, you just need to set `--model_name co-evolution` and provide the directory with the contact maps:

```
python -m cherryml \
    --output_path learned_rate_matrix_co-evolution.txt \
    --model_name co-evolution \
    --msa_dir demo_data/msas \
    --contact_map_dir demo_data/contact_maps \
    --tree_dir demo_data/trees \
    --site_rates_dir demo_data/site_rates \
    --cache_dir _cache_demo \
    --num_epochs 10
```

We use 10 epochs in this demo so that it runs faster, but in practice we use 500 epochs. The speed of training will depend on the computer architecture.

Expected output: `learned_rate_matrix_co-evolution.txt` contains the learned rate matrix.

Each file in `contact_map_dir` should list the contact map for a family following the format in the following toy example:
```
5 sites
10101
01110
11110
01111
10011
```

As before, if you have not estimated trees already, you can omit the `tree_dir` and CherryML will estimate these for you. (In this case, we recommend using `--num_rate_categories 1` since the coevolution model does not model site rate variation.)

# Full API

The CherryML API provides extensive functionality through additional flags, which we describe below (this is shown when running `python -m cherryml --help`):

```
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Filepath where to write the learned rate matrix (default:
                        None)
  --model_name MODEL_NAME
                        Either "LG" or "co-evolution". If "LG", a 20x20 rate
                        matrix will be learned. If "co-evolution", a 400x400 rate
                        matrix will be learned. (default: None)
  --msa_dir MSA_DIR     Directory where the training multiple sequence alignments
                        (MSAs) are stored. See README at
                        https://github.com/songlab-cal/CherryML for the expected
                        format of these files. (default: None)
  --contact_map_dir CONTACT_MAP_DIR
                        Directory where the contact maps are stored. See README
                        at https://github.com/songlab-cal/CherryML for the
                        expected format of these files. (default: None)
  --tree_dir TREE_DIR   Directory where the trees are stored. See README at
                        https://github.com/songlab-cal/CherryML for the expected
                        format of these files. If not provided, trees will be
                        estimated with FastTree. (default: None)
  --site_rates_dir SITE_RATES_DIR
                        Directory where the site rates are stored. See README at
                        https://github.com/songlab-cal/CherryML for the expected
                        format of these files. If not provided, site rates will
                        be estimated with FastTree. (default: None)
  --cache_dir CACHE_DIR
                        Directory to use to cache intermediate computations for
                        re-use in future runs of cherryml. Use a different cache
                        directory for different input datasets. If not provided,
                        a temporary directory will be used. (default: None)
  --num_processes_tree_estimation NUM_PROCESSES_TREE_ESTIMATION
                        Number of processes to parallelize tree estimation (with
                        FastTree). (default: 32)
  --num_processes_counting NUM_PROCESSES_COUNTING
                        Number of processes to parallelize counting transitions.
                        (default: 1)
  --num_processes_optimization NUM_PROCESSES_OPTIMIZATION
                        Number of processes to parallelize optimization (if using
                        cpu). (default: 1)
  --num_rate_categories NUM_RATE_CATEGORIES
                        Number of rate categories to use in FastTree to estimate
                        trees and site rates (if trees are not provided).
                        (default: 20)
  --initial_tree_estimator_rate_matrix_path INITIAL_TREE_ESTIMATOR_RATE_MATRIX_PATH
                        Rate matrix to use in FastTree to estimate trees and site
                        rates (the first time around, and only if trees and site
                        rates are not provided) (default:
                        data/rate_matrices/lg.txt)
  --num_iterations NUM_ITERATIONS
                        Number of times to iterate tree estimation and rate
                        matrix estimation. For highly accurate rate matrix
                        estimation this is a good idea, although tree
                        reconstruction becomes the bottleneck. (default: 1)
  --quantization_grid_center QUANTIZATION_GRID_CENTER
                        The center value used for time quantization. (default:
                        0.03)
  --quantization_grid_step QUANTIZATION_GRID_STEP
                        The geometric spacing between time quantization points.
                        (default: 1.1)
  --quantization_grid_num_steps QUANTIZATION_GRID_NUM_STEPS
                        The number of quantization points to the left and right
                        of the center. (default: 64)
  --use_cpp_counting_implementation USE_CPP_COUNTING_IMPLEMENTATION
                        Whether to use C++ MPI implementation to count
                        transitions ('True' or 'False'). This requires mpirun to
                        be installed. If you do not have mpirun installed, set
                        this argument to False to use a Python implementation
                        (but it will be much slower). (default: True)
  --optimizer_device OPTIMIZER_DEVICE
                        Either "cpu" or "cuda". Device to use in PyTorch. "cpu"
                        is fast enough for applications, but if you have a GPU
                        using "cuda" might provide faster runtime. (default: cpu)
  --learning_rate LEARNING_RATE
                        The learning rate in the PyTorch optimizer. (default:
                        0.1)
  --num_epochs NUM_EPOCHS
                        The number of epochs of the PyTorch optimizer. (default:
                        500)
  --minimum_distance_for_nontrivial_contact MINIMUM_DISTANCE_FOR_NONTRIVIAL_CONTACT
                        Minimum distance in primary structure used to determine
                        if two site are in non-trivial contact. (default: 7)
  --families FAMILIES   Subset of families on which to run rate matrix
                        estimation. (default: None)
  --sites_subset_dir SITES_SUBSET_DIR
                        Directory where the subset of sites from each family used
                        to learn the rate matrix are specified. Currently only
                        implemented for the LG model. This enables learning e.g.
                        domain-specific or structure-specific rate matrices. See
                        README at https://github.com/songlab-cal/CherryML for the
                        expected format of these files. (default: None)
  --tree_estimator_name TREE_ESTIMATOR_NAME
                        Tree estimator to use. Can be either 'FastTree' or
                        'PhyML'. (default: FastTree)
```

# Evaluation API

For the purpose of model selection, we have exposed a simple API enabling the evaluation of a rate matrix's fit to a set of MSAs. This API is a simple wrapper around the FastTree and PhyML programs. For example, to compute the fit of the LG rate matrix to the 3 MSAs under `tests/evaluation_tests/a3m_small`, you can simply run:

```
python -m cherryml.evaluation \
    --msa_dir tests/evaluation_tests/a3m_small \
    --rate_matrix_path data/rate_matrices/lg.txt \
    --num_rate_categories 4 \
    --output_path log_likelihoods.txt \
    --cache_dir _cache_demo \
    --num_processes_tree_estimation 3 \
    --tree_estimator_name FastTree
```

The output - written to the file `log_likelihoods.txt` - looks like this:

```
Total log-likelihood: -700.1151
Total number of sites: 48
Average log-likelihood per site: -14.58573125
Families: 1e7l_1_A 5a0l_1_A 6anz_1_B
Log-likelihood per family: -198.2552 -216.9863 -284.8736
Sites per family: 16 16 16
```

The first line indicates the total log-likelihood (over all families). The second line indicates the total number of sites across all the provided MSAs. The third line shows the average log-likelihood per site (which is the ratio of the two previous quantities). The fourth line lists the families which were used to compute the log-likelihood. Next, the log-likelihood for each family is shown. Finally, the number of sites in each family is shown. Note that the total log-likelihood is equal to the sum of the log-likelihoods of each family.

Note that by default, FastTree computes log-likelihoods under MLE rates. To compute log-likelihoods under a Gamma model, provide this option through the `--extra_command_line_args` argument. Thus, to compute Gamma log-likelihoods, you can use:

```
python -m cherryml.evaluation \
    --msa_dir tests/evaluation_tests/a3m_small \
    --rate_matrix_path data/rate_matrices/lg.txt \
    --num_rate_categories 4 \
    --output_path log_likelihoods.txt \
    --cache_dir _cache_demo \
    --num_processes_tree_estimation 3 \
    --tree_estimator_name FastTree \
    --extra_command_line_args='-gamma'
```

In this case, the output is:

```
Total log-likelihood: -723.442
Total number of sites: 48
Average log-likelihood per site: -15.071708333333333
Families: 1e7l_1_A 5a0l_1_A 6anz_1_B
Log-likelihood per family: -205.047 -225.683 -292.712
Sites per family: 16 16 16
```

As you can see, log-likelihoods are lower under the Gamma model because this model accounts for the possibility that the site rates could have been something else other than the MLE rates.

We can also use PhyML instead of FastTree. PhyML computes likelihoods under a Gamma model by default. Generally, PhyML is slower than FastTree but more precise:

```
python -m cherryml.evaluation \
    --msa_dir tests/evaluation_tests/a3m_small \
    --rate_matrix_path data/rate_matrices/lg.txt \
    --num_rate_categories 4 \
    --output_path log_likelihoods.txt \
    --cache_dir _cache_demo \
    --num_processes_tree_estimation 3 \
    --tree_estimator_name PhyML
```

The output is:

```
Total log-likelihood: -717.50699
Total number of sites: 48
Average log-likelihood per site: -14.948062291666666
Families: 1e7l_1_A 5a0l_1_A 6anz_1_B
Log-likelihood per family: -204.41537 -223.26711 -289.82451
Sites per family: 16 16 16
```

You will note that PhyML obtained a better Gamma log-likelihood than FastTree. Unless over-ridden by the user with `--extra_command_line_args`, PhyML is being run with the extra command line arguments `--datatype aa --pinv e --r_seed 0 --bootstrap 0 -f m --alpha e --print_site_lnl`.

# Full API

The command line tool can be invoked with `python -m cherryml.evaluation` and accepts the following arguments:

```
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Filepath where to write the log-likelihood (default:
                        None)
  --rate_matrix_path RATE_MATRIX_PATH
                        Filepath where the rate matrix to evaluate is stored.
                        (default: None)
  --msa_dir MSA_DIR     Directory where the multiple sequence alignments
                        (MSAs) are stored. See README at
                        https://github.com/songlab-cal/CherryML for the
                        expected format of these files. (default: None)
  --cache_dir CACHE_DIR
                        Directory to use to cache intermediate computations
                        for re-use in future runs of cherryml. Use a different
                        cache directory for different input datasets. If not
                        provided, a temporary directory will be used.
                        (default: None)
  --num_processes_tree_estimation NUM_PROCESSES_TREE_ESTIMATION
                        Number of processes to parallelize tree estimation.
                        (default: 32)
  --num_rate_categories NUM_RATE_CATEGORIES
                        Number of rate categories to use in the tree estimator
                        to estimate trees and site rates. (default: 20)
  --families [FAMILIES [FAMILIES ...]]
                        Subset of families for which to evaluate log
                        likelihood. If not provided, all families in the
                        `msa_dir` will be used. (default: None)
  --tree_estimator_name TREE_ESTIMATOR_NAME
                        Tree estimator to use. Can be either 'FastTree' or
                        'PhyML'. (default: FastTree)
  --extra_command_line_args EXTRA_COMMAND_LINE_ARGS
                        Extra command line arguments for the tree estimator,
                        e.g. `-gamma` for FastTree to compute Gamma
                        likelihoods. (default: None)
```

# End-to-end worked-out application: plant dataset

We now combine the model estimation step and model selection steps to show a concrete example of applying CherryML to obtain a rate matrix superior than LG in record time. For this, we will use the plant dataset from Ran et al. (2018), `Phylogenomics resolves the deep phylogeny of seed plants and indicates partial convergent or homoplastic evolution between Gnetales and angiosperms`, with the train-test splits in the QMaker paper. The training MSAs are located at `demo_data/plant_train` and the testing MSAs are located at `demo_data/plant_test`. We start by fitting the LG model using FastTree tree estimator and the CherryML rate matrix optimizer. We start from the LG rate matrix and perform two rounds of alternating rate matrix and tree optimization (which is usually enough for convergence when adjusting the LG rate matrix to a new dataset). We will use 4 CPU cores in this example, as when running on a personal computer:

```
time python -m cherryml \
    --output_path plant_CherryML.txt \
    --model_name LG \
    --msa_dir demo_data/plant_train \
    --cache_dir _cache_plant \
    --num_processes_tree_estimation 4 \
    --num_processes_counting 4 \
    --num_processes_optimization 2 \
    --num_rate_categories 4 \
    --initial_tree_estimator_rate_matrix_path data/rate_matrices/lg.txt \
    --num_iterations 2 \
    --tree_estimator_name FastTree
```

<!-- ```
real	21m55.722s
user	79m8.700s
sys	1m52.647s
``` -->

End-to-end rate matrix estimation took 22 minutes wall-clock time on a MacBook Pro with the following specs:

```
Processor: 2.6 GHz 6-Core Intel Core i7
Memory: 16 GB 2400 MHz DDR4
```

Now we proceed to evaluate model fit on held-out data. The testing MSAs are located at `demo_data/plant_test`. Thus:

```
time python -m cherryml.evaluation \
    --msa_dir demo_data/plant_test \
    --rate_matrix_path plant_CherryML.txt \
    --num_rate_categories 4 \
    --output_path log_likelihoods_plant_CherryML.txt \
    --cache_dir _cache_plant \
    --num_processes_tree_estimation 4 \
    --tree_estimator_name FastTree
```

<!-- ```
real	3m1.096s
user	10m45.304s
sys	0m17.869s
``` -->

Evaluation took 3 minutes wall-clock time on the same computer. The output is:

```
Total log-likelihood: -2042877.196799998
Total number of sites: 101064
Average log-likelihood per site: -20.21369821895035
[...]
```

Finally, we compute the model fit of the LG rate matrix:

```
time python -m cherryml.evaluation \
    --msa_dir demo_data/plant_test \
    --rate_matrix_path data/rate_matrices/lg.txt \
    --num_rate_categories 4 \
    --output_path log_likelihoods_plant_LG.txt \
    --cache_dir _cache_plant \
    --num_processes_tree_estimation 4 \
    --tree_estimator_name FastTree
```

<!-- ```
real	3m41.567s
user	13m19.375s
sys	0m20.158s
``` -->

Evaluation took 4 minutes wall-clock time. The output is:

```
Total log-likelihood: -2072516.731100001
Total number of sites: 101064
Average log-likelihood per site: -20.50697311703476
[...]
```

As we can see, the de-novo estimated rate matrix outperforms the LG rate matrix, with an average increase in log-likelihood per site of `0.293` (1.4%).

# Reproducing all figures in our paper

To reproduce all figures in our paper, proceed as described below. Please note that this will not work in the compute capsule associated with this work since memory and compute are limited in the capsule. To reproduce all figures, you will need a machine with 32 CPU cores and 150G of storage; the Pfam dataset is large and we are in the realm of high-performance computing, which is out of reach with a compute capsule.

## Demo: Reproducing a simplified version of Figure 1e (runtime on a normal computer: ~10 minutes)

Nonetheless, in the compute capsule we reproduce a simplified version of Fig. 1e, using FastTree instead of PhyML to evaluate likelihoods - and excluding EM since it is very slow (takes ~12 hours to train) - as follows:

```
time python reproduce_fig_1e_simplified_demo.py
```

Expected output: `fig_1e_simplified/` contains the reproduced version of Fig. 1e (without EM).

FastTree is faster, which is better for the demo, and the results are similar. Reproducing Fig. 1e (excluding EM) with FastTree takes ~10 minutes. Using PhyML (as in `reproduce_all_figures.py`, and as in our paper), would take ~4 hours. Note that if you have less than 32 cores available, you should change `num_processes=32` to a different value in `reproduce_fig_1e_simplified_demo.py`. In this case, it will take longer than ~10 minutes.

## System requirements

CherryML has been tested on an Ubuntu system (20.04) with Python (3.8.5, miniconda 4.9.2).

First, install all required Python libraries, e.g.:

```
pip install -r requirements.txt
```

The following are system requirements:

```
autoconf
automake
gcc
libboost-dev
libboost-regex-dev
libgsl-dev
libopenmpi-dev
mpich
pkg-config
wget
zlib1g-dev
```

All third-party software, including FastTree (`FastTree` program), PhyML (`phyml` program), and XRATE (`xrate` program), will be automatically installed locally into this repository by our code if you have not installed it already on your system. If you would like to install these third-party tools on your system, you can do e.g.:

To install FastTree (again, this is optional, we will install FastTree locally otherwise):
```
mkdir -p /opt/FastTree/bin/
mkdir -p /opt/FastTree/download/
export PATH=/opt/FastTree/bin:$PATH
wget http://www.microbesonline.org/fasttree/FastTree.c -P /opt/FastTree/download/
gcc -DNO_SSE -DUSE_DOUBLE -O3 -finline-functions -funroll-loops -Wall \
    -o /opt/FastTree/bin/FastTree /opt/FastTree/download/FastTree.c -lm
```

To install PhyML (again, this is optional, we will install PhyML locally otherwise):
```
mkdir -p /opt/phyml/bin/
mkdir -p /opt/phyml/download/
export PATH=/opt/phyml/bin:$PATH
git clone https://github.com/stephaneguindon/phyml /opt/phyml/download/phyml
pushd /opt/phyml/download/phyml/
bash ./autogen.sh
./configure --enable-phyml --prefix=/opt/phyml/bin/
make
make install
popd
```

To install XRATE (again, this is optional, we will install XRATE locally otherwise):
```
mkdir -p /opt/xrate/bin/
mkdir -p /opt/xrate/download/
export PATH=/opt/xrate/bin:$PATH
git clone https://github.com/ihh/dart /opt/xrate/download/xrate
pushd /opt/xrate/download/xrate/
./configure --without-guile
make xrate
cp /opt/xrate/download/xrate/bin/xrate /opt/xrate/bin/xrate
popd
```

Once you have met all the requirements, run the fast tests to make sure they pass:

```
python -m pytest tests
```

The run _all_ tests (including the slow tests, such as those for PhyML), and make sure they pass:

```
python -m pytest tests --runslow
```

This should take a few minutes. If all tests pass, you are good to go. You can install the `cherryml` package for future use in other projects by running `pip install .`. Then you can use it with `python -m cherryml [...]` as in the demo above.

## Download data

Once all tests are passing, you will need to download the data from the trRosetta paper into this repository, which is available at the following link:

https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz

After downloading and untarring the data into this repository, rename the `training_set` directory to `input_data`.

You do not need to worry about downloading the data from the LG paper - we will download this automatically for you. Similarly, we will download the QMaker datasets.

## Run code to reproduce all figures

You are now ready to reproduce all figures in our paper. Just run `reproduce_all_figures.py` to reproduce all figures in our paper. The approximate runtime needed to reproduce each figure this way is commented in `reproduce_all_figures.py`. Note that the computational bottlenecks to reproduce all figures are (1) benchmarking EM with XRATE and (2) tree estimation (as opposed to the CherryML optimizer). To reproduce a specific figure, comment out the figures you do not want in `reproduce_all_figures.py`. The code is written in a functional style, so the functions can be run in any order at any time and will reproduce the results. All the intermediate computations are cached, so re-running the code will be very fast the second time around. The output figures will be found in the `images` folder.

Tree estimation is parallelized, so by default you will need a machine with at least 32 cores. If you would like to use more (or less) cores, modify the value of `NUM_PROCESSES_TREE_ESTIMATION` at the top of the `figures.py` module. (However, note that the bottleneck when reproducing all figures is not tree estimation but performing EM with XRATE (Fig. 1b and Supp Fig. 1), which will take around 3-4 days regardless.)
