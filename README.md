# CherryML: Scalable Maximum Likelihood Estimation of Phylogenetic Models

To reproduce all figures in our paper, proceed as follows:

## Install requirements

First, install all required Python libraries:

```
pip install -r requirements.txt
```

To be able to use Historian, you must make sure to have these installed:

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

## Reproduce figures

You are now ready to reproduce all figures in our paper. Just run `main.py` to reproduce all figures in our paper. The approximate runtime needed to reproduce each figure this way is commented in `main.py`. To reproduce a specific figure, comment out the figures you do not want in `main.py`. The code is written in a functional style, so the functions can be run in any order at any time and will reproduce the results. All the intermediate computations are cached, so re-running the code will be very fast the second time around. The output figures will be found in the `images` folder.

Tree estimation is parallelized, so by default you will need a machine with at least 32 cores. If you would like to use more (or less) cores, modify the default values of `num_processes_tree_estimation` in the signatures of `figures.py`. (However, note that the bottleneck is not tree estimation but performing EM with Historian (Fig. 1b), which will take around 60 hours regardless.)
