import time
import os
from cherryml import io as cherryml_io
from cherryml import markov_chain
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from ete3 import Tree as TreeETE
from cherryml import caching
from scipy.stats import gamma
import logging
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
from cherryml._siterm._learn_site_rate_matrix import learn_site_rate_matrices
from cherryml._siterm._learn_site_rate_matrix import get_standard_site_rate_grid, get_standard_site_rate_prior, _get_msa_example, _get_test_msa, _get_test_tree, _get_tree_newick, _convert_newick_to_CherryML_Tree


def learn_site_specific_rate_matrices(
    tree_newick: Optional[str],
    msa: Dict[str, str],  # TODO: Allow path to MSA too.
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float = 0.5,
    device: str = "cpu",
    num_rate_categories: int = 20,
    alphabet_for_site_rate_estimation: Optional[List[str]] = None,
    rate_matrix_for_site_rate_estimation: Optional[pd.DataFrame] = None,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = 64,
) -> Dict:
    """
    Learn a rate matrix per site given the tree and leaf states.

    This function implements learning under the SiteRM model. Briefly, the
    SiteRM model uses a different rate matrix per site. It is described in
    detail in our paper:

    ```
    Prillo, S., Wu, W., Song, Y.S.  (NeurIPS 2024) Ultrafast classical
    phylogenetic method beats large protein language models on variant
    effect prediction.
    ```

    Args:
        tree_newick: The tree in newick format. If `None`, then FastCherries
            will be used to estimate the tree (cherries).
        msa: For each leaf in the tree, the states for that leaf.
        alphabet: List of valid states, e.g. ["A", "C", "G", "T"].
        regularization_rate_matrix: Rate matrix to use to regularize the learnt
            rate matrix.
        regularization_strength: Between 0 and 1. 0 means no regularization at all,
            and 1 means fully regularized model.
        device: Whether to use "cpu" or GPU ("cuda").
        num_rate_categories: Grid of site rates to consider. The standard site
            rate grid (used in FastCherries/SiteRM) can be obtained with
            `get_standard_site_rate_grid()`.
        alphabet_for_site_rate_estimation: Alphabet for learning the SITE RATES.
            If `None`, then the alphabet for the learnt rate matrix, i.e.
            `alphabet`, will be used. In the FastCherries/SiteRM paper, we have
            observed that for ProteinGym variant effect prediction, it works
            best to *exclude* gaps while estimating site rates (as is standard
            in statistical phylogenetics), but then use gaps when learning the
            rate matrix at the site. In that case, one would use
            `alphabet_for_site_rate_estimation=["A", "C", "G", "T"]` in
            combination with `alphabet=["A", "C", "G", "T", "-"]`.
        rate_matrix_for_site_rate_estimation: If provided, the rate matrix to
            use to estimate site rates. If `None`, then the
            `regularization_rate_matrix` will be used.
        num_epochs: Number of epochs (Adam steps) in the PyTorch optimizer.
        return_pandas_dataframes: If False, then a 3D numpy array will be returned, which
            is much faster than returning a list of pandas dataframes. (Pandas is so
            slow!)
        quantization_grid_num_steps: Number of quantization points to use will
            be `2 * quantization_grid_num_steps + 1` (we take this number of
            steps left and right of the grid center). Lowering
            `quantization_grid_num_steps` leads to faster training at the
            expense of some accuracy. By default,
            `quantization_grid_num_steps=64` works really well, but reasonable
            estimates can be obtained very fast with as low as
            `quantization_grid_num_steps=8`.

    Returns:
        A dictionary with the following entries:
            - "learnt_rate_matrices": A numpy array with the learnt rate matrix per site.
            - "learnt_site_rates": A List[float] with the learnt site rate per site.
            - "time_...": The time taken by this substep. (They should add up
                to the total runtime).
    """
    site_rate_grid = get_standard_site_rate_grid(num_site_rates=num_rate_categories)
    site_rate_prior = get_standard_site_rate_prior(num_site_rates=num_rate_categories)
    res_dict = learn_site_rate_matrices(
        tree_newick=tree_newick,
        leaf_states=msa,
        alphabet=alphabet,
        regularization_rate_matrix=regularization_rate_matrix,
        regularization_strength=regularization_strength,
        use_fast_implementation=True,
        fast_implementation_device=device,
        fast_implementation_num_cores=1,  # Doesn't really make sense to vectorize at this level.
        site_rate_grid=site_rate_grid,
        site_rate_prior=site_rate_prior,
        alphabet_for_site_rate_estimation=alphabet_for_site_rate_estimation,
        rate_matrix_for_site_rate_estimation=rate_matrix_for_site_rate_estimation,
        num_epochs=num_epochs,
        use_fast_site_rate_implementation=True,
        quantization_grid_num_steps=quantization_grid_num_steps,
        return_pandas_dataframes=False,
    )
    return res_dict


def test_learn_site_specific_rate_matrices():
    res_dict = learn_site_specific_rate_matrices(
        tree_newick="(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);",
        msa={"leaf_1": "C", "leaf_2": "C", "leaf_3": "C", "leaf_4": "G"},
        alphabet=["A", "C", "G", "T"],
        regularization_rate_matrix=pd.DataFrame(
            [
                [-3.0, 1.0, 1.0, 1.0],
                [1.0, -3.0, 1.0, 1.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ) / 3.0,
        regularization_strength=0.5,
    )
    learnt_site_rates = res_dict["learnt_site_rates"]
    np.testing.assert_almost_equal(learnt_site_rates, [0.6231236])
    site_rate_matrices = res_dict["learnt_rate_matrices"]
    np.testing.assert_array_almost_equal(
        site_rate_matrices[0],
        pd.DataFrame(
            [
                [-0.48, 0.03, 0.24, 0.21],
                [ 0.01, -0.62, 0.6, 0.01],
                [ 0.12, 1.22, -1.47, 0.12],
                [ 0.21, 0.03, 0.24, -0.48],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).to_numpy(),
        decimal=1,
    )


def test_learn_site_specific_rate_matrices_large():
    tree_newick = _get_tree_newick()
    leaves = _convert_newick_to_CherryML_Tree(tree_newick).leaves()
    alphabet = ["A", "C", "G", "T"]
    site_rate_matrices = learn_site_specific_rate_matrices(
        tree_newick=tree_newick,
        # msa={leaf: "A" for leaf in leaves},  # Gives very slow rate matrix
        msa={**{"hg38": "A"}, **{leaf: "C" for leaf in leaves if leaf != "hg38"}},  # Gives reasonable rate matrix
        # msa={leaf: alphabet[len(leaf) % 4] for leaf in leaves},  # Gives rate matrix with massive rates.
        alphabet=alphabet,
        regularization_rate_matrix=pd.DataFrame(
            [
                [-3.0, 1.0, 1.0, 1.0],
                [1.0, -3.0, 1.0, 1.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
        regularization_strength=0.5,
    )["learnt_rate_matrices"]
    np.testing.assert_array_almost_equal(
        site_rate_matrices[0].T,
        pd.DataFrame(
            [[-0.31,  0.04,  0.05,  0.05],
             [ 0.2 , -0.06,  0.05,  0.05],
             [ 0.05,  0.01, -0.15,  0.05],
             [ 0.05,  0.01,  0.05, -0.15]],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).to_numpy(),
        decimal=2,
    )


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_vectorized_GOAT():
    """
    Tests that fast vectorized SiteRM implementation runs on real MSA.
    """
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    msa = _get_msa_example()
    leaf_states = {
        msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
        for species_id in range(msa.shape[1])
    }
    tree_newick = _get_tree_newick()
    st = time.time()
    res_dict = learn_site_specific_rate_matrices(
        tree_newick=tree_newick,
        msa={
            k: v[:num_sites]
            for (k, v) in leaf_states.items()
        },
        alphabet=["A", "C", "G", "T", "-"],
        regularization_rate_matrix=pd.DataFrame(
            [
                [-1.0, 0.25, 0.25, 0.25, 0.25],
                [0.25, -1.0, 0.25, 0.25, 0.25],
                [0.25, 0.25, -1.0, 0.25, 0.25],
                [0.25, 0.25, 0.25, -1.0, 0.25],
                [0.25, 0.25, 0.25, 0.25, -1.0],
            ],
            index=["A", "C", "G", "T", "-"],
            columns=["A", "C", "G", "T", "-"],
        ),
        regularization_strength=0.5,
        alphabet_for_site_rate_estimation=["A", "C", "G", "T"],
        rate_matrix_for_site_rate_estimation=pd.DataFrame(
            [
                [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
        num_epochs=200,
    )


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fastcherries():
    """
    Same as above but we infer the tree instead.
    """
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    msa = _get_msa_example()
    leaf_states = {
        msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
        for species_id in range(msa.shape[1])
    }
    tree_newick = None  # (instead of _get_tree_newick())
    st = time.time()
    res_dict = learn_site_specific_rate_matrices(
        tree_newick=tree_newick,
        msa={
            k: v[:num_sites]
            for (k, v) in leaf_states.items()
        },
        alphabet=["A", "C", "G", "T", "-"],
        regularization_rate_matrix=pd.DataFrame(
            [
                [-1.0, 0.25, 0.25, 0.25, 0.25],
                [0.25, -1.0, 0.25, 0.25, 0.25],
                [0.25, 0.25, -1.0, 0.25, 0.25],
                [0.25, 0.25, 0.25, -1.0, 0.25],
                [0.25, 0.25, 0.25, 0.25, -1.0],
            ],
            index=["A", "C", "G", "T", "-"],
            columns=["A", "C", "G", "T", "-"],
        ),
        regularization_strength=0.5,
        alphabet_for_site_rate_estimation=["A", "C", "G", "T"],
        rate_matrix_for_site_rate_estimation=pd.DataFrame(
            [
                [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
        num_epochs=200,
    )


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_cuda_GOAT():
    """
    Tests that CUDA SiteRM implementation runs on real MSA.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    num_sites = 10  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    for device in ["cpu", "cuda"]:
        print(f"***** device = {device} *****")
        msa = _get_msa_example()
        leaf_states = {
            msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
            for species_id in range(msa.shape[1])
        }
        tree_newick = _get_tree_newick()
        st = time.time()
        res_dict = learn_site_rate_matrices(
            tree_newick=tree_newick,
            msa={
                k: v[:num_sites]
                for (k, v) in leaf_states.items()
            },
            alphabet=["A", "C", "G", "T", "-"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 0.25, 0.25, 0.25, 0.25],
                    [0.25, -1.0, 0.25, 0.25, 0.25],
                    [0.25, 0.25, -1.0, 0.25, 0.25],
                    [0.25, 0.25, 0.25, -1.0, 0.25],
                    [0.25, 0.25, 0.25, 0.25, -1.0],
                ],
                index=["A", "C", "G", "T", "-"],
                columns=["A", "C", "G", "T", "-"],
            ),
            regularization_strength=0.5,
            device=device,
            alphabet_for_site_rate_estimation=["A", "C", "G", "T"],
            rate_matrix_for_site_rate_estimation=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            num_epochs=200,
        )
        site_rate_matrices[device] = res_dict["learnt_rate_matrices"]
        total_time = 0.0
        for k, v in res_dict.items():
            if k.startswith("time_"):
                print(k, v)
                total_time += float(v)
        print(f"total_time = {total_time}")
    assert(
        len(site_rate_matrices["cpu"]) == num_sites
    )
    assert(
        len(site_rate_matrices["cpu"]) == num_sites
    )
    for Q_1, Q_2 in zip(site_rate_matrices["cpu"], site_rate_matrices["cuda"]):
        np.testing.assert_array_almost_equal(Q_1, Q_2, decimal=2)


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_data():
    """
    Tests that fast site rate implementation gives same results as older one.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    NUM_PYTORCH_EPOCHS = 30  # Seems like we can go down to 30 alright.
    GRID_SIZE = 8  # We can go as low as 16 or even 8!

    site_rate_matrices = {}
    for device in ["cpu", "cuda"]:
        print(f"***** device = {device} *****")
        msa = _get_msa_example()
        leaf_states = {
            msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper() * 512  # Repeat to get 65536 sites, like in GPN-MSA's batch size.
            for species_id in range(msa.shape[1])
        }
        tree_newick = _get_tree_newick()
        st = time.time()
        res_dict = learn_site_rate_matrices(
            tree_newick=tree_newick,
            msa={
                k: v[:num_sites]
                for (k, v) in leaf_states.items()
            },
            alphabet=["A", "C", "G", "T", "-"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 0.25, 0.25, 0.25, 0.25],
                    [0.25, -1.0, 0.25, 0.25, 0.25],
                    [0.25, 0.25, -1.0, 0.25, 0.25],
                    [0.25, 0.25, 0.25, -1.0, 0.25],
                    [0.25, 0.25, 0.25, 0.25, -1.0],
                ],
                index=["A", "C", "G", "T", "-"],
                columns=["A", "C", "G", "T", "-"],
            ),
            regularization_strength=0.5,
            device="cuda",

            num_rate_categories=5,

            alphabet_for_site_rate_estimation=["A", "C", "G", "T"],
            rate_matrix_for_site_rate_estimation=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            num_epochs=NUM_PYTORCH_EPOCHS,
            quantization_grid_num_steps=GRID_SIZE,
        )
        site_rate_matrices[device] = res_dict["learnt_rate_matrices"]
        total_time = 0.0
        for k, v in res_dict.items():
            if k.startswith("time_"):
                print(k, v)
                total_time += float(v)
        print(f"total_time {device} = {total_time} (sum of times) vs {time.time() - st} (real time)")
    assert(
        len(site_rate_matrices["cpu"]) == num_sites
    )
    assert(
        len(site_rate_matrices["cpu"]) == num_sites
    )
    np.testing.assert_array_almost_equal(
        site_rate_matrices["cpu"],
        site_rate_matrices["cuda"],
        decimal=2,
    )


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_data_2():
    """
    This test was used to tune the hyperparameters.

    I plot the rate matrices and make sure they are close to the gold standard.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return

    num_sites = 65536  # Use all 65536 to appreciate speedup of vectorized implementation.
    NUM_PYTORCH_EPOCHS = 30  # Seems like we can go down to 30 alright.
    GRID_SIZE = 8  # We can go as low as 16 or even 8!

    site_rate_matrices = {}
    msa = _get_msa_example()
    leaf_states = {
        msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper() * 512  # Repeat to get 65536 sites, like in GPN-MSA's batch size.
        for species_id in range(msa.shape[1])
    }
    tree_newick = _get_tree_newick()
    for repetition in range(2):
        # First repetition is used to warm up the GPU.
        if repetition == 1:
            st = time.time()
        res_dict = learn_site_rate_matrices(
            tree_newick=tree_newick,
            msa={
                k: v[:num_sites]
                for (k, v) in leaf_states.items()
            },
            alphabet=["A", "C", "G", "T", "-"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 0.25, 0.25, 0.25, 0.25],
                    [0.25, -1.0, 0.25, 0.25, 0.25],
                    [0.25, 0.25, -1.0, 0.25, 0.25],
                    [0.25, 0.25, 0.25, -1.0, 0.25],
                    [0.25, 0.25, 0.25, 0.25, -1.0],
                ],
                index=["A", "C", "G", "T", "-"],
                columns=["A", "C", "G", "T", "-"],
            ),
            regularization_strength=0.5,
            device="cuda",
            alphabet_for_site_rate_estimation=["A", "C", "G", "T"],
            rate_matrix_for_site_rate_estimation=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            num_epochs=NUM_PYTORCH_EPOCHS,
            quantization_grid_num_steps=GRID_SIZE,
        )
    site_rate_matrices = res_dict["learnt_rate_matrices"]
    total_time = 0.0
    for k, v in res_dict.items():
        if k.startswith("time_"):
            print(k, v)
            total_time += float(v)
    print(f"total_time cuda = {total_time} (sum of times) vs {time.time() - st} (real time)")

    ##### Uncomment this to plot site-specific rate matrices.
    # # Plot rate matrices
    # fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # Adjust figsize as needed for readability
    # fig.tight_layout(pad=4.0)  # Adjust spacing between plots

    # # Define the fixed color scale range
    # vmin, vmax = -5, 3
    # cmap = sns.diverging_palette(0, 240, s=75, l=50, n=500, center="light")

    # for i, ax in enumerate(axes.flat):  # Flatten the 2D grid of axes into a 1D array
    #     if i < 20:  # Ensure we don't exceed the number of plots
    #         sns.heatmap(
    #             site_rate_matrices[i],
    #             cmap=cmap, 
    #             center=0, 
    #             vmin=vmin, 
    #             vmax=vmax, 
    #             ax=ax, 
    #             cbar=False,  # Turn off individual colorbars for clarity
    #             annot=True,
    #             fmt=".1f",
    #         )
    #         ax.set_title(f"Site {i}")
    #     else:
    #         ax.axis("off")  # Turn off unused axes

    # # Add a single colorbar for the entire figure
    # cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # sns.heatmap([[vmin, vmax]], cmap=cmap, cbar=True, cbar_ax=cbar_ax)
    # cbar_ax.set_yticks([vmin, 0, vmax])
    # cbar_ax.set_yticklabels([vmin, 0, vmax])

    # plt.savefig(f"grid_plot__{GRID_SIZE}_qps__{NUM_PYTORCH_EPOCHS}_epochs__{NUM_SITE_RATES}_rates.png")
    # plt.show()
    # plt.close()
    # # assert(False)
