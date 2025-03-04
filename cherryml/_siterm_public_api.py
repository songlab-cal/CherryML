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
from cherryml._siterm._learn_site_rate_matrix import get_standard_site_rate_grid, get_standard_site_rate_prior, _get_msa_example, _get_test_msa, _get_test_tree, _get_tree_newick, convert_newick_to_CherryML_Tree
from cherryml.utils import get_amino_acids


def learn_site_specific_rate_matrices(
    tree: Optional[cherryml_io.Tree],
    msa: Dict[str, str],
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float = 0.5,
    device: str = "cpu",
    num_rate_categories: int = 20,
    alphabet_for_site_rate_estimation: Optional[List[str]] = None,
    rate_matrix_for_site_rate_estimation: Optional[pd.DataFrame] = None,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = 64,
    use_vectorized_implementation: bool = True,
    just_run_fast_cherries: bool = False,
) -> Dict:
    """
    Learn a rate matrix per site given an MSA (and optionally a tree).

    This function implements learning under the SiteRM model. Briefly, the
    SiteRM model uses a different rate matrix per site. It is described in
    detail in our paper:

    ```
    Prillo, S., Wu, W., Song, Y.S.  (NeurIPS 2024) Ultrafast classical
    phylogenetic method beats large protein language models on variant
    effect prediction.
    ```

    We offer two different implementation of SiteRM training, a "vectorized"
    version an a "non-vectorized" version. The default implementation is the
    vectorized one. Briefly, in the vectorized implementation a single
    computational graph is built encompassing all sites in the MSA, with the
    loss (i.e. the data log-likelihood) being the sum over all sites. This
    implementation uses 4D tensors to batch all the sites together, and is
    great when used in combination with `device="cuda"`. In other words, the
    vectorized implementation is recommended when GPU is available. When only
    CPU is available, we provide an alternative implementation where we simply
    loop over all sites in the MSA, one at a time. This implementation solves
    one optimization problem per site in the MSA, and involves only 3D tensors.
    This non-vectorized implementation makes sense if RAM memory becomes
    a bottleneck (e.g. if working on a personal computer).

    Args:
        tree: If `None`, then FastCherries will be used to estimate the tree.
            Otherwise, you can provide your own tree, which should be of
            the CherryML Tree type.
            NOTE: You can easily convert a newick tree to the CherryML Tree
            type using:
            ```
            from cherryml.io import convert_newick_to_CherryML_Tree
            tree = convert_newick_to_CherryML_Tree(tree_newick)
            ```
            Alternatively, if you have a file containing a tree in the CherryML
            format, you can just do:
            ```
            from cherryml.io import read_tree
            tree = read_tree(tree_path)
            ```
        msa: Dictionary mapping each leaf in the tree to the states (e.g.
            protein or DNA sequence) for that leaf.
        alphabet: List of valid states, e.g. ["A", "C", "G", "T"] for DNA.
        regularization_rate_matrix: Rate matrix to use to regularize the learnt
            rate matrices.
        regularization_strength: Between 0 and 1. 0 means no regularization at
            all, and 1 means fully regularized model. This is lambda in our
            paper.
        device: Whether to use "cpu" or GPU ("cuda"). Note that this is only
            used for the vectorized implementation, i.e. if
            `use_vectorized_implementation=False` then only CPU will be used,
            as it doesn't really make sense to use GPU in this case.
        num_rate_categories: Number of rate categories to use.
        alphabet_for_site_rate_estimation: Alphabet for learning the SITE RATES.
            If `None`, then the alphabet for the learnt rate matrices, i.e.
            `alphabet`, will be used. In our FastCherries/SiteRM paper, we have
            observed that for ProteinGym variant effect prediction, it works
            best to *exclude* gaps while estimating site rates (as is standard
            in statistical phylogenetics), but then use gaps when learning the
            rate matrix at the site. In that case, one would use
            `alphabet_for_site_rate_estimation=["A", "C", "G", "T"]`
            together with `alphabet=["A", "C", "G", "T", "-"]`.
        rate_matrix_for_site_rate_estimation: If provided, the rate matrix to
            use to estimate site rates. If `None`, then the
            `regularization_rate_matrix` will be used.
        num_epochs: Number of epochs (Adam steps) in the PyTorch optimizer.
        quantization_grid_num_steps: Number of quantization points to use will
            be `2 * quantization_grid_num_steps + 1` (as we take this number of
            steps left and right of the grid center). Lowering
            `quantization_grid_num_steps` leads to faster training at the
            expense of some accuracy. By default,
            `quantization_grid_num_steps=64` works really well, but great
            estimates can still be obtained faster with as low as
            `quantization_grid_num_steps=8`.
        use_vectorized_implementation: When `True`, a single computational
            graph including all sites of the MSA will be constructed to learn
            the site-specific rate matrices. Otherwise, the algorithm will loop
            over each site in the MSA separately, running coordinate ascent
            once for each site. As a note, while the vectorized and
            non-vectorized implementation converge to the same solution, they
            don't converge with the same trajectories, so when using a smaller
            number of `num_epochs` (e.g. `num_epochs=30`) the learnt rate
            matrices may differ in non-trivial ways. However, when using a
            larger number of epochs, e.g. `num_epochs=200`, they should be
            very similar. The main reason to use the non-vectorized
            implementation is because it requires less RAM memory, as each
            site it processes separately, making it faster when RAM is limited.
        just_run_fast_cherries: If `True`, then only the trees estimated with
            FastCherries will be returned, i.e. we will skip SiteRM. This is
            useful if all you need are the cherries and site rates of
            FastCherries. Recall that FastCherries only estimates the cherries
            in the tree, so the returned tree will be a star-type tree with all
            the inferred cherries hanging from the root. `learnt_rate_matrices`
            will be None in this case.

    Returns:
        A dictionary with the following entries:
            - "learnt_rate_matrices": A 3D numpy array with the learnt rate
                matrix per site.
            - "learnt_site_rates": A List[float] with the learnt site rate per
                site.
            - "learnt_tree": The FastCherries learnt tree (or the provided tree
                if it was provided). It is of type cherryml_io.Tree. Note that
                FastCherries only estimates the cherries in the tree and
                therefore returns a star-type tree with all the inferred
                cherries hanging from the root. Such as tree might look like
                this in newick format:
                "((leaf_1:0.17,leaf_2:0.17)internal-0:1,(leaf_3:0.17,leaf_4:0.17)internal-1:1);"
            - "time_...": The time taken by this substep. (They should add up
                to the total runtime, up to a small rather negligible
                difference).
    """
    site_rate_grid = get_standard_site_rate_grid(num_site_rates=num_rate_categories)
    site_rate_prior = get_standard_site_rate_prior(num_site_rates=num_rate_categories)
    res_dict = learn_site_rate_matrices(
        tree=tree,
        leaf_states=msa,
        alphabet=alphabet,
        regularization_rate_matrix=regularization_rate_matrix,
        regularization_strength=regularization_strength,
        use_vectorized_implementation=use_vectorized_implementation,
        vectorized_implementation_device=device,
        vectorized_implementation_num_cores=1,  # Doesn't really make sense to vectorize at this level.
        site_rate_grid=site_rate_grid,
        site_rate_prior=site_rate_prior,
        alphabet_for_site_rate_estimation=alphabet_for_site_rate_estimation,
        rate_matrix_for_site_rate_estimation=rate_matrix_for_site_rate_estimation,
        num_epochs=num_epochs,
        use_fast_site_rate_implementation=True,
        quantization_grid_num_steps=quantization_grid_num_steps,
        just_run_fast_cherries=just_run_fast_cherries,
    )
    return res_dict


def test_learn_site_specific_rate_matrices():
    res_dict = learn_site_specific_rate_matrices(
        tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
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


def test_just_run_fast_cherries():
    res_dict = learn_site_specific_rate_matrices(
        tree=None,
        msa={"leaf_1": "AAAA", "leaf_2": "AAAT", "leaf_3": "TTTA", "leaf_4": "TTTT"},
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
        just_run_fast_cherries=True,
    )
    assert res_dict["learnt_site_rates"]is not None
    assert res_dict["learnt_tree"] is not None
    assert res_dict["learnt_rate_matrices"] is None


def test_learn_site_specific_rate_matrices_large():
    tree_newick = _get_tree_newick()
    leaves = convert_newick_to_CherryML_Tree(tree_newick).leaves()
    alphabet = ["A", "C", "G", "T"]
    site_rate_matrices = learn_site_specific_rate_matrices(
        tree=convert_newick_to_CherryML_Tree(tree_newick),
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
        tree=convert_newick_to_CherryML_Tree(tree_newick),
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
        num_epochs=30,
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

    # plt.savefig(f"test_learn_site_specific_rate_matrices_real_vectorized_GOAT.png")
    # plt.show()
    # plt.close()
    # # assert(False)


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries():
    """
    Same as above but we infer the tree instead.

    NOTE: To run this specific test, you can do:
    $ python -m pytest cherryml/_siterm_public_api.py::test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries --runslow
    """
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    msa = _get_msa_example()
    leaf_states = {
        msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
        for species_id in range(msa.shape[1])
    }
    tree = None  # (instead of _get_tree_newick())
    st = time.time()
    res_dict = learn_site_specific_rate_matrices(
        tree=tree,
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
        num_epochs=30,
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

    # plt.savefig(f"test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries.png")
    # plt.show()
    # plt.close()
    # # assert(False)


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
            tree=convert_newick_to_CherryML_Tree(tree_newick),
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
            num_epochs=30,
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
            tree=convert_newick_to_CherryML_Tree(tree_newick),
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
            tree=convert_newick_to_CherryML_Tree(tree_newick),
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


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins():
    """
    Same as test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries
    but with protein alignments
    """
    site_rate_matrices = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    msa = cherryml_io.read_msa(os.path.join(dir_path, "../tests/data/Aln0037_txt-gb_phyml.txt"))
    st = time.time()
    res_dict = learn_site_specific_rate_matrices(
        tree=None,
        msa=msa,
        alphabet=get_amino_acids(),
        regularization_rate_matrix=cherryml_io.read_rate_matrix("data/rate_matrices/equ.txt"),
        regularization_strength=0.5,
        alphabet_for_site_rate_estimation=get_amino_acids(),
        rate_matrix_for_site_rate_estimation=cherryml_io.read_rate_matrix("data/rate_matrices/equ.txt"),
        num_epochs=30,
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
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust figsize as needed for readability
    # fig.tight_layout(pad=4.0)  # Adjust spacing between plots

    # # Define the fixed color scale range
    # vmin, vmax = -5, 3
    # cmap = sns.diverging_palette(0, 240, s=75, l=50, n=500, center="light")

    # for i, ax in enumerate(axes.flat):  # Flatten the 2D grid of axes into a 1D array
    #     if i < 4:  # Ensure we don't exceed the number of plots
    #         sns.heatmap(
    #             pd.DataFrame(site_rate_matrices[i], columns=get_amino_acids(), index=get_amino_acids()),
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

    # plt.savefig(f"test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins.png")
    # plt.show()
    # plt.close()
    # # assert(False)


@pytest.mark.slow
def test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins_not_vectorized():
    """
    Same as test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins
    but we don't use the vectorized implementation. Note that the rate matrices learnt by these two tests
    are different because we set `num_epochs=30`, but when set to higher e.g. `num_epochs=200` they are
    very similar.

    NOTE: To run this specific test, you can do:
    $ python -m pytest cherryml/_siterm_public_api.py::test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins_not_vectorized --runslow
    """
    site_rate_matrices = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    msa = cherryml_io.read_msa(os.path.join(dir_path, "../tests/data/Aln0037_txt-gb_phyml.txt"))
    st = time.time()
    res_dict = learn_site_specific_rate_matrices(
        tree=None,
        msa=msa,
        alphabet=get_amino_acids(),
        regularization_rate_matrix=cherryml_io.read_rate_matrix("data/rate_matrices/equ.txt"),
        regularization_strength=0.5,
        alphabet_for_site_rate_estimation=get_amino_acids(),
        rate_matrix_for_site_rate_estimation=cherryml_io.read_rate_matrix("data/rate_matrices/equ.txt"),
        num_epochs=30,
        use_vectorized_implementation=False,
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
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust figsize as needed for readability
    # fig.tight_layout(pad=4.0)  # Adjust spacing between plots

    # # Define the fixed color scale range
    # vmin, vmax = -5, 3
    # cmap = sns.diverging_palette(0, 240, s=75, l=50, n=500, center="light")

    # for i, ax in enumerate(axes.flat):  # Flatten the 2D grid of axes into a 1D array
    #     if i < 4:  # Ensure we don't exceed the number of plots
    #         sns.heatmap(
    #             pd.DataFrame(site_rate_matrices[i], columns=get_amino_acids(), index=get_amino_acids()),
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

    # plt.savefig(f"test_learn_site_specific_rate_matrices_real_vectorized_GOAT_with_fast_cherries_proteins_not_vectorized.png")
    # plt.show()
    # plt.close()
    # # assert(False)
