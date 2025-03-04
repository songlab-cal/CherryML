import tempfile
import time
import os
from cherryml import io as cherryml_io
from cherryml.io import convert_newick_to_CherryML_Tree
from cherryml import markov_chain
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from ._site_specific_rate_matrix import _estimate_site_specific_rate_matrices_given_tree_and_site_rates
from scipy.stats import gamma
import logging
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
from cherryml._siterm.fast_site_rates import compute_optimal_site_rates  # Import the compiled Cython function
from cherryml.phylogeny_estimation._fast_cherries import fast_cherries

# Default quantization grid:
QUANTIZATION_GRID_CENTER = 0.03
QUANTIZATION_GRID_STEP = 1.1
QUANTIZATION_GRID_NUM_STEPS = 64


def _get_cherry_transitions(
    tree: cherryml_io.Tree,
    msa: Dict[str, Any],
) -> List[Tuple[Any, Any, float]]:
    """
    Note: copy-pasta from protein-evolution...

    NOTE: Does not include reverse transitions. I.e. if (x, y, t) is a cherry, then (y, x, t) will not be returned.
    You will need to take care of this later if you want reverse transitions.
    """
    cherries = []

    def dfs(node) -> Tuple[Optional[int], Optional[float]]:
        """
        Pair up leaves under me.

        Return a single unpaired leaf and its distance, it such exists.
        """
        if tree.is_leaf(node):
            return (node, 0.0)
        unmatched_leaves_under = []
        distances_under = []
        for child, branch_length in tree.children(node):
            maybe_unmatched_leaf, maybe_distance = dfs(child)
            if maybe_unmatched_leaf is not None:
                assert maybe_distance is not None
                unmatched_leaves_under.append(maybe_unmatched_leaf)
                distances_under.append(maybe_distance + branch_length)
        assert len(unmatched_leaves_under) == len(distances_under)
        index = 0

        while index + 1 <= len(unmatched_leaves_under) - 1:
            (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                (unmatched_leaves_under[index], distances_under[index]),
                (
                    unmatched_leaves_under[index + 1],
                    distances_under[index + 1],
                ),
            )
            leaf_seq_1, leaf_seq_2 = msa[leaf_1], msa[leaf_2]
            cherries.append(
                (
                    leaf_seq_1,
                    leaf_seq_2,
                    branch_length_1 + branch_length_2
                )
            )
            index += 2
        if len(unmatched_leaves_under) % 2 == 0:
            return (None, None)
        else:
            return (unmatched_leaves_under[-1], distances_under[-1])

    dfs(tree.root())
    assert len(cherries) == int(len(tree.leaves()) / 2)
    return cherries


def _get_test_tree() -> cherryml_io.Tree:
    tree = cherryml_io.Tree()
    tree.add_nodes(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    tree.add_edges([
        ("0", "1", 0.01),
        ("0", "2", 0.02),
        ("1", "3", 0.13),
        ("1", "7", 0.17),
        ("2", "4", 0.24),
        ("2", "8", 0.28),
        ("3", "5", 0.35),
        ("3", "6", 0.36),
        ("4", "9", 0.49),
        ("4", "10", 0.410),
    ])
    return tree


def _get_test_msa() -> Dict[str, str]:
    msa = {
        "5": "AG",
        "6": "BH",
        "7": "CI",
        "8": "DJ",
        "9": "EK",
        "10": "FL",
    }
    return msa


def test_get_cherry_transitions():
    tree = _get_test_tree()
    msa = _get_test_msa()
    cherries = _get_cherry_transitions(tree, msa)
    expected_cherries = [
        ("AG", "BH", 0.35 + 0.36),
        ("EK", "FL", 0.49 + 0.410),
        ("CI", "DJ", 0.17 + 0.28 + 0.01 + 0.02),
    ]
    if cherries != expected_cherries:
        raise ValueError(
            f"test_get_cherry_transitions failed.\n"
            f"expected_cherries={expected_cherries}\n"
            f"cherries={cherries}"
        )


def _matrix_exponential_reversible(
    rate_matrix: np.array,
    exponents: List[float],
    device: str = "cpu",
) -> np.array:
    """
    Compute matrix exponential (batched).

    Args:
        rate_matrix: Rate matrix for which to compute the matrix exponential
        exponents: List of exponents.
    Returns:
        3D tensor where res[:, i, i] contains exp(rate_matrix * exponents[i])
    """
    return markov_chain.matrix_exponential_reversible(
        exponents=exponents,
        fact=markov_chain.FactorizedReversibleModel(rate_matrix),
        device=device,
    )


def _get_site_rate_log_likelihood_given_matrix_exponentials(
    cherries: List[Tuple[str, str, float]],
    mexps: List[pd.DataFrame]
) -> float:
    res = 0.0
    alphabet_set = set(mexps[0].columns)
    for (cherry_id, (x_i, y_i, t)) in enumerate(cherries):
        assert(len(x_i) == 1)
        assert(len(y_i) == 1)
        if (x_i not in alphabet_set) or (y_i not in alphabet_set):
            if x_i.islower():
                raise ValueError(f"Lowercase state found: '{x_i}' . Did you forget to make it uppercase?")
            if y_i.islower():
                raise ValueError(f"Lowercase state found: '{y_i}' . Did you forget to make it uppercase?")
            continue
        res += np.log(mexps[cherry_id].loc[x_i, y_i])
    return res


def _get_site_rate_log_likelihood(
    cherries: List[Tuple[str, str, float]],
    site_rate: float,
    rate_matrix: pd.DataFrame,
) -> float:
    res = 0.0
    alphabet_set = set(rate_matrix.columns)
    for (x_i, y_i, t) in cherries:
        assert(len(x_i) == 1)
        assert(len(y_i) == 1)
        if (x_i not in alphabet_set) or (y_i not in alphabet_set):
            if x_i.islower():
                raise ValueError(f"Lowercase state found: '{x_i}' . Did you forget to make it uppercase?")
            if y_i.islower():
                raise ValueError(f"Lowercase state found: '{y_i}' . Did you forget to make it uppercase?")
            continue
        mexp = pd.DataFrame(
            _matrix_exponential_reversible(rate_matrix=rate_matrix.to_numpy(), exponents=[t * site_rate])[0, :, :],
            columns=rate_matrix.columns,
            index=rate_matrix.index,
        )
        res += np.log(mexp.loc[x_i, y_i])
    return res


def test__get_site_rate_log_likelihood():
    res = _get_site_rate_log_likelihood(
        cherries=[("A", "A", 1.0), ("A", "T", 0.1), ("G", "C", 0.2), ("C", "A", 0.3), ("G", "T", 6.0), ("N", "A", 1.0), ("A", "-", 1.0)],
        site_rate=0.5,
        rate_matrix=pd.DataFrame(
            [
                [-3.0, 1.0, 1.0, 1.0],
                [1.0, -3.0, 1.0, 1.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
    )
    # print(f"res = {res}")
    np.testing.assert_almost_equal(res, -10.204000096823325)  # NOTE: I should be able to solve analytically by hand.


def _estimate_site_rate(
    tree: cherryml_io.Tree,
    leaf_states: Dict[str, str],
    rate_matrix: pd.DataFrame,
    site_rate_grid: List[float] = [2.0 ** i for i in range(-10, 10)],
    site_rate_prior: List[float] = [1.0 for i in range(-10, 10)],
) -> float:
    """
    For reference, Wilson's implementation for CherrierML is in:

    cherryml/phylogeny_estimation/mcle_cherries/likelihood_functions.cpp (branch neurips-2024)
    """
    if len(site_rate_grid) != len(site_rate_prior):
        raise ValueError(
            f"site_rate_grid and site_rate_prior should have the same length. "
            f"You provided: site_rate_grid='{site_rate_grid}' and "
            f"site_rate_prior='{site_rate_prior}'."
        )
    cherries = _get_cherry_transitions(
        tree=tree,
        msa=leaf_states,
    )
    # We will use the reverse transitions too to compute the site rate.
    # These aren't automatically provided by _get_cherry_transitions so we do it.
    cherries_with_reverses = cherries + [
        (y, x, t)
        for (x, y, t) in cherries
    ]
    # We'll now brute-force a grid of candidate site rates.
    log_likelihoods_and_site_rates = [
        (
            np.log(site_prior) + _get_site_rate_log_likelihood(
                cherries=cherries_with_reverses,
                site_rate=site_rate,
                rate_matrix=rate_matrix
            ),
            site_rate
        )
        for site_rate, site_prior in zip(site_rate_grid, site_rate_prior)
    ]
    # print(f"log_likelihoods_and_site_rates = {log_likelihoods_and_site_rates}")
    optimal_site_rate = sorted(log_likelihoods_and_site_rates)[-1][1]
    return optimal_site_rate


def test__estimate_site_rate():
    site_rate = _estimate_site_rate(
        tree=_get_test_tree(),
        leaf_states={  # If you play with this you can increase or decrease the site rate. The cherries are (5, 6), (7, 8), (9, 10)
            "5": "A",
            "6": "A",
            "7": "T",
            "8": "T",
            "9": "G",
            "10": "C",
        },
        rate_matrix=pd.DataFrame(
            [
                [-3.0, 1.0, 1.0, 1.0],
                [1.0, -3.0, 1.0, 1.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
    )
    # print(f"site_rate= {site_rate}")
    assert(site_rate == 0.25)


# ##### Non-cython version of function. Keeping here for reference.
# def _estimate_site_rates_fast(
#     tree: cherryml_io.Tree,
#     leaf_states: Dict[str, str],
#     site_rate_grid: List[float],
#     site_rate_prior: List[float],
#     rate_matrix: pd.DataFrame,
# ) -> List[float]:
#     """
#     For reference, Wilson's implementation for CherrierML is in:

#     cherryml/phylogeny_estimation/mcle_cherries/likelihood_functions.cpp (branch neurips-2024)

#     Assumes an ASCII alphabet.
#     """
#     if len(site_rate_grid) == 1:
#         # We don't need to do any math.
#         num_sites = len(list(leaf_states.values())[0])
#         return [site_rate_grid[0]] * num_sites
#     st = time.time()
#     # Now we need to convert the alphabet to integers to avoid slow pandas indexing.
#     # Gap will get mapped to the last index
#     mapping = {chr(i): -1 for i in range(128)}
#     for state_idx, state in enumerate(list(rate_matrix.columns)):
#         mapping[state] = state_idx
#     leaf_states_int = {
#         leaf: list(map(lambda x: mapping[x], states))
#         for (leaf, states) in leaf_states.items()
#     }
#     print(f"time map states to int: {time.time() - st}")

#     st = time.time()
#     cherries = _get_cherry_transitions(
#         tree=tree,
#         msa=leaf_states_int,
#     )
#     # We will use the reverse transitions too to compute the site rate.
#     # These aren't automatically provided by _get_cherry_transitions so we do it.
#     cherries = cherries + [
#         (y, x, t)
#         for (x, y, t) in cherries
#     ]
#     print(f"time _get_cherry_transitions: {time.time() - st}")

#     # Precompute all mexps needed
#     st = time.time()
#     num_rates = len(site_rate_grid)
#     num_cherries = len(cherries)
#     num_states = rate_matrix.shape[0]
#     log_mexps_tensor = np.log(
#         _matrix_exponential_reversible(
#             rate_matrix=rate_matrix.to_numpy(),
#             exponents=[rate * t for rate in site_rate_grid for (x, y, t) in cherries],
#             device="cpu",
#         )
#     ).reshape(
#         num_rates,
#         num_cherries,
#         num_states,
#         num_states,
#     )  # Indexed by [rate, cherry, x, y]
#     # We need to ignore transitions with gaps. This is achieved by setting the gap state index to num_states, and setting all its log likelihoods to zeros, as follows:
#     log_mexps_tensor_w_gaps = np.zeros(
#         shape=(
#             num_rates,
#             num_cherries,
#             num_states + 1,
#             num_states + 1,
#         )
#     )
#     log_mexps_tensor_w_gaps[:, :, :num_states, :num_states] = log_mexps_tensor
#     print(f"time log_mexps_tensor: {time.time() - st}")

#     st = time.time()
#     num_sites = len(cherries[0][0])
#     optimal_site_rates = []
#     for site_id in range(num_sites):
#         # Estimate rate for this site.
#         cherries_at_site = [
#             (x[site_id], y[site_id], t) for (x, y, t) in cherries
#         ]
#         log_likelihoods_and_site_rates = [
#             (
#                 np.log(site_prior) + sum(
#                     [
#                         log_mexps_tensor_w_gaps[site_rate_id, cherry_id, x_i, y_i]
#                         for (cherry_id, (x_i, y_i, t)) in enumerate(cherries_at_site)
#                     ]
#                 ),
#                 site_rate
#             )
#             for ((site_rate_id, site_rate), site_prior) in zip(list(enumerate(site_rate_grid)), site_rate_prior)
#         ]
#         # print(f"log_likelihoods_and_site_rates = {log_likelihoods_and_site_rates}")
#         optimal_site_rate = sorted(log_likelihoods_and_site_rates)[-1][1]
#         optimal_site_rates.append(optimal_site_rate)
#     print(f"time optimal_site_rates: {time.time() - st}")
#     # assert(False)
#     return optimal_site_rates


def _estimate_site_rates_fast(
    tree,
    leaf_states,
    site_rate_grid,
    site_rate_prior,
    rate_matrix,
):
    """
    For reference, Wilson's implementation for CherrierML is in:

    cherryml/phylogeny_estimation/mcle_cherries/likelihood_functions.cpp (branch neurips-2024)

    Assumes an ASCII alphabet.
    """
    if len(site_rate_grid) == 1:
        # We don't need to do any math.
        num_sites = len(list(leaf_states.values())[0])
        return [site_rate_grid[0]] * num_sites
    st = time.time()
    # Now we need to convert the alphabet to integers to avoid slow pandas indexing.
    # Gap will get mapped to the last index
    mapping = {chr(i): -1 for i in range(128)}
    for state_idx, state in enumerate(list(rate_matrix.columns)):
        mapping[state] = state_idx
    leaf_states_int = {
        leaf: list(map(lambda x: mapping[x], states))
        for (leaf, states) in leaf_states.items()
    }
    # print(f"time map states to int: {time.time() - st}")

    st = time.time()
    cherries = _get_cherry_transitions(
        tree=tree,
        msa=leaf_states_int,
    )
    # We will use the reverse transitions too to compute the site rate.
    # These aren't automatically provided by _get_cherry_transitions so we do it.
    cherries = cherries + [
        (y, x, t)
        for (x, y, t) in cherries
    ]
    # print(f"time _get_cherry_transitions: {time.time() - st}")

    # Precompute all mexps needed
    st = time.time()
    num_rates = len(site_rate_grid)
    num_cherries = len(cherries)
    num_states = rate_matrix.shape[0]
    log_mexps_tensor = np.log(
        _matrix_exponential_reversible(
            rate_matrix=rate_matrix.to_numpy(),
            exponents=[rate * t for rate in site_rate_grid for (x, y, t) in cherries],
            device="cpu",
        )
    ).reshape(
        num_rates,
        num_cherries,
        num_states,
        num_states,
    )  # Indexed by [rate, cherry, x, y]
    # print(f"time log_mexps_tensor: {time.time() - st}")

    # Precompute all mexps needed (this remains the same)
    st = time.time()
    log_mexps_tensor_w_gaps = np.zeros(
        shape=(
            num_rates,
            num_cherries,
            num_states + 1,
            num_states + 1,
        )
    )
    log_mexps_tensor_w_gaps[:, :, :num_states, :num_states] = log_mexps_tensor
    # print(f"time init log_mexps_tensor_w_gaps: {time.time() - st}")

    num_sites = len(cherries[0][0])

    # Call the Cython function
    st = time.time()
    optimal_site_rates = compute_optimal_site_rates(
        num_sites,
        cherries,
        log_mexps_tensor_w_gaps,
        site_rate_grid,
        site_rate_prior,
    )
    # print(f"time cython: {time.time() - st}")
    return optimal_site_rates


def _estimate_site_rates(
    tree: cherryml_io.Tree,
    leaf_states: Dict[str, str],
    site_rate_grid: List[float],
    site_rate_prior: List[float],
    rate_matrix: pd.DataFrame,
) -> List[float]:
    """
    For reference, Wilson's implementation for CherrierML is in:

    cherryml/phylogeny_estimation/mcle_cherries/likelihood_functions.cpp (branch neurips-2024)
    """
    cherries = _get_cherry_transitions(
        tree=tree,
        msa=leaf_states,
    )
    # We will use the reverse transitions too to compute the site rate.
    # These aren't automatically provided by _get_cherry_transitions so we do it.
    cherries = cherries + [
        (y, x, t)
        for (x, y, t) in cherries
    ]

    # Precompute all mexps needed
    mexps = [
        [
            pd.DataFrame(
                _matrix_exponential_reversible(rate_matrix=rate_matrix.to_numpy(), exponents=[site_rate * t])[0, :, :],
                columns=rate_matrix.columns,
                index=rate_matrix.index,
            )
            for (_, _, t) in cherries
        ]
        for site_rate in site_rate_grid
    ]  # mexps[site_rate_id][cherry_id] gives exp(site_rate * t)
    num_sites = len(cherries[0][0])
    optimal_site_rates = []
    for site_id in range(num_sites):
        # Estimate rate for this site.
        log_likelihoods_and_site_rates = [
            (
                np.log(site_prior) + _get_site_rate_log_likelihood_given_matrix_exponentials(
                    cherries=[
                        (x[site_id], y[site_id], t)
                        for (x, y, t) in cherries
                    ],  # Subset cherries to site
                    mexps=mexps[site_rate_id],  # Subset mexps to rate
                ),
                site_rate
            )
            for ((site_rate_id, site_rate), site_prior) in zip(list(enumerate(site_rate_grid)), site_rate_prior)
        ]
        # print(f"log_likelihoods_and_site_rates = {log_likelihoods_and_site_rates}")
        optimal_site_rate = sorted(log_likelihoods_and_site_rates)[-1][1]
        optimal_site_rates.append(optimal_site_rate)
    return optimal_site_rates


def test__estimate_site_rates():
    site_rates = _estimate_site_rates(
        tree=_get_test_tree(),
        leaf_states={  # If you play with this you can increase or decrease the site rate. The cherries are (5, 6), (7, 8), (9, 10)
            "5": "A",
            "6": "A",
            "7": "T",
            "8": "T",
            "9": "G",
            "10": "C",
        },
        rate_matrix=pd.DataFrame(
            [
                [-3.0, 1.0, 1.0, 1.0],
                [1.0, -3.0, 1.0, 1.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
        site_rate_grid=[2.0 ** i for i in range(-10, 10)],
        site_rate_prior=[1.0 for i in range(20)],
    )
    # print(f"site_rates = {site_rates}")
    assert(site_rates == [0.25])


def _learn_site_rate_matrix_given_site_rate_too(
    tree: cherryml_io.Tree,
    site_rate: float,
    leaf_states: Dict[str, str],
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = QUANTIZATION_GRID_NUM_STEPS,
    num_cores: int = 1,
) -> pd.DataFrame:
    quantization_grid_center = QUANTIZATION_GRID_CENTER
    quantization_grid_step = QUANTIZATION_GRID_STEP ** (QUANTIZATION_GRID_NUM_STEPS / quantization_grid_num_steps)
    # quantization_grid_num_steps = QUANTIZATION_GRID_NUM_STEPS
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    learned_rate_tensor_numpy = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
        # Key arguments
        tree=tree,
        site_rates=[site_rate],
        msa=leaf_states,
        alphabet=alphabet,
        # Hyperparameters
        regularization_strength=regularization_strength,
        regularization_rate_matrix=regularization_rate_matrix.to_numpy(),
        quantization_points=quantization_points,  # (to quantize branch lengths)
        optimization_num_epochs=num_epochs,
        transitions_strategy="cherry++",
        include_reverse_transitions=True,
        rate_matrix_parameterization="pande_reversible",
        log_dir=None,
        plot_site_specific_rate_matrices=0,
        vectorized_cherryml_implementation_num_cores=num_cores,
    )["res"]
    site_specific_rate_matrix = pd.DataFrame(
        learned_rate_tensor_numpy[0, :, :],
        columns=alphabet,
        index=alphabet,
    )
    return site_specific_rate_matrix


def test__learn_site_rate_matrix_given_site_rate_too():
    site_rate_matrix = _learn_site_rate_matrix_given_site_rate_too(
        tree=_get_test_tree(),
        site_rate=0.25,
        leaf_states={
            "5": "A",
            "6": "A",
            "7": "T",
            "8": "T",
            "9": "G",
            "10": "C",
        },
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
        ),
        regularization_strength=0.5,
    )
    pd.testing.assert_frame_equal(
        site_rate_matrix.T,
        pd.DataFrame(
            [
                [-0.25240713357925415, 0.08471570909023285, 0.08471573144197464, 0.08297567069530487],
                [0.10483385622501373, -1.833121657371521, 1.6207798719406128, 0.10750797390937805],
                [0.10483382642269135, 1.6207789182662964, -1.8331207036972046, 0.10750793665647507],
                [0.10106398910284042, 0.1058153510093689, 0.10581538081169128, -0.3126947283744812],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).T,
    )


def _learn_site_rate_matrices_given_site_rates_too(
    tree: cherryml_io.Tree,
    site_rates: List[float],
    leaf_states: Dict[str, str],
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float,
    use_vectorized_cherryml_implementation: bool = True,
    vectorized_cherryml_implementation_device: str = "cpu",
    vectorized_cherryml_implementation_num_cores: int = 1,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = QUANTIZATION_GRID_NUM_STEPS,
) -> Dict:
    """
    Wrapper around _estimate_site_specific_rate_matrices_given_tree_and_site_rates
    which automatically tunes the quantization grid based on
    `quantization_grid_num_steps`.

    Returns a fictionary with:
        - "res": A List[pd.DataFrame] with the site-specific rate matrices.
        - "time_...": The time taken for this specific substep (should add up
            to the total runtime.)
    """
    profiling_res = {}
    st = time.time()
    quantization_grid_center = QUANTIZATION_GRID_CENTER
    quantization_grid_step = QUANTIZATION_GRID_STEP ** (QUANTIZATION_GRID_NUM_STEPS / quantization_grid_num_steps)
    # quantization_grid_num_steps = QUANTIZATION_GRID_NUM_STEPS
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]
    profiling_res["time_build_quantization_points"] = time.time() - st

    learned_rate_tensor_numpy__res_dict = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
        # Key arguments
        tree=tree,
        site_rates=site_rates,
        msa=leaf_states,
        alphabet=alphabet,
        # Hyperparameters
        regularization_strength=regularization_strength,
        regularization_rate_matrix=regularization_rate_matrix.to_numpy(),
        quantization_points=quantization_points,  # (to quantize branch lengths)
        optimization_num_epochs=num_epochs,
        transitions_strategy="cherry++",
        include_reverse_transitions=True,
        rate_matrix_parameterization="pande_reversible",
        log_dir=None,
        plot_site_specific_rate_matrices=0,
        use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
        vectorized_cherryml_implementation_device=vectorized_cherryml_implementation_device,
        vectorized_cherryml_implementation_num_cores=vectorized_cherryml_implementation_num_cores,
    )
    st = time.time()
    learned_rate_tensor_numpy = learned_rate_tensor_numpy__res_dict["res"]
    learned_rate_tensor_numpy__profiling_dict = {k: v for k, v in learned_rate_tensor_numpy__res_dict.items() if k.startswith("time_")}
    site_specific_rate_matrices = learned_rate_tensor_numpy
    profiling_res["time_build_pandas_return"] = time.time() - st
    res = {
        "res": site_specific_rate_matrices,
    }
    res = {**res, **learned_rate_tensor_numpy__profiling_dict, **profiling_res}
    return res


def test__learn_site_rate_matrices_given_site_rates_too():
    site_rate_matrices = _learn_site_rate_matrices_given_site_rates_too(
        tree=_get_test_tree(),
        site_rates=[0.25],
        leaf_states={
            "5": "A",
            "6": "A",
            "7": "T",
            "8": "T",
            "9": "G",
            "10": "C",
        },
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
        ),
        regularization_strength=0.5,
    )["res"]
    np.testing.assert_array_almost_equal(
        site_rate_matrices[0].T,
        pd.DataFrame(
            [
                [-0.25240713357925415, 0.08471570909023285, 0.08471573144197464, 0.08297567069530487],
                [0.10483385622501373, -1.833121657371521, 1.6207798719406128, 0.10750797390937805],
                [0.10483382642269135, 1.6207789182662964, -1.8331207036972046, 0.10750793665647507],
                [0.10106398910284042, 0.1058153510093689, 0.10581538081169128, -0.3126947283744812],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).T.to_numpy(),
        decimal=2,
    )


def test__learn_site_rate_matrices_given_site_rates_too_2():
    """
    Same as test__learn_site_rate_matrices_given_site_rates_too but with a
    rate matrix with a non-trivial stationary distribution.
    """
    site_rate_matrices = _learn_site_rate_matrices_given_site_rates_too(
        tree=_get_test_tree(),
        site_rates=[0.25, 0.5],
        leaf_states={
            "5": "AA",
            "6": "AC",
            "7": "TA",
            "8": "TT",
            "9": "GA",
            "10": "CG",
        },
        alphabet=["A", "C", "G", "T"],
        regularization_rate_matrix=pd.DataFrame(
            [
                [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                [1.0, 1.0, -3.0, 1.0],
                [1.0, 1.0, 1.0, -3.0],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ),
        regularization_strength=0.5,
    )["res"]
    np.testing.assert_array_almost_equal(
        site_rate_matrices[0],
        pd.DataFrame(
            [
                [-0.12, 0.04, 0.1, 0.07],
                [0.03, -1.13, 1.91, 0.06],
                [0.05, 1.06, -2.09, 0.08],
                [0.04, 0.03, 0.08, -0.21],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).T.to_numpy(),
        decimal=2,
    )


def learn_site_rate_matrix(
    tree: cherryml_io.Tree,
    leaf_states: Dict[str, str],
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float,
    site_rate_grid: List[float] = [2.0 ** i for i in range(-10, 10)],  # For backwards compatibility reasons. Use get_standard_site_rate_grid() instead.
    site_rate_prior: List[float] = [1.0 for i in range(-10, 10)],  # For backwards compatibility reasons. Use get_standard_site_rate_prior() instead.
    alphabet_for_site_rate_estimation: Optional[List[str]] = None,
    rate_matrix_for_site_rate_estimation: Optional[pd.DataFrame] = None,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = QUANTIZATION_GRID_NUM_STEPS,
    site_rate: Optional[float] = None,
    num_cores: int = 1,
) -> Dict:
    """
    Learn a rate matrix for a site given the tree and leaf states.

    Args:
        tree: The tree in CherryML's tree format.
        leaf_states: For each leaf in the tree, the state for that leaf.
        alphabet: List of valid states used to estimate the rate matrix.
            Standard use is ["A", "C", "G", "T"]. However, in
            FastCherries/SiteRM paper, we have that for ProteinGym variant
            effect prediction, it works best to include gaps. In that case,
            one would use `alphabet=["A", "C", "G", "T", "-"]`.
        regularization_rate_matrix: Rate matrix to use to regularize the learnt
            rate matrix.
        regularization_strength: Between 0 and 1. 0 means no regularization at all,
            and 1 means fully regularized model. The way this works is that the
            empirical transitions are mixed with pseudo-transitions from the prior
            with the mixing factor given by the `regularization_strength`.
        site_rate_grid: Grid of site rates to consider. The standard site
            rate grid (used in FastCherries/SiteRM) can be obtained with
            `get_standard_site_rate_grid()`.
        site_rate_prior: Prior probabilities for the site rates. A
            gamma(shape=3, scale=1/3) is typical (what we use in
            FastCherries/SiteRM). This can be obtained with
            `get_standard_site_rate_prior()`.
        alphabet_for_site_rate_estimation: Alphabet for learning the SITE RATES.
            If `None`, then the alphabet for the learnt rate matrix, i.e.
            `alphabet`, will be used. In the FastCherries/SiteRM paper, we have
            observed that for ProteinGym variant effect prediction, it works
            best to *exclude* gaps while estimating site rates (as is standard
            in statistical phylogenetics), but then use gaps when learning the
            rate matrix at the site. In that case, one would use
            `alphabet_for_site_rate_estimation=["A", "C", "G", "T"]` in
            combination with `alphabet=["A", "C", "G", "T", "-"]`.
        rate_matrix_for_site_rate_estimation: If provided, the rate matrix to use
            to estimate site rates. If `None`, then the
            `regularization_rate_matrix` will be used.
        site_rate: The site rate. If `None`, it will be estimated.
        num_cores: Number of cores used internally by PyTorch to speed up.

    Returns:
        A dictionary with the following entries:
            res["learnt_rate_matrix"]: The learnt rate matrix.
            res["learnt_site_rate"]: The learnt site rate.
    """
    if alphabet_for_site_rate_estimation is None:
        alphabet_for_site_rate_estimation = alphabet[:]
    if rate_matrix_for_site_rate_estimation is None:
        rate_matrix_for_site_rate_estimation = regularization_rate_matrix.copy()
    time_estimate_site_rate = 0.0
    time_learn_site_rate_matrix_given_site_rate_too = 0.0
    st = time.time()
    if site_rate is None:
        site_rate = _estimate_site_rate(
            tree=tree,
            leaf_states=leaf_states,
            rate_matrix=rate_matrix_for_site_rate_estimation,
            site_rate_grid=site_rate_grid,
            site_rate_prior=site_rate_prior,
        )
    time_estimate_site_rate = time.time() - st
    st = time.time()
    learnt_rate_matrix = _learn_site_rate_matrix_given_site_rate_too(
        tree=tree,
        site_rate=site_rate,
        leaf_states=leaf_states,
        alphabet=alphabet,
        regularization_rate_matrix=regularization_rate_matrix,
        regularization_strength=regularization_strength,
        num_epochs=num_epochs,
        quantization_grid_num_steps=quantization_grid_num_steps,
        num_cores=num_cores,
    )
    time_learn_site_rate_matrix_given_site_rate_too = time.time() - st
    res = {
        "learnt_rate_matrix": learnt_rate_matrix,
        "learnt_site_rate": site_rate,
        "time_estimate_site_rate": time_estimate_site_rate,
        "time_learn_site_rate_matrix_given_site_rate_too": time_learn_site_rate_matrix_given_site_rate_too
    }
    return res


def test_learn_site_rate_matrix():
    site_rate_matrix = learn_site_rate_matrix(
        tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
        leaf_states={"leaf_1": "A", "leaf_2": "A", "leaf_3": "C", "leaf_4": "G"},
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
        ),
        regularization_strength=0.5,
    )["learnt_rate_matrix"]
    pd.testing.assert_frame_equal(
        site_rate_matrix.T,
        pd.DataFrame(
            [
                [-0.09099140763282776, 0.027805309742689133, 0.027805369347333908, 0.03538072481751442],
                [0.04163093864917755, -1.5431658029556274, 1.4439103603363037, 0.057624444365501404],
                [0.04163086414337158, 1.4439047574996948, -1.543160080909729, 0.057624418288469315],
                [0.10580798238515854, 0.11509860306978226, 0.1150989904999733, -0.3360055685043335],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).T,
    )


def get_standard_site_rate_grid(num_site_rates: int = 20) -> List[float]:
    """
    Site rate grid used in FastCherries/SiteRM paper.
    """
    res = [
        num_site_rates ** (-1.0 + 2.0 * (num_site_rates - i) / (num_site_rates - 1.0))
        for i in range(1, num_site_rates + 1)
    ][::-1]
    return res


def get_standard_site_rate_prior(num_site_rates: int = 20) -> List[float]:
    """
    Site rate prior used in FastCherries/SiteRM paper.
    """
    res = [
        gamma.pdf(site_rate, a=3.0, scale=1.0/3.0)
        for site_rate in get_standard_site_rate_grid(num_site_rates=num_site_rates)
    ]
    return res


def test_learn_site_rate_matrix_with_site_rate_prior():
    """
    Test that using a site rate prior gives better site rate estimates.
    """
    # First run without priors. We should get degenerate site rates (too large/small)
    for leaf_states, expected_site_rate in [
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "C", "leaf_4": "G"}, 0.5),
        ({"leaf_1": "A", "leaf_2": "C", "leaf_3": "G", "leaf_4": "T"}, 16.0),  # Very large
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "A", "leaf_4": "A"}, 0.0009765625),  # Very small
    ]:
        res_dict = learn_site_rate_matrix(
            tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
            leaf_states=leaf_states,
            alphabet=["A", "C", "G", "T"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            regularization_strength=0.5,
        )
        site_rate = res_dict["learnt_site_rate"]
        np.testing.assert_almost_equal(
            min(site_rate, 16.0), min(expected_site_rate, 16.0),  # Adding min(_,16.0) bc test fails otherwise, hitting 512.0 on crowfoot.
        )

    # Now with priors. The extreme site rates should disappear.
    site_rate_grid = [2.0 ** i for i in range(-10, 10)]
    site_rate_prior = [gamma.pdf(site_rate, a=3.0, scale=1.0/3.0) for site_rate in site_rate_grid]
    for leaf_states, expected_site_rate in [
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "C", "leaf_4": "G"}, 0.5),
        ({"leaf_1": "A", "leaf_2": "C", "leaf_3": "G", "leaf_4": "T"}, 1.0),  # This is no longer too large
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "A", "leaf_4": "A"}, 0.25),  # This is no longer too small
    ]:
        res_dict = learn_site_rate_matrix(
            tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
            leaf_states=leaf_states,
            alphabet=["A", "C", "G", "T"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            regularization_strength=0.5,
            site_rate_grid=site_rate_grid,
            site_rate_prior=site_rate_prior,
        )
        site_rate = res_dict["learnt_site_rate"]
        np.testing.assert_almost_equal(
            site_rate, expected_site_rate,
        )

    # Now with the site rate grid and priors from the FastCherries/SiteRM paper.
    site_rate_grid = get_standard_site_rate_grid()
    site_rate_prior = get_standard_site_rate_prior()
    for leaf_states, expected_site_rate in [
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "C", "leaf_4": "G"}, 0.62312361621777),
        ({"leaf_1": "A", "leaf_2": "C", "leaf_3": "G", "leaf_4": "T"}, 0.8541314966877565),
        ({"leaf_1": "A", "leaf_2": "A", "leaf_3": "A", "leaf_4": "A"}, 0.17651113509036334),
    ]:
        res_dict = learn_site_rate_matrix(
            tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
            leaf_states=leaf_states,
            alphabet=["A", "C", "G", "T"],
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            regularization_strength=0.5,
            site_rate_grid=site_rate_grid,
            site_rate_prior=site_rate_prior,
        )
        site_rate = res_dict["learnt_site_rate"]
        np.testing.assert_almost_equal(
            site_rate, expected_site_rate,
        )


def test_learn_site_rate_matrix_with_site_rate_prior_and_gaps():
    """
    Test usage similar to that of FastCherries/SiteRM paper, where we estimate
    site rates *excluding* gaps (as is standard in statistical phylogenetics)
    but then learn a site rate matrix that *includes* the gap state.
    """
    res_dict = learn_site_rate_matrices(
        tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
        leaf_states={"leaf_1": "A", "leaf_2": "-", "leaf_3": "A", "leaf_4": "A"},
        alphabet=["A", "C", "G", "T", "-"],
        regularization_rate_matrix=pd.DataFrame(
            [
                [-1.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0],
                [1.0/4.0, -1.0, 1.0/4.0, 1.0/4.0, 1.0/4.0],
                [1.0/4.0, 1.0/4.0, -1.0, 1.0/4.0, 1.0/4.0],
                [1.0/4.0, 1.0/4.0, 1.0/4.0, -1.0, 1.0/4.0],
                [1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, -1.0],
            ],
            index=["A", "C", "G", "T", "-"],
            columns=["A", "C", "G", "T", "-"],
        ),
        regularization_strength=0.5,
        site_rate_grid=get_standard_site_rate_grid(),
        site_rate_prior=get_standard_site_rate_prior(),
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
        use_vectorized_implementation=True,
    )
    learnt_site_rate = res_dict["learnt_site_rates"][0]
    np.testing.assert_almost_equal(
        learnt_site_rate,
        0.33164477502323253,
    )
    learnt_rate_matrix = res_dict["learnt_rate_matrices"][0]
    np.testing.assert_array_almost_equal(
        learnt_rate_matrix.T,
        pd.DataFrame(
            [
                [-0.5652167201042175, 0.0038684408646076918, 0.003868441330268979, 0.0038684408646076918, 0.5536113381385803],
                [0.018508626148104668, -0.31188488006591797, 0.08713673055171967, 0.08713670074939728, 0.11910282075405121],
                [0.01850862428545952, 0.08713671565055847, -0.31188488006591797, 0.08713671565055847, 0.11910280585289001],
                [0.018508626148104668, 0.08713670074939728, 0.08713673055171967, -0.3118848204612732, 0.11910276859998703],
                [1.1817187070846558, 0.05313650146126747, 0.05313650518655777, 0.053136471658945084, -1.3411281108856201],
            ],
            index=["A", "C", "G", "T", "-"],
            columns=["A", "C", "G", "T", "-"],
        ).T.to_numpy(),
        decimal=1,
    )


def learn_site_rate_matrices(
    tree: Optional[cherryml_io.Tree],
    leaf_states: Dict[str, str],
    alphabet: List[str],
    regularization_rate_matrix: pd.DataFrame,
    regularization_strength: float,
    use_vectorized_implementation: bool,
    vectorized_implementation_device: str = "cpu",
    vectorized_implementation_num_cores: int = 1,
    site_rate_grid: List[float] = [2.0 ** i for i in range(-10, 10)],  # For backwards compatibility reasons. Use get_standard_site_rate_grid() instead.
    site_rate_prior: List[float] = [1.0 for i in range(-10, 10)],  # For backwards compatibility reasons. Use get_standard_site_rate_prior() instead.
    alphabet_for_site_rate_estimation: Optional[List[str]] = None,
    rate_matrix_for_site_rate_estimation: Optional[pd.DataFrame] = None,
    num_epochs: int = 100,
    use_fast_site_rate_implementation: bool = False,
    quantization_grid_num_steps: int = QUANTIZATION_GRID_NUM_STEPS,
    just_run_fast_cherries: bool = False,
) -> Dict:
    """
    Learn a rate matrix per site given the tree and leaf states.

    Args:
        tree: The tree in the CherryML Tree type. If `None`, then FastCherries
            will be used to estimate the tree.
        leaf_states: For each leaf in the tree, the states for that leaf.
        alphabet: List of valid states.
        regularization_rate_matrix: Rate matrix to use to regularize the learnt
            rate matrix.
        regularization_strength: Between 0 and 1. 0 means no regularization at all,
            and 1 means fully regularized model.
        use_vectorized_implementation: Whether to use vectorized implementation.
        vectorized_implementation_device: Whether to use "cpu" or "cuda".
        vectorized_implementation_num_cores: Limit to the number of cores that
            pytorch/numpy will internally use in the vectorized implementation.
            This is NOT about multiprocessing - it's about how many cores
            pytorch/numpy can use to perform e.g. large matrix multiplies.
        site_rate_grid: Grid of site rates to consider. The standard site
            rate grid (used in FastCherries/SiteRM) can be obtained with
            `get_standard_site_rate_grid()`.
        site_rate_prior: Prior probabilities for the site rates. A
            gamma(shape=3, scale=1/3) is typical (what we use in
            FastCherries/SiteRM). This can be obtained with
            `get_standard_site_rate_prior()`.
        alphabet_for_site_rate_estimation: Alphabet for learning the SITE RATES.
            If `None`, then the alphabet for the learnt rate matrix, i.e.
            `alphabet`, will be used. In the FastCherries/SiteRM paper, we have
            observed that for ProteinGym variant effect prediction, it works
            best to *exclude* gaps while estimating site rates (as is standard
            in statistical phylogenetics), but then use gaps when learning the
            rate matrix at the site. In that case, one would use
            `alphabet_for_site_rate_estimation=["A", "C", "G", "T"]` in
            combination with `alphabet=["A", "C", "G", "T", "-"]`.
        rate_matrix_for_site_rate_estimation: If provided, the rate matrix to use
            to estimate site rates. If `None`, then the
            `regularization_rate_matrix` will be used.
        num_epochs: Number of epochs (Adam steps) in the PyTorch optimizer.
        use_fast_site_rate_implementation: Whether to use fast site rate implementation.
            Only used if `use_vectorized_implementation=True` and if tree was
            provided (indeed, if the tree was not provided, then we will use the
            site rates provided by FastCherries).

    Returns:
        A dictionary with the following entries:
            - "learnt_rate_matrices": A List[pd.DataFrame] with the learnt rate matrix per site.
            - "learnt_site_rates": A List[float] with the learnt site rate per site.
            - "learnt_tree": The FastCherries learnt tree (or the provided tree if it was provided).
                It is of type cherryml_io.Tree.
            - "time_...": The time taken by this substep. (They should add up
                to the total runtime).
    """
    profiling_res = {}
    st = time.time()
    if alphabet_for_site_rate_estimation is None:
        alphabet_for_site_rate_estimation = alphabet[:]
    if rate_matrix_for_site_rate_estimation is None:
        rate_matrix_for_site_rate_estimation = regularization_rate_matrix.copy()
    site_rate_grid = site_rate_grid[:]
    site_rate_prior = site_rate_prior[:]
    assert(list(rate_matrix_for_site_rate_estimation.columns) == alphabet_for_site_rate_estimation)
    assert(list(regularization_rate_matrix.columns) == alphabet)
    logger_to_shut_up = logging.getLogger("cherryml.estimation._ratelearn.trainer")
    logger_to_shut_up.setLevel(0)
    profiling_res["time_init_learn_site_rate_matrices"] = time.time() - st

    # Estimate the tree if not provided. We also get the site rates for free.
    st = time.time()
    site_rates_fast_cherries = None
    if tree is None:
        assert rate_matrix_for_site_rate_estimation is not None
        with tempfile.TemporaryDirectory() as temporary_dir:
            # temporary_dir = "temporary_dir"
            # Write the rate matrix
            _rate_matrix_path = os.path.join(temporary_dir, "rate_matrix.txt")
            cherryml_io.write_rate_matrix(rate_matrix_for_site_rate_estimation.to_numpy(), list(rate_matrix_for_site_rate_estimation.columns), _rate_matrix_path)
            # Write the MSA
            _msa_dir = os.path.join(temporary_dir, "msa_dir")
            _msa_path = os.path.join(_msa_dir, "family_0.txt")
            os.makedirs(_msa_dir)
            cherryml_io.write_msa(leaf_states, _msa_path)
            # Create the temporary output dirs
            _output_tree_dir = os.path.join(temporary_dir, "tree")
            _output_site_rates_dir = os.path.join(temporary_dir, "site_rates")
            _output_likelihood_dir = os.path.join(temporary_dir, "lls")
            fast_cherries(
                msa_dir=_msa_dir,
                families=["family_0"],
                rate_matrix_path=_rate_matrix_path,
                num_rate_categories=20,
                max_iters=50,
                num_processes=1,
                verbose=False,
                output_tree_dir=_output_tree_dir,
                output_site_rates_dir=_output_site_rates_dir,
                output_likelihood_dir=_output_likelihood_dir,
            )
            tree = cherryml_io.read_tree(os.path.join(_output_tree_dir, "family_0.txt"))
            site_rates_fast_cherries = cherryml_io.read_site_rates(os.path.join(_output_site_rates_dir, "family_0.txt"))
    else:
        if just_run_fast_cherries:
            raise ValueError("If just_run_fast_cherries is True, then tree must be None.")
    time_estimate_tree = time.time() - st

    time_estimate_site_rate = 0.0
    st = time.time()
    if site_rates_fast_cherries is not None:
        site_rates = site_rates_fast_cherries
    else:
        # Only if we have not estimated the tree with FastCherries, we estimate the site rates here.
        # The advantage of this implementation over FastCherries is that it is completely in-memory;
        # i.e., if tree is provided and site rates are not, then the call to
        # `learn_site_rate_matrices` will involve no disk I/O, making it extremely fast in
        # applications such as learning DNA rate matrices.
        site_rate_fn = _estimate_site_rates_fast if use_fast_site_rate_implementation else _estimate_site_rates
        site_rates = site_rate_fn(
            tree=tree,
            leaf_states=leaf_states,
            site_rate_grid=site_rate_grid,
            site_rate_prior=site_rate_prior,
            rate_matrix=rate_matrix_for_site_rate_estimation,
        )
    time_estimate_site_rate += time.time() - st

    if just_run_fast_cherries:
        learnt_rate_matrices = None
        learnt_rate_matrices__profiling_dict = {}
    else:
        # For this step, we profile it at the subroutine level so we don't do `st = time.time()` as before.
        # Instead, the profiling information comes from the returned dictionary.
        learnt_rate_matrices__res_dict = _learn_site_rate_matrices_given_site_rates_too(
            tree=tree,
            site_rates=site_rates,
            leaf_states=leaf_states,
            alphabet=alphabet,
            regularization_rate_matrix=regularization_rate_matrix,
            regularization_strength=regularization_strength,
            use_vectorized_cherryml_implementation=use_vectorized_implementation,
            vectorized_cherryml_implementation_device=vectorized_implementation_device,
            vectorized_cherryml_implementation_num_cores=vectorized_implementation_num_cores,
            num_epochs=num_epochs,
            quantization_grid_num_steps=quantization_grid_num_steps,
        )
        learnt_rate_matrices = learnt_rate_matrices__res_dict["res"]
        learnt_rate_matrices__profiling_dict = {k: v for k, v in learnt_rate_matrices__res_dict.items() if k.startswith("time_")}

    res = {
        "learnt_rate_matrices": learnt_rate_matrices,
        "learnt_site_rates": site_rates,
        "learnt_tree": tree,
        "time_estimate_tree": time_estimate_tree,
        "time_estimate_site_rate": time_estimate_site_rate,
    }
    res = {**res, **learnt_rate_matrices__profiling_dict}
    return res


def test_learn_site_rate_matrices():
    for use_vectorized_implementation in [False, True]:
        res_dict = learn_site_rate_matrices(
            tree=convert_newick_to_CherryML_Tree("(((leaf_1:1.0,leaf_2:1.0):1.0):1.0,((leaf_3:1.0,leaf_4:1.0):1.0):1.0);"),
            leaf_states={"leaf_1": "C", "leaf_2": "C", "leaf_3": "C", "leaf_4": "G"},
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
            use_vectorized_implementation=use_vectorized_implementation,
        )
        learnt_site_rates = res_dict["learnt_site_rates"]
        np.testing.assert_almost_equal(learnt_site_rates, [0.5])
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


def _get_tree_newick() -> str:
    res = ""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for line in open(os.path.join(dir_path, "_data/_tree_abbreviated_90.txt"), "r"):
        res += line.strip("\n")
    return res


def test_learn_site_rate_matrix_large():
    tree_newick = _get_tree_newick()
    leaves = convert_newick_to_CherryML_Tree(tree_newick).leaves()
    alphabet = ["A", "C", "G", "T"]
    res_dict = learn_site_rate_matrix(
        tree=convert_newick_to_CherryML_Tree(tree_newick),
        # leaf_states={leaf: "A" for leaf in leaves},  # Gives very slow rate matrix
        leaf_states={**{"hg38": "A"}, **{leaf: "C" for leaf in leaves if leaf != "hg38"}},  # Gives reasonable rate matrix
        # leaf_states={leaf: alphabet[len(leaf) % 4] for leaf in leaves},  # Gives rate matrix with massive rates.
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
    )
    learnt_site_rate = res_dict["learnt_site_rate"]
    site_rate_matrix = res_dict["learnt_rate_matrix"]
    pd.testing.assert_frame_equal(
        site_rate_matrix.T,
        pd.DataFrame(
            [
                [-0.24429482221603394, 0.1794303059577942, 0.03243225812911987, 0.03243225812911987],
                [0.0377686470746994, -0.04979332536458969, 0.006012338679283857, 0.006012338679283857],
                [0.033945366740226746, 0.029895860701799393, -0.09651384502649307, 0.03267262130975723],
                [0.033945366740226746, 0.029895860701799393, 0.03267262130975723, -0.09651384502649307],
            ],
            index=["A", "C", "G", "T"],
            columns=["A", "C", "G", "T"],
        ).T,
    )


def test_learn_site_rate_matrices_large():
    tree_newick = _get_tree_newick()
    leaves = convert_newick_to_CherryML_Tree(tree_newick).leaves()
    alphabet = ["A", "C", "G", "T"]
    for use_vectorized_implementation in [False, True]:
        site_rate_matrices = learn_site_rate_matrices(
            tree=convert_newick_to_CherryML_Tree(tree_newick),
            # leaf_states={leaf: "A" for leaf in leaves},  # Gives very slow rate matrix
            leaf_states={**{"hg38": "A"}, **{leaf: "C" for leaf in leaves if leaf != "hg38"}},  # Gives reasonable rate matrix
            # leaf_states={leaf: alphabet[len(leaf) % 4] for leaf in leaves},  # Gives rate matrix with massive rates.
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
            use_vectorized_implementation=use_vectorized_implementation,
        )["learnt_rate_matrices"]
        np.testing.assert_array_almost_equal(
            site_rate_matrices[0, :, :].T,
            pd.DataFrame(
                [
                    [-0.24429482221603394, 0.1794303059577942, 0.03243225812911987, 0.03243225812911987],
                    [0.0377686470746994, -0.04979332536458969, 0.006012338679283857, 0.006012338679283857],
                    [0.033945366740226746, 0.029895860701799393, -0.09651384502649307, 0.03267262130975723],
                    [0.033945366740226746, 0.029895860701799393, 0.03267262130975723, -0.09651384502649307],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ).T.to_numpy(),
            decimal=2,
        )


def _get_msa_example() -> pd.DataFrame:
    """
    Returns a pandas DataFrame of size num_sites x num_species containing real sites in the genome.

    This was obtained as follows:
        ```
        genome_msa = GenomeMSA("zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip")
        X = genome_msa.get_msa(chrom="1", start=144917515, end=144917643, strand="+", tokenize=False)
        pd.DataFrame(X).astype(str).to_csv("X.csv")
        ```
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    res = pd.read_csv(os.path.join(dir_path, "_data/X.csv"), index_col=0)
    # print(f"res = {res}")
    species_names = open(os.path.join(dir_path, "_data/89.txt")).read().split("\n")
    # print(f"species_names = {species_names}")
    res.columns = species_names
    return res


def test_learn_site_rate_matrix_real():
    """
    Tests that SiteRM runs on real MSA.
    """
    msa = _get_msa_example()
    tree_newick = _get_tree_newick()
    # print(f"msa = {msa}")
    times = []
    num_sites = msa.shape[0]
    num_sites = 10
    for site_id in range(num_sites):
        leaf_states = {
            msa.columns[species_id]: msa.iloc[site_id, species_id]
            for species_id in range(msa.shape[1])
        }
        alphabet = ["A", "C", "G", "T"]
        st = time.time()
        res_dict = learn_site_rate_matrix(
            tree=convert_newick_to_CherryML_Tree(tree_newick),
            leaf_states=leaf_states,
            alphabet=alphabet,
            regularization_rate_matrix=pd.DataFrame(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ],
                index=["A", "C", "G", "T"],
                columns=["A", "C", "G", "T"],
            ),
            regularization_strength=0.5,
        )
        learnt_site_rate = res_dict["learnt_site_rate"]
        site_rate_matrix = res_dict["learnt_rate_matrix"]
        times.append(time.time() - st)
    time_per_site = sum(times) / num_sites
    print(f"time for each site = {times}")
    print(f"Average time per site: {time_per_site}")
    # assert(time_per_site <= 0.01)


@pytest.mark.slow
def test_learn_site_rate_matrices_real_vectorized_GOAT():
    """
    Tests that fast vectorized SiteRM implementation runs on real MSA, and gives
    same results as the slow implementation.
    """
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    for use_vectorized_implementation in [True, False]:
        print(f"***** use_vectorized_implementation = {use_vectorized_implementation} *****")
        msa = _get_msa_example()
        leaf_states = {
            msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
            for species_id in range(msa.shape[1])
        }
        tree_newick = _get_tree_newick()
        st = time.time()
        res_dict = learn_site_rate_matrices(
            tree=convert_newick_to_CherryML_Tree(tree_newick),
            leaf_states={
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
            use_vectorized_implementation=use_vectorized_implementation,
            vectorized_implementation_device="cpu",
            vectorized_implementation_num_cores=1,
            site_rate_grid=get_standard_site_rate_grid(),
            site_rate_prior=get_standard_site_rate_prior(),
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
        site_rate_matrices[use_vectorized_implementation] = res_dict["learnt_rate_matrices"]
        for k, v in res_dict.items():
            if k.startswith("time_"):
                print(k, v)
    assert(
        len(site_rate_matrices[False]) == num_sites
    )
    assert(
        len(site_rate_matrices[False]) == num_sites
    )
    for Q_1, Q_2 in zip(site_rate_matrices[False], site_rate_matrices[True]):
        np.testing.assert_array_almost_equal(Q_1, Q_2, decimal=2)


@pytest.mark.slow
def test_learn_site_rate_matrices_real_cuda_GOAT():
    """
    Tests that CUDA SiteRM implementation runs on real MSA, and gives
    same results as the CPU implementation.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    num_sites = 10  # Use all 128 to appreciate speedup of vectorized implementation.
    site_rate_matrices = {}
    for vectorized_implementation_device in ["cpu", "cuda"]:
        print(f"***** vectorized_implementation_device = {vectorized_implementation_device} *****")
        msa = _get_msa_example()
        leaf_states = {
            msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper()
            for species_id in range(msa.shape[1])
        }
        tree_newick = _get_tree_newick()
        st = time.time()
        res_dict = learn_site_rate_matrices(
            tree=convert_newick_to_CherryML_Tree(tree_newick),
            leaf_states={
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
            use_vectorized_implementation=True,
            vectorized_implementation_device=vectorized_implementation_device,
            vectorized_implementation_num_cores=1,
            site_rate_grid=get_standard_site_rate_grid(),
            site_rate_prior=get_standard_site_rate_prior(),
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
        site_rate_matrices[vectorized_implementation_device] = res_dict["learnt_rate_matrices"]
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
def test_fast_site_rate_estimation_real_data():
    """
    Tests that fast site rate implementation gives same results as older one.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    num_sites = 128  # Use all 128 to appreciate speedup of vectorized implementation.
    SITE_RATE_GRID_MOD = 5  # with =5 gives [0.05, 0.2419483326789812, 1.1707799137227792, 5.665364961185359] which gives nice results. Just 4 rate categories! And gives similar results
    NUM_PYTORCH_EPOCHS = 30  # Seems like we can go down to 30 alright.
    GRID_SIZE = 8  # We can go as low as 16 or even 8!

    site_rate_matrices = {}
    for vectorized_implementation_device in ["cpu", "cuda"]:
        print(f"***** vectorized_implementation_device = {vectorized_implementation_device} *****")
        msa = _get_msa_example()
        leaf_states = {
            msa.columns[species_id]: (''.join(list(msa.iloc[:, species_id])))[:num_sites].upper() * 512  # Repeat to get 65536 sites, like in GPN-MSA's batch size.
            for species_id in range(msa.shape[1])
        }
        tree_newick = _get_tree_newick()
        st = time.time()
        res_dict = learn_site_rate_matrices(
            tree=convert_newick_to_CherryML_Tree(tree_newick),
            leaf_states={
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
            use_vectorized_implementation=True,
            use_fast_site_rate_implementation=vectorized_implementation_device == "cuda",
            vectorized_implementation_device="cuda",
            vectorized_implementation_num_cores=1,

            site_rate_grid=[x for (i, x) in enumerate(get_standard_site_rate_grid()) if (i - 10) % SITE_RATE_GRID_MOD == 0],  # Thinned out grid provides speedup. 1 site rate gives ~1.17.
            site_rate_prior=[x for (i, x) in enumerate(get_standard_site_rate_prior()) if (i - 10) % SITE_RATE_GRID_MOD == 0],  # Thinned out grid provides speedup.

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
        site_rate_matrices[vectorized_implementation_device] = res_dict["learnt_rate_matrices"]
        total_time = 0.0
        for k, v in res_dict.items():
            if k.startswith("time_"):
                print(k, v)
                total_time += float(v)
        print(f"total_time {vectorized_implementation_device} = {total_time} (sum of times) vs {time.time() - st} (real time)")
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
def test_fast_site_rate_estimation_real_data_2():
    """
    This test was used to tune the hyperparameters.

    I plot the rate matrices and make sure they are close to the gold standard.

    This test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return

    num_sites = 65536  # Use all 65536 to appreciate speedup of vectorized implementation.
    NUM_SITE_RATES = 20
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
            leaf_states={
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
            use_vectorized_implementation=True,
            use_fast_site_rate_implementation=True,
            vectorized_implementation_device="cuda",
            vectorized_implementation_num_cores=1,
            site_rate_grid=get_standard_site_rate_grid(num_site_rates=NUM_SITE_RATES),
            site_rate_prior=get_standard_site_rate_prior(num_site_rates=NUM_SITE_RATES),
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
