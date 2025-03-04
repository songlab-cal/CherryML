import sys
import logging
import os
import tempfile
import time
import numpy as np
import pandas as pd
from cherryml import io as cherryml_io
import cherryml
from typing import Dict, List, Optional, Tuple
from threadpoolctl import threadpool_limits
from cherryml.estimation._ratelearn import RateMatrixLearner
from cherryml.markov_chain import compute_stationary_distribution
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import multiprocessing
from ._utils import _condition_on_non_gap
from ._cherryml_vectorized import quantized_transitions_mle_vectorized_over_sites
from cherryml import markov_chain
import pytest
from cherryml import caching
from cherryml import utils
import torch

GAP_CHARACTER = "-"  # NOTE: This is specific to some applications.


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _quantized_transitions_mle(
    count_matrices: List[Tuple[float, pd.DataFrame]],
    initialization: Optional[np.array],
    learning_rate: float = 1e-1,
    num_epochs: int = 2000,
    do_adam: bool = True,
    loss_normalization: bool = True,
    OMP_NUM_THREADS: Optional[int] = 1,
    OPENBLAS_NUM_THREADS: Optional[int] = 1,
    TORCH_NUM_THREAS: Optional[int] = 1,
    return_best_iter: bool = True,
    rate_matrix_parameterization: str = "pande_reversible",
) -> pd.DataFrame:
    """
    In-memory minimal version of CherryML's `quantized_transitions_mle`
    """
    states = list(count_matrices[0][1].index)
    with threadpool_limits(limits=OPENBLAS_NUM_THREADS, user_api="blas"):
        with threadpool_limits(limits=OMP_NUM_THREADS, user_api="openmp"):
            torch.set_num_threads(TORCH_NUM_THREAS)  # Ensure PyTorch respects the limit
            with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
                rate_matrix_learner = RateMatrixLearner(
                    branches=[x[0] for x in count_matrices],
                    mats=[x[1].to_numpy() for x in count_matrices],
                    states=states,
                    output_dir=output_rate_matrix_dir,
                    stationnary_distribution=None,
                    mask=None,
                    rate_matrix_parameterization=rate_matrix_parameterization,
                    device="cpu",
                    initialization=initialization,
                    skip_writing_to_output_dir=True,
                )
                rate_matrix_learner.train(
                    lr=learning_rate,
                    num_epochs=num_epochs,
                    do_adam=do_adam,
                    loss_normalization=loss_normalization,
                    return_best_iter=return_best_iter,
                )
                res = rate_matrix_learner.get_learnt_rate_matrix()
                return res


def _get_cherry_transitions(
    tree: cherryml_io.Tree,
    msa: Dict[str, str],
) -> List[Tuple[str, str, float]]:
    """
    Note: copy-pasta from CherryML...
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
            f"_test_get_cherry_transitions failed.\n"
            f"expected_cherries={expected_cherries}\n"
            f"cherries={cherries}"
        )


def _get_raw_count_matrices(
    transitions: List[Tuple[str, str, float]],
    quantization_points_sorted: List[float],
    alphabet: List[str],
    include_reverse_transitions: bool = True,
) -> np.array:
    """
    Returns 4D numpy array with raw counts.

    Assumes an ASCII alphabet.
    """
    st = time.time()
    # Step 1) Translate to int
    alphabet_to_int = {chr(i): -1 for i in range(128)}
    for i, char in enumerate(alphabet):
        alphabet_to_int[char] = i
    # Translate transitions to integer representation
    transitions_int = [
        (
            [alphabet_to_int[s] for s in x],
            [alphabet_to_int[s] for s in y],
            t
        )
        for x, y, t in transitions
    ]
    # print(f"time translate: {time.time() - st}")

    st = time.time()
    num_quantization_points = len(quantization_points_sorted)
    quantization_points_sorted = np.array(quantization_points_sorted)
    num_sites = len(transitions[0][0])
    num_states = len(alphabet)
    raw_count_matrices = np.zeros(
        shape=(
            num_sites,  # L
            num_quantization_points,  # B
            num_states,  # S
            num_states  # S
        )
    )
    st = time.time()
    bs = [
        cherryml.utils.quantization_idx(
            branch_length=t,
            quantization_points_sorted=quantization_points_sorted
        )
        for (x, y, t) in transitions_int
    ]
    # print(f"time quantize branch lengths: {time.time() - st}")
    st = time.time()
    for i, (x, y, t) in enumerate(transitions_int):
        assert(len(x) == num_sites and len(y) == num_sites)
        # b = cherryml.utils.quantization_idx(
        #     branch_length=t,
        #     quantization_points_sorted=quantization_points_sorted
        # )
        b = bs[i]
        if b is None:
            if t < quantization_points_sorted[0]:
                # Branch length is too short (e.g. 5e-9 ~ 0), skip.
                continue
            elif t > quantization_points_sorted[-1]:
                # Branch length is too long, skip
                continue
        for l in range(num_sites):
            x_l, y_l = x[l], y[l]
            if x_l >= 0 and y_l >= 0:
                raw_count_matrices[l, b, x_l, y_l] += 1.0
    if include_reverse_transitions:
        raw_count_matrices = (raw_count_matrices + raw_count_matrices.transpose(0, 1, 3, 2)) / 2.0
    # print(f"time count: {time.time() - st}")
    # assert(False)
    return raw_count_matrices


def test_get_raw_count_matrices():
    transitions = [
        ("AG", "BH", 0.35 + 0.36),
        ("EG", "FH", 0.49 + 0.410),
        ("CG", "DG", 0.17 + 0.28 + 0.01 + 0.02),
    ]
    alphabet = ["-", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    raw_count_matrices = _get_raw_count_matrices(
        transitions=transitions,
        quantization_points_sorted=[0.40, 0.80, 2.0],
        alphabet=alphabet,
    )
    alphabet_to_int = {alphabet[i]: i for i in range(len(alphabet))}
    expected_raw_count_matrices = np.zeros(
        shape=(2, 3, 14, 14)
    )
    expected_raw_count_matrices[0, 0, alphabet_to_int["C"], alphabet_to_int["D"]] += 0.5
    expected_raw_count_matrices[0, 0, alphabet_to_int["D"], alphabet_to_int["C"]] += 0.5
    expected_raw_count_matrices[0, 1, alphabet_to_int["A"], alphabet_to_int["B"]] += 0.5
    expected_raw_count_matrices[0, 1, alphabet_to_int["B"], alphabet_to_int["A"]] += 0.5
    expected_raw_count_matrices[0, 1, alphabet_to_int["E"], alphabet_to_int["F"]] += 0.5
    expected_raw_count_matrices[0, 1, alphabet_to_int["F"], alphabet_to_int["E"]] += 0.5
    expected_raw_count_matrices[1, 0, alphabet_to_int["G"], alphabet_to_int["G"]] += 0.5
    expected_raw_count_matrices[1, 0, alphabet_to_int["G"], alphabet_to_int["G"]] += 0.5
    expected_raw_count_matrices[1, 1, alphabet_to_int["G"], alphabet_to_int["H"]] += 0.5
    expected_raw_count_matrices[1, 1, alphabet_to_int["H"], alphabet_to_int["G"]] += 0.5
    expected_raw_count_matrices[1, 1, alphabet_to_int["G"], alphabet_to_int["H"]] += 0.5
    expected_raw_count_matrices[1, 1, alphabet_to_int["H"], alphabet_to_int["G"]] += 0.5
    np.testing.assert_almost_equal(
        raw_count_matrices, expected_raw_count_matrices
    )


def test_get_raw_count_matrices_no_reverse_transitions():
    transitions = [
        ("AG", "BH", 0.35 + 0.36),
        ("EG", "FH", 0.49 + 0.410),
        ("CG", "DG", 0.17 + 0.28 + 0.01 + 0.02),
    ]
    alphabet = ["-", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    raw_count_matrices = _get_raw_count_matrices(
        transitions=transitions,
        quantization_points_sorted=[0.40, 0.80, 2.0],
        alphabet=alphabet,
        include_reverse_transitions=False,
    )
    alphabet_to_int = {alphabet[i]: i for i in range(len(alphabet))}
    expected_raw_count_matrices = np.zeros(
        shape=(2, 3, 14, 14)
    )
    expected_raw_count_matrices[0, 0, alphabet_to_int["C"], alphabet_to_int["D"]] += 1.0
    expected_raw_count_matrices[0, 1, alphabet_to_int["A"], alphabet_to_int["B"]] += 1.0
    expected_raw_count_matrices[0, 1, alphabet_to_int["E"], alphabet_to_int["F"]] += 1.0
    expected_raw_count_matrices[1, 0, alphabet_to_int["G"], alphabet_to_int["G"]] += 1.0
    expected_raw_count_matrices[1, 1, alphabet_to_int["G"], alphabet_to_int["H"]] += 1.0
    expected_raw_count_matrices[1, 1, alphabet_to_int["G"], alphabet_to_int["H"]] += 1.0
    np.testing.assert_almost_equal(
        raw_count_matrices, expected_raw_count_matrices
    )


def _get_count_prior_probability_matrices(
    rate_matrix: np.array,
    quantization_points_sorted: List[float],
) -> np.array:
    num_quantization_points = len(quantization_points_sorted)
    num_states = rate_matrix.shape[0]
    stationary_distribution = compute_stationary_distribution(
        rate_matrix=rate_matrix,
    )
    matrix_exponentials = cherryml._siterm._utils.matrix_exponential_reversible(
        rate_matrix=rate_matrix,
        exponents=quantization_points_sorted,
    )
    count_prior_probability_matrices = np.zeros(
        shape=(
            num_quantization_points,  # B
            num_states,  # S
            num_states  # S
        )
    )
    for b in range(num_quantization_points):
        for row_id in range(num_states):
            count_prior_probability_matrices[b, row_id, :] = \
                stationary_distribution[row_id] * matrix_exponentials[b, row_id, :]
    for b in range(num_quantization_points):
        matrix_sum = float(count_prior_probability_matrices[b, :, :].sum().sum())
        if abs(matrix_sum - 1.0) > 1e-6:
            raise ValueError(
                f"count_prior_probability_matrices[b, :, :] does not add up to 1!"
            )
    return count_prior_probability_matrices


def test_get_count_prior_probability_matrices():
    """
    Regression test
    """
    count_prior_probability_matrices = _get_count_prior_probability_matrices(
        rate_matrix=np.array(
            [
                [-0.5, 0.5],
                [1.0, -1.0],
            ]
        ),
        quantization_points_sorted=[0.1, 1.0, 10.0],
    )
    expected_count_prior_probability_matrices = np.array(
        [
            [
                [0.63571288, 0.03095378],
                [0.03095378, 0.30237955],
            ],
            [
                [0.49402892, 0.17263774],
                [0.17263774, 0.16069559],
            ],
            [
                [0.44444451, 0.22222215],
                [0.22222215, 0.11111118],
            ]
        ]
    )
    np.testing.assert_almost_equal(
        expected_count_prior_probability_matrices,
        count_prior_probability_matrices,
    )


def _get_edge_transitions(
    tree: cherryml_io.Tree,
    msa: Dict[str, str]
) -> List[Tuple[str, str, float]]:
    assert(
        sorted(tree.nodes()) ==
        sorted(msa.keys())
    )
    transitions = [
        (msa[u], msa[v], t)
        for (u, v, t) in tree.edges()
    ]
    return transitions


def test_get_edge_transitions():
    tree = _get_test_tree_2(node_prefix="node-", equal_edge_lengths=False)
    msa = {
        'node-0': 'DG', 'node-1': 'DV', 'node-2': 'TG',
        'node-3': 'DV', 'node-4': 'DV', 'node-5': 'TG',
        'node-6': 'TG', 'node-7': 'DV', 'node-8': 'AV',
        'node-9': 'TV', 'node-10': 'DV', 'node-11': 'TG',
        'node-12': 'SG', 'node-13': 'TG', 'node-14': 'TG'
    }
    transitions = _get_edge_transitions(tree=tree, msa=msa)
    expected_transitions = [
        ("DG", "DV", 0.01),
        ("DG", "TG", 0.02),
        ("DV", "DV", 0.13),
        ("DV", "DV", 0.14),
        ("TG", "TG", 0.25),
        ("TG", "TG", 0.26),
        ("DV", "DV", 0.37),
        ("DV", "AV", 0.38),
        ("DV", "TV", 0.49),
        ("DV", "DV", 0.410),
        ("TG", "TG", 0.511),
        ("TG", "SG", 0.512),
        ("TG", "TG", 0.613),
        ("TG", "TG", 0.614),
    ]
    if sorted(transitions) != sorted(expected_transitions):
        raise ValueError(
            f"_test_get_edge_transitions failed.\n"
            f"expected_transitions={expected_transitions}\n"
            f"transitions={transitions}"
        )


def _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
    # Key arguments
    tree: cherryml_io.Tree,
    site_rates: List[float],  # Only used for the pseudocounts (i.e. if lambda = 0 then this has no effect on the output)
    msa: Dict[str, str],
    alphabet: List[str],
    # Hyperparameters
    regularization_strength: float,  # lambda
    regularization_rate_matrix: np.array,  # Q_0
    quantization_points: List[float],
    optimization_num_epochs: int,
    transitions_strategy: str = "cherry++",
    include_reverse_transitions: bool = True,
    rate_matrix_parameterization: str = "pande_reversible",
    log_dir: Optional[str] = None,
    plot_site_specific_rate_matrices: int = 0,
    use_vectorized_cherryml_implementation: bool = False,
    vectorized_cherryml_implementation_device: str = "cpu",
    vectorized_cherryml_implementation_num_cores: int = 1,
) -> Dict:
    """
    Given trees, site rates, and an MSA, estimate site-specific rate matrices.

    Returns:
        A dictionary with:
        "res": The resulting np.array
        "time_...": The time taken for this substep. They should add up to the total time.
    """
    profiling_res = {}
    st = time.time()
    logger = logging.getLogger(__name__)

    quantization_points_sorted = sorted(quantization_points)
    del quantization_points

    if transitions_strategy == "cherry++":
        assert(sorted(tree.leaves()) == sorted(msa.keys()))
        transitions = _get_cherry_transitions(tree=tree, msa=msa)
    elif transitions_strategy == "edges":
        assert(sorted(tree.nodes()) == sorted(msa.keys()))
        transitions = _get_edge_transitions(tree=tree, msa=msa)
    else:
        raise ValueError(f"Unknown transitions_strategy: {transitions_strategy}")
    num_sites = len(transitions[0][0])

    num_quantization_points = len(quantization_points_sorted)
    num_states = len(alphabet)

    profiling_res["time_get_transitions"] = time.time() - st

    # First we get the raw counts
    st = time.time()
    raw_count_matrices = _get_raw_count_matrices(
        transitions=transitions,
        quantization_points_sorted=quantization_points_sorted,
        alphabet=alphabet,
        include_reverse_transitions=include_reverse_transitions,
    )
    profiling_res["time_get_raw_count_matrices"] = time.time() - st

    # Now we need to get the pseudocounts, which involves getting the matrix
    # exponentials first.
    st = time.time()
    count_prior_probability_matrices = _get_count_prior_probability_matrices(
        rate_matrix=regularization_rate_matrix,
        quantization_points_sorted=quantization_points_sorted,
    )
    profiling_res["time_get_count_prior_probability_matrices"] = time.time() - st
    st = time.time()
    pseudocount_matrices = np.zeros(
        shape=(
            num_sites,  # L
            num_quantization_points,  # B
            num_states,  # S
            num_states  # S
        )
    )
    l1_norms = raw_count_matrices.sum(axis=(2, 3))   # Vectorized precomputation makes it much faster.
    for l in range(num_sites):
        for b in range(num_quantization_points):
            # l1_norm = raw_count_matrices[l, b].sum().sum()  # This is slower that the vectorized precomputation with `l1_norms = raw_count_matrices.sum(axis=(2, 3))`.
            l1_norm = l1_norms[l, b]

            if l1_norm <= 0:
                continue
            # We need to adjust b by the site rate
            t = quantization_points_sorted[b]
            b_adjusted = cherryml.utils.quantization_idx(
                branch_length=t * site_rates[l],
                quantization_points_sorted=quantization_points_sorted
            )
            if b_adjusted is None:
                if t * site_rates[l] > quantization_points_sorted[-1]:
                    # It's a very large time. Can happen for very
                    # quickly evolving sites where the generalized cherry
                    # is very long.
                    # Since the chain has mixed at this point, we'll just
                    # deal with this by setting b_adjusted=last bucket,
                    # which is mathematically almost exactly equal to
                    # having used the right time.
                    b_adjusted = len(quantization_points_sorted) - 1
                else:
                    # It's a very small time. This is unusual, I think.
                    # If for any reason I need to relax this check, I can just
                    # set b_adjusted = 0. (DONE -- needed for DMS dataset!)
                    assert(t * site_rates[l] < quantization_points_sorted[0])
                    # raise ValueError(
                    #     "Trying to add pseudocounts for a very small time:"
                    #     f" t = {t}, site_rates[l] = {site_rates[l]}, "
                    #     f"l1_norm = {l1_norm}."
                    # )
                    b_adjusted = 0
            pseudocount_matrices[l, b, :, :] = (
                l1_norm
                * count_prior_probability_matrices[b_adjusted, :, :]
            )

    if (
        abs(float(raw_count_matrices.sum()) - float(pseudocount_matrices.sum())) > 0.4
        and abs(float(raw_count_matrices.sum())/float(pseudocount_matrices.sum()) - 1.0) > 1e-6
    ):
        raise ValueError(
            "Raw counts matrix and pseudocounts matrix have different counts: "
            f"{raw_count_matrices.sum()} vs {pseudocount_matrices.sum()}"
        )
    profiling_res["time_get_pseudocount_matrices"] = time.time() - st
    # print(f'profiling_res["time_get_pseudocount_matrices"] = {profiling_res["time_get_pseudocount_matrices"]}')
    # assert(False)

    st = time.time()
    count_matrices = (
        raw_count_matrices * (1.0 - regularization_strength)
        + pseudocount_matrices * regularization_strength
    )
    site_specific_rate_matrices = np.zeros(
        shape=(
            num_sites,
            num_states,
            num_states,
        )
    )
    profiling_res["time_get_count_matrices"] = time.time() - st
    if use_vectorized_cherryml_implementation:
        st = time.time()
        initialization = np.zeros(shape=(num_sites, num_states, num_states))
        for l in range(num_sites):
            initialization[l, :, :] = regularization_rate_matrix * site_rates[l]

        # Now get the compactified count matrices.
        # First we need to figure how large it will be
        count_matrices_sums = count_matrices.sum(axis=(2, 3))  # Much faster compactification via precomputed sums!
        count_matrices_with_times = [
            [
                (
                    quantization_points_sorted[b],
                    count_matrices[l, b, :, :],
                )
                for b in range(num_quantization_points)
                if float(count_matrices_sums[l, b]) > 0  # Again, we can filter out matrices with zero counts!
            ]
            for l in range(num_sites)
        ]
        effective_number_of_time_buckets_per_site = [
            len(count_matrices_with_times[l])
            for l in range(num_sites)
        ]
        effective_number_of_time_buckets = max(effective_number_of_time_buckets_per_site)
        count_matrices_compactified = np.zeros(shape=(num_sites, effective_number_of_time_buckets, num_states, num_states))
        times_compactified = [[1.0 for j in range(effective_number_of_time_buckets)] for i in range(num_sites)]
        for l in range(num_sites):
            for b_id, (t, count_matrix) in enumerate(count_matrices_with_times[l]):
                count_matrices_compactified[l, b_id, :, :] = count_matrix
                times_compactified[l][b_id] = t
        profiling_res["time_get_count_matrices_compactified"] = time.time() - st

        quantized_transitions_mle_vectorized_over_sites__res_dict = quantized_transitions_mle_vectorized_over_sites(
            # counts=count_matrices,
            # times=np.array([quantization_points_sorted for _ in range(num_sites)]),
            counts=count_matrices_compactified,
            times=times_compactified,
            num_epochs=optimization_num_epochs,
            initialization=initialization,
            device=vectorized_cherryml_implementation_device,
            num_cores=vectorized_cherryml_implementation_num_cores,
        )

        site_specific_rate_matrices = quantized_transitions_mle_vectorized_over_sites__res_dict["res"]
        # Add the profiling times of the subroutine.
        quantized_transitions_mle_vectorized_over_sites__profiling = {
            k: v for (k, v) in quantized_transitions_mle_vectorized_over_sites__res_dict.items() if k.startswith("time_")
        }
        for k, v in quantized_transitions_mle_vectorized_over_sites__profiling.items():
            assert k not in profiling_res
            profiling_res[k] = v
    else:
        st = time.time()
        for l in range(num_sites):
            # logger.info(
            #     f"Learning rate matrix for site: {l + 1} / {num_sites}"
            # )
            count_matrices_with_times = [
                (
                    quantization_points_sorted[b],
                    pd.DataFrame(
                        count_matrices[l, b, :, :],
                        index=alphabet,
                        columns=alphabet,
                    )
                )
                for b in range(num_quantization_points)
                if float(count_matrices[l, b, :, :].sum().sum()) > 0  # Again, we can filter out matrices with zero counts!
            ]
            if len(count_matrices_with_times) == 0:
                # Can happen for positions with only gaps...
                # We just use the prior.
                site_specific_rate_matrices[l, :, :] = regularization_rate_matrix * site_rates[l]
            else:
                site_specific_rate_matrices[l, :, :] = _quantized_transitions_mle(
                    count_matrices=count_matrices_with_times,
                    initialization=regularization_rate_matrix * site_rates[l],
                    learning_rate=1e-1,
                    num_epochs=optimization_num_epochs,
                    do_adam=True,
                    loss_normalization=True,
                    OMP_NUM_THREADS=1,
                    OPENBLAS_NUM_THREADS=1,
                    return_best_iter=True,
                    rate_matrix_parameterization=rate_matrix_parameterization,
                )
        profiling_res["time__quantized_transitions_mle"] = time.time() - st
    st = time.time()
    if log_dir is not None and plot_site_specific_rate_matrices > 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        for i in range(-1, min(num_sites, plot_site_specific_rate_matrices), 1):  # -1 is for the regularization_rate_matrix
            cmap = sns.diverging_palette(0, 240, s=75, l=50, n=500, center="light")
            sns.heatmap(
                site_specific_rate_matrices[i, :, :] if i >= 0 else regularization_rate_matrix,
                xticklabels=alphabet,
                yticklabels=alphabet,
                cmap=cmap,
                center=0,
            )
            if i >= 0:
                plt.title(f"Rate matrix for site: {i} (0-based)")
            else:
                plt.title("Regularization rate matrix")
            fig_path = os.path.join(
                log_dir,
                f"{i}_rate_matrix.png" if i >= 0 else "regularization_rate_matrix.png"
            )
            plt.savefig(fig_path)
            plt.close()
            if i >= 0:
                sns.heatmap(
                    np.log1p(raw_count_matrices[i, :, :, :].sum(axis=0)),
                    xticklabels=alphabet,
                    yticklabels=alphabet,
                    cmap=cmap,
                    center=0,
                )
                plt.title(f"Log1p counts for site: {i} (0-based)")
                fig_path = os.path.join(
                    log_dir,
                    f"{i}_log1p_counts.png",
                )
                plt.savefig(fig_path)
                plt.close()

                # Now raw counts.
                sns.heatmap(
                    raw_count_matrices[i, :, :, :].sum(axis=0),
                    xticklabels=alphabet,
                    yticklabels=alphabet,
                    cmap=cmap,
                    center=0,
                )
                plt.title(f"Raw counts for site: {i} (0-based)")
                fig_path = os.path.join(
                    log_dir,
                    f"{i}_raw_counts.png",
                )
                plt.savefig(fig_path)
                plt.close()
    profiling_res["time_plotting"] = time.time() - st
    res = {
        "res": site_specific_rate_matrices,
    }
    res = {**res, **profiling_res}
    return res


def _get_test_tree_2(node_prefix: str = "", equal_edge_lengths: bool = True) -> cherryml_io.Tree:
    tree = cherryml_io.Tree()
    nodes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    nodes = [node_prefix + x for x in nodes]
    tree.add_nodes(nodes)
    if equal_edge_lengths:
        edges = [
            ("0", "1", 0.1),
            ("0", "2", 0.1),
            ("1", "3", 0.1),
            ("1", "4", 0.1),
            ("2", "5", 0.1),
            ("2", "6", 0.1),
            ("3", "7", 0.1),
            ("3", "8", 0.1),
            ("4", "9", 0.1),
            ("4", "10", 0.1),
            ("5", "11", 0.1),
            ("5", "12", 0.1),
            ("6", "13", 0.1),
            ("6", "14", 0.1),
        ]
    else:
        edges = [
            ("0", "1", 0.01),
            ("0", "2", 0.02),
            ("1", "3", 0.13),
            ("1", "4", 0.14),
            ("2", "5", 0.25),
            ("2", "6", 0.26),
            ("3", "7", 0.37),
            ("3", "8", 0.38),
            ("4", "9", 0.49),
            ("4", "10", 0.410),
            ("5", "11", 0.511),
            ("5", "12", 0.512),
            ("6", "13", 0.613),
            ("6", "14", 0.614),
        ]
    edges = [
        (node_prefix + u, node_prefix + v, t) for (u, v, t) in edges
    ]
    tree.add_edges(edges)
    return tree


def _get_test_msa_2(node_prefix: str = "") -> Dict[str, str]:
    msa = {
        "7": "DV",
        "8": "AV",
        "9": "TV",
        "10": "DV",
        "11": "TG",
        "12": "SG",
        "13": "TG",
        "14": "TG",
    }
    msa = {
        node_prefix + key: value
        for (key, value) in msa.items()
    }
    return msa


def _get_test_msa_all_missing(node_prefix: str = "") -> Dict[str, str]:
    msa = {
        "7": "--",
        "8": "--",
        "9": "--",
        "10": "--",
        "11": "--",
        "12": "--",
        "13": "--",
        "14": "--",
    }
    msa = {
        node_prefix + key: value
        for (key, value) in msa.items()
    }
    return msa


def _get_test_msa_some_all_missing(node_prefix: str = "") -> Dict[str, str]:
    msa = {
        "7": "-V",
        "8": "-V",
        "9": "-V",
        "10": "-V",
        "11": "-G",
        "12": "-G",
        "13": "-G",
        "14": "-G",
    }
    msa = {
        node_prefix + key: value
        for (key, value) in msa.items()
    }
    return msa


def _maximum_parsimony(
    tree: cherryml_io.Tree,
    msa: Dict[str, str]
) -> Dict[str, str]:
    assert(sorted(tree.leaves()) == sorted(msa.keys()))
    logger = logging.getLogger(__name__)

    # check if the binary exists
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "_maximum_parsimony"
    )
    cpp_path = os.path.join(dir_path, "maximum_parsimony.cpp")
    bin_path = os.path.join(dir_path, "maximum_parsimony")
    if not os.path.exists(bin_path):
        command = f"g++ -std=c++11 -O3 -o {bin_path} {cpp_path}"
        logger.info(f"Going to compile maximum_parsimony.cpp with: {command}")
        os.system(command)
        if not os.path.exists(bin_path):
            raise Exception(
                "Couldn't compile maximum_parsimony.cpp. "
                f"Command: {command}"
            )

    with tempfile.NamedTemporaryFile("w") as tree_file:
        tree_path = tree_file.name
        # Write out the tree
        tree_nodes = tree.nodes()
        tree_node_to_int_dict = {tree_nodes[i]: i for i in range(len(tree_nodes))}
        tree_str = f"{len(tree_nodes)}\n" + "\n".join([f"{tree_node_to_int_dict[u]} {tree_node_to_int_dict[v]}" for (u, v, _) in tree.edges()])
        tree_file.write(tree_str)
        tree_file.flush()
        with tempfile.NamedTemporaryFile("w") as sequences_file:
            sequences_path = sequences_file.name
            # Write out the MSA
            sequences_str = f"{len(tree.leaves())}\n" + "\n".join(
                [f"{tree_node_to_int_dict[seq_name]} {seq}" for (seq_name, seq) in msa.items()]
            )
            sequences_file.write(sequences_str)
            sequences_file.flush()
            with tempfile.NamedTemporaryFile("w") as solution_file:
                solution_path = solution_file.name
                command = f"{bin_path} {tree_path} {sequences_path} {solution_path}"
                os.system(command)
                with open(solution_path, "r") as solution_file_read:
                    mp_string = solution_file_read.read()

                    def parse_mp_string(mp_string: str) -> Dict[str, str]:
                        """
                        Parse the MP string into an MSA
                        """
                        lines = mp_string.split("\n")
                        if lines[-1] == "":
                            # Strip endline at end of file
                            lines = lines[:-1]
                        assert(len(lines) - 1 == int(lines[0]))
                        res = {}
                        for line in lines[1:]:
                            i_str, seq_str = line.split(" ")
                            res[tree_nodes[int(i_str)]] = seq_str
                        return res
                    mp_msa = parse_mp_string(mp_string)
                    return mp_msa


def test_maximum_parsimony():
    tree = _get_test_tree_2(node_prefix="node-")
    msa = _get_test_msa_2(node_prefix="node-")
    mp_msa = _maximum_parsimony(tree=tree, msa=msa)
    assert(
        mp_msa == {
            'node-0': 'DG', 'node-1': 'DV', 'node-2': 'TG',
            'node-3': 'DV', 'node-4': 'DV', 'node-5': 'TG',
            'node-6': 'TG', 'node-7': 'DV', 'node-8': 'AV',
            'node-9': 'TV', 'node-10': 'DV', 'node-11': 'TG',
            'node-12': 'SG', 'node-13': 'TG', 'node-14': 'TG'
        }
    )


def evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
    transitions: List[Tuple[str, str, float]],
    site_specific_rate_matrices: np.array,
    alphabet: List[str],
    condition_on_non_gap: bool = False,
) -> List[List[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the
    site-specific rate matrix model.

    It is assumed that the site_specific_rate_matrices are reversible.

    The log-likelihood under the site-specific rate matrix model is given by:
    P(y[i] | x[i], t) = log( exp(rate_matrix[i] * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        site_specific_rate_matrices: The site-specific rate matrices of the model;
            3D array indexed by [site_id, state_1, state_2]
        alphabet: Alphabet (of states).
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
    Returns:
        lls: The log-likelihood of each transition.
    """
    assert(len(transitions[0][0]) == site_specific_rate_matrices.shape[0])
    num_sites = len(transitions[0][0])
    matrix_exponentials = [
        cherryml._siterm._utils.matrix_exponential_reversible(
            rate_matrix=site_specific_rate_matrices[site_id, :, :],
            exponents=[t for (x, y, t) in transitions],
        )
        for site_id in range(num_sites)
    ]  # Indexed by [site_id][transition_id, :, :]
    res = []
    for i, (x, y, t) in enumerate(transitions):
        if len(x) != len(y):
            raise ValueError(
                f"Transition has two sequences of different lengths: {x}, {y}."
            )
        mexp_dfs = [
            pd.DataFrame(
                matrix_exponentials[site_id][i, :, :],
                index=alphabet,
                columns=alphabet,
            )
            for site_id in range(num_sites)
        ]  # Indexed by [site_id][:, :]
        if condition_on_non_gap:
            mexp_dfs = [
                _condition_on_non_gap(
                    mexp_dfs[site_id]
                )
                for site_id in range(num_sites)
            ]  # Indexed by [site_id][:, :]
        assert len(x) == len(y)
        assert len(x) == site_specific_rate_matrices.shape[0]
        lls = [
            np.log(
                mexp_dfs[site_id].at[x[site_id], y[site_id]]
            )
            for site_id in range(len(x))
        ]
        res.append(lls)
    return res


def evaluate_lg_model_transitions_log_likelihood(
    transitions: List[Tuple[str, str, float]],
    site_specific_rate_matrices: np.array,
    alphabet: str,
) -> List[float]:
    """
    Compute the log-likelihood of the given transitions under the
    site-specific rate matrix model.

    It is assumed that the site_specific_rate_matrices are reversible.

    The log-likelihood under the site-specific rate matrix model is given by:
    P(y | x, t) = sum_i log( exp(rate_matrix[i] * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        site_specific_rate_matrices: The site-specific rate matrices of the model;
            3D array indexed by [site_id, state_1, state_2]
        alphabet: Alphabet (of states).
    Returns:
        lls: The log-likelihood of each transition.
    """
    lls_per_site = evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
        transitions=transitions,
        site_specific_rate_matrices=site_specific_rate_matrices,
        alphabet=alphabet,
    )
    lls = [sum(x) for x in lls_per_site]
    return lls


def _get_test_expected_orderings(t: float = 1.0) -> List[List[Tuple[str, str, float]]]:
    expected_orderings = [
        # [  # Problematic bc S has low prob under the stationary distribution while T has higher, requires extremely small t to pass.
        #     ("SG", "SG", t),  # bc no mutations
        #     ("SG", "TG", t),  # bc first site (less conserved) mutated into something seen very close
        # ],
        [
            ("SG", "TG", t),  # bc first site (less conserved) mutated into something seen very close
            ("SG", "DG", t),  # bc first site (less conserved) mutated into something seen further away
        ],
        [
            ("SG", "DG", t),  # bc first site (less conserved) mutated into something seen further away
            ("SG", "AG", t),  # bc first site (less conserved) mutated into something seen far away
        ],
        [
            ("SG", "AG", t),  # bc first site (less conserved) mutated into something seen far away
            ("SG", "-G", t),  # bc first site (less conserved) mutated into something unseen
        ],
        [
            ("SG", "SG", t),  # bc no mutations
            ("SG", "SV", t),  # bc second site (more conserved) mutated into something seen
        ],
        [
            ("SG", "SV", t),  # bc second site (more conserved) mutated into something seen
            ("SG", "S-", t),  # bc second site (most conserved) mutated into something unseen
        ],
        [
            ("SG", "-G", t),  # bc first site (less conserved) mutated into something unseen
            ("SG", "S-", t),  # bc second site (most conserved) mutated into something unseen
        ],
        [
            ("SG", "S-", t),  # bc second site (most conserved) mutated into something unseen
            ("SG", "--", t),  # bc both sites mutate into something unseen
        ],
    ]
    return expected_orderings


def test_estimate_site_specific_rate_matrices_given_tree_and_site_rates(plot: bool = False):
    tree = _get_test_tree_2(node_prefix="node-")
    msa = _get_test_msa_2(node_prefix="node-")
    mp_msa = _maximum_parsimony(
        tree=tree,
        msa=msa,
    )
    site_rates = [2.0, 0.5]
    alphabet = ["A", "D", "G", "S", "T", "V", "-"]

    quantization_grid_center = 0.03
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 64
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    regularization_rate_matrix = np.zeros(
        shape=(
            len(alphabet),
            len(alphabet)
        )
    )
    for i in range(len(alphabet)):
        regularization_rate_matrix[i, i] = -1
        for j in range(len(alphabet)):
            if j != i:
                regularization_rate_matrix[i, j] = 1/(len(alphabet) - 1)

    for (transitions_strategy, include_reverse_transitions, curr_msa) in [
        ("cherry++", True, msa),
        ("edges", False, mp_msa),
        ("edges", True, mp_msa)
    ]:
        site_specific_rate_matrices = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
            tree=tree,
            site_rates=site_rates,
            msa=curr_msa,
            alphabet=alphabet,
            # Hyperparameters
            regularization_strength=0.5,
            regularization_rate_matrix=regularization_rate_matrix,
            quantization_points=quantization_points,
            optimization_num_epochs=500,
            transitions_strategy=transitions_strategy,
            include_reverse_transitions=include_reverse_transitions,
        )["res"]

        # Check the log-likelihood of the GEMME examples & more
        expected_orderings = _get_test_expected_orderings()
        for transition_pair in expected_orderings:
            per_site_tlls = evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
                transitions=transition_pair,
                site_specific_rate_matrices=site_specific_rate_matrices,
                alphabet=alphabet,
            )
            if sum(per_site_tlls[0]) > sum(per_site_tlls[1]) + 1e-4:
                print(f"PASSED: {(transition_pair[0], per_site_tlls[0])} > {(transition_pair[1], per_site_tlls[1])}")
            else:
                print(f"FAILED: {(transition_pair[0], per_site_tlls[0])} > {(transition_pair[1], per_site_tlls[1])}")

        if plot:
            for i in range(2):
                # Create a diverging colormap from blue to white to red
                cmap = sns.diverging_palette(0, 240, s=75, l=50, n=500, center="light")
                sns.heatmap(
                    site_specific_rate_matrices[i],
                    xticklabels=alphabet,
                    yticklabels=alphabet,
                    cmap=cmap,
                    center=0,
                )
                plt.savefig(f"site_specific_rate_matrices__{transitions_strategy}__rev_trans_{include_reverse_transitions}__{i}.png")
                plt.close()


def test_estimate_site_specific_rate_matrices_given_tree_and_site_rates_vectorized():
    tree = _get_test_tree_2(node_prefix="node-")
    msa = _get_test_msa_2(node_prefix="node-")
    mp_msa = _maximum_parsimony(
        tree=tree,
        msa=msa,
    )
    site_rates = [2.0, 0.5]
    alphabet = ["A", "D", "G", "S", "T", "V", "-"]

    quantization_grid_center = 0.03
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 64
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    regularization_rate_matrix = np.zeros(
        shape=(
            len(alphabet),
            len(alphabet)
        )
    )
    for i in range(len(alphabet)):
        regularization_rate_matrix[i, i] = -1
        for j in range(len(alphabet)):
            if j != i:
                regularization_rate_matrix[i, j] = 1/(len(alphabet) - 1)

    for (transitions_strategy, include_reverse_transitions, curr_msa) in [
        ("cherry++", True, msa),
        ("edges", False, mp_msa),
        ("edges", True, mp_msa)
    ]:
        rms = {}
        for use_vectorized_cherryml_implementation in [False, True]:
            rms[use_vectorized_cherryml_implementation] = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
                tree=tree,
                site_rates=site_rates,
                msa=curr_msa,
                alphabet=alphabet,
                # Hyperparameters
                regularization_strength=0.5,
                regularization_rate_matrix=regularization_rate_matrix,
                quantization_points=quantization_points,
                optimization_num_epochs=500,
                transitions_strategy=transitions_strategy,
                include_reverse_transitions=include_reverse_transitions,
                use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
            )["res"]
        errors = np.abs(rms[True] - rms[False])
        print(f"errors = {errors[:2, :5, :5]}")
        print(f"rms[True] = {rms[True][:2, :5, :5]}")
        print(f"rms[False] = {rms[False][:2, :5, :5]}")
        sum_of_errors = np.sum(errors)
        print(f"sum_of_errors = {sum_of_errors}; shape = {errors.shape}; average error = {sum_of_errors / errors.shape[0] / errors.shape[1] / errors.shape[2]}")
        assert(sum_of_errors < 0.1)


def test_estimate_site_specific_rate_matrices_given_tree_and_site_rates_vectorized_all_missing():
    tree = _get_test_tree_2(node_prefix="node-")
    msa = _get_test_msa_all_missing(node_prefix="node-")
    mp_msa = _maximum_parsimony(
        tree=tree,
        msa=msa,
    )
    site_rates = [2.0, 0.5]
    alphabet = ["A", "D", "G", "S", "T", "V"]

    quantization_grid_center = 0.03
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 64
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    regularization_rate_matrix = np.zeros(
        shape=(
            len(alphabet),
            len(alphabet)
        )
    )
    for i in range(len(alphabet)):
        regularization_rate_matrix[i, i] = -1
        for j in range(len(alphabet)):
            if j != i:
                regularization_rate_matrix[i, j] = 1/(len(alphabet) - 1)

    for (transitions_strategy, include_reverse_transitions, curr_msa) in [
        ("cherry++", True, msa),
        ("edges", False, mp_msa),
        ("edges", True, mp_msa)
    ]:
        rms = {}
        for use_vectorized_cherryml_implementation in [False, True]:
            rms[use_vectorized_cherryml_implementation] = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
                tree=tree,
                site_rates=site_rates,
                msa=curr_msa,
                alphabet=alphabet,
                # Hyperparameters
                regularization_strength=0.5,
                regularization_rate_matrix=regularization_rate_matrix,
                quantization_points=quantization_points,
                optimization_num_epochs=500,
                transitions_strategy=transitions_strategy,
                include_reverse_transitions=include_reverse_transitions,
                use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
            )["res"]
        errors = np.abs(rms[True] - rms[False])
        print(f"errors = {errors[:2, :5, :5]}")
        print(f"rms[True] = {rms[True][:2, :5, :5]}")
        print(f"rms[False] = {rms[False][:2, :5, :5]}")
        sum_of_errors = np.sum(errors)
        print(f"sum_of_errors = {sum_of_errors}; shape = {errors.shape}; average error = {sum_of_errors / errors.shape[0] / errors.shape[1] / errors.shape[2]}")
        assert(sum_of_errors < 0.1)


def test_estimate_site_specific_rate_matrices_given_tree_and_site_rates_vectorized_some_all_missing():
    tree = _get_test_tree_2(node_prefix="node-")
    msa = _get_test_msa_some_all_missing(node_prefix="node-")
    mp_msa = _maximum_parsimony(
        tree=tree,
        msa=msa,
    )
    site_rates = [2.0, 0.5]
    alphabet = ["A", "D", "G", "S", "T", "V"]

    quantization_grid_center = 0.03
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 64
    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    regularization_rate_matrix = np.zeros(
        shape=(
            len(alphabet),
            len(alphabet)
        )
    )
    for i in range(len(alphabet)):
        regularization_rate_matrix[i, i] = -1
        for j in range(len(alphabet)):
            if j != i:
                regularization_rate_matrix[i, j] = 1/(len(alphabet) - 1)

    for (transitions_strategy, include_reverse_transitions, curr_msa) in [
        ("cherry++", True, msa),
        ("edges", False, mp_msa),
        ("edges", True, mp_msa)
    ]:
        rms = {}
        for use_vectorized_cherryml_implementation in [False, True]:
            rms[use_vectorized_cherryml_implementation] = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
                tree=tree,
                site_rates=site_rates,
                msa=curr_msa,
                alphabet=alphabet,
                # Hyperparameters
                regularization_strength=0.5,
                regularization_rate_matrix=regularization_rate_matrix,
                quantization_points=quantization_points,
                optimization_num_epochs=500,
                transitions_strategy=transitions_strategy,
                include_reverse_transitions=include_reverse_transitions,
                use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
            )["res"]
        errors = np.abs(rms[True] - rms[False])
        print(f"errors = {errors[:2, :5, :5]}")
        print(f"rms[True] = {rms[True][:2, :5, :5]}")
        print(f"rms[False] = {rms[False][:2, :5, :5]}")
        sum_of_errors = np.sum(errors)
        print(f"sum_of_errors = {sum_of_errors}; shape = {errors.shape}; average error = {sum_of_errors / errors.shape[0] / errors.shape[1] / errors.shape[2]}")
        assert(sum_of_errors < 0.1)


def _train_site_specific_rate_matrix_model_per_family__cached__map_func(args: List[str]):
    assert(len(args) == 19)
    msa_dir = args[0]
    families = args[1]
    regularization_rate_matrix_path = args[2]
    site_rates_dir = args[3]
    tree_dir = args[4]
    regularization_strength = args[5]
    quantization_grid_center = args[6]
    quantization_grid_step = args[7]
    quantization_grid_num_steps = args[8]
    optimization_num_epochs = args[9]
    transitions_strategy = args[10]
    include_reverse_transitions = args[11]
    rate_matrix_parameterization = args[12]
    alphabet = args[13]
    output_model_dir = args[14]
    plot_site_specific_rate_matrices = args[15]
    use_vectorized_cherryml_implementation = args[16]
    vectorized_cherryml_implementation_device = args[17]
    vectorized_cherryml_implementation_num_cores = args[18]

    logger = logging.getLogger(__name__)

    quantization_points = [
        quantization_grid_center * quantization_grid_step**i
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    regularization_rate_matrix = cherryml_io.read_rate_matrix(
        regularization_rate_matrix_path,
    )

    # Write the alphabet first
    cherryml_io.write_pickle(
        alphabet,
        os.path.join(
            output_model_dir,
            "alphabet.txt",
        )
    )

    for family in families:
        logger.info(
            f"Learning Site Specific Rate Matrices for family: {family}\n"
            f"tree_dir = {tree_dir}\n"
            f"site_rates_dir = {site_rates_dir}\n"
            f"msa_dir = {msa_dir}"
        )
        # Learn model for this family
        tree = cherryml_io.read_tree(
            os.path.join(
                tree_dir,
                family + ".txt"
            )
        )
        site_rates = cherryml_io.read_site_rates(
            os.path.join(
                site_rates_dir,
                family + ".txt"
            )
        )
        msa = cherryml_io.read_msa(
            os.path.join(
                msa_dir,
                family + ".txt"
            )
        )
        st = time.time()
        site_specific_rate_matrices = _estimate_site_specific_rate_matrices_given_tree_and_site_rates(
            # Key arguments
            tree=tree,
            site_rates=site_rates,  # Only used for the pseudocounts (i.e. if lambda = 0 then this has no effect on the output)
            msa=msa,
            alphabet=alphabet,
            # Hyperparameters
            regularization_strength=regularization_strength,  # lambda
            regularization_rate_matrix=regularization_rate_matrix.to_numpy(),  # Q_0
            quantization_points=quantization_points,
            optimization_num_epochs=optimization_num_epochs,
            transitions_strategy=transitions_strategy,
            include_reverse_transitions=include_reverse_transitions,
            rate_matrix_parameterization=rate_matrix_parameterization,
            # Logging
            log_dir=os.path.join(output_model_dir, family),
            plot_site_specific_rate_matrices=plot_site_specific_rate_matrices,
            use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
            vectorized_cherryml_implementation_device=vectorized_cherryml_implementation_device,
            vectorized_cherryml_implementation_num_cores=vectorized_cherryml_implementation_num_cores,
        )["res"]
        profiling_str = f"Total time: {time.time() - st}\n"
        profiling_file_path = os.path.join(
            output_model_dir,
            family + ".profiling"
        )
        with open(profiling_file_path, 'w') as f:
            f.write(profiling_str)
        output_file_path = os.path.join(
            output_model_dir,
            family + ".txt"
        )
        with open(output_file_path, 'wb') as f:
            np.save(f, site_specific_rate_matrices)
        caching.secure_parallel_output(
            output_dir=output_model_dir, parallel_arg=family
        )


@caching.cached_parallel_computation(
    output_dirs=["output_model_dir"],
    parallel_arg="families",
    exclude_args=["num_processes", "plot_site_specific_rate_matrices"],
    exclude_args_if_default=["use_vectorized_cherryml_implementation", "vectorized_cherryml_implementation_device", "vectorized_cherryml_implementation_num_cores"],
)
def train_site_specific_rate_matrix_model__cached(
    msa_dir: str,
    families: List[str],
    regularization_rate_matrix_path: str,
    site_rates_dir: str,
    tree_dir: str,
    # num_rate_categories: int = 4,  # Replaced by `site_rates_dir` and `tree_dir` above to avoid coupling to FastTree
    # fast_tree_rate_matrix_path: str = "data/rate_matrices/wag.txt",  # Replaced by `site_rates_dir` and `tree_dir` above to avoid coupling to FastTree
    regularization_strength: float = 0.5,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    optimization_num_epochs: int = 500,  # TODO: Use less?
    transitions_strategy: str = "edges",
    include_reverse_transitions: bool = True,
    rate_matrix_parameterization: str = "pande_reversible",
    alphabet: List[str] = list(utils.amino_acids) + [GAP_CHARACTER],
    num_processes: int = 1,
    plot_site_specific_rate_matrices: int = 0,
    output_model_dir: Optional[str] = None,
    use_vectorized_cherryml_implementation: bool = False,
    vectorized_cherryml_implementation_device: str = "cpu",
    vectorized_cherryml_implementation_num_cores: int = 1,
    _version: str = "2024_04_26_v1",
):
    """
    Train the site-specific rate matrix model on the given transitions.

    Args:
        training_data_dirs: Directories with the training data.
        train_site_rates_dir: Training site rates.
        families: Families to use for training.

    Returns:
        path where the trained model is stored.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Going to train_site_specific_rate_matrix_model__cached on "
        f"{len(families)} families with {num_processes} processes."
    )
    map_args = [
        [
            msa_dir,
            utils.get_process_args(process_rank, num_processes, families),
            regularization_rate_matrix_path,
            site_rates_dir,
            tree_dir,
            regularization_strength,
            quantization_grid_center,
            quantization_grid_step,
            quantization_grid_num_steps,
            optimization_num_epochs,
            transitions_strategy,
            include_reverse_transitions,
            rate_matrix_parameterization,
            alphabet,
            output_model_dir,
            plot_site_specific_rate_matrices,
            use_vectorized_cherryml_implementation,
            vectorized_cherryml_implementation_device,
            vectorized_cherryml_implementation_num_cores,
        ]
        for process_rank in range(num_processes)
    ]

    map_func = _train_site_specific_rate_matrix_model_per_family__cached__map_func
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(
                        map_func,
                        map_args,
                    ),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(
                    map_func,
                    map_args,
                ),
                total=len(map_args),
            )
        )


def _evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached__map_func(
    args: List,
):
    """
    Auxiliary version of
    "evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached"
    used for multiprocessing.
    """
    assert len(args) == 6
    transitions_dir = args[0]
    families = args[1]
    model_dir = args[2]
    output_transitions_log_likelihood_dir = args[3]
    output_transitions_log_likelihood_per_site_dir = args[4]
    condition_on_non_gap = args[5]
    alphabet = cherryml_io.read_pickle(
        os.path.join(model_dir, "alphabet.txt")
    )
    for family in families:
        transitions = cherryml_io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        site_specific_rate_matrices = np.load(os.path.join(model_dir, f"{family}.txt"))

        ##### Now do the per-site LLs
        st = time.time()
        transitions_log_likelihood_per_site = (
            evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
                transitions=transitions,
                site_specific_rate_matrices=site_specific_rate_matrices,
                alphabet=alphabet,
                condition_on_non_gap=condition_on_non_gap,
            )
        )
        cherryml_io.write_transitions_log_likelihood_per_site(
            transitions_log_likelihood_per_site=transitions_log_likelihood_per_site,
            transitions_log_likelihood_per_site_path=os.path.join(
                output_transitions_log_likelihood_per_site_dir, family + ".txt"
            ),
        )
        caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_per_site_dir,
            parallel_arg=family,
        )
        ##### Now do total LLs
        transitions_log_likelihood = [
            sum(x) for x in transitions_log_likelihood_per_site
        ]
        cherryml_io.write_transitions_log_likelihood(
            transitions_log_likelihood=transitions_log_likelihood,
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )

        profiling_str = f"Total time: {time.time() - st}\n"
        for profiling_path in [  # (We write it in both places)
            os.path.join(
                output_transitions_log_likelihood_per_site_dir,
                family + ".profiling"
            ),
            os.path.join(
                output_transitions_log_likelihood_dir,
                family + ".profiling"
            )
        ]:
            with open(profiling_path, "w") as f:
                f.write(profiling_str)


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args=["num_processes"],
    exclude_args_if_default=["condition_on_non_gap"],
    write_extra_log_files=True,
)
def evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    families: List[str],
    model_dir: str,
    condition_on_non_gap: bool = False,
    num_processes: int = 1,
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2024_04_26_v1",
) -> None:
    """
    Compute transitions log-likelihood under the site-specific rate matrix model.

    Rate matrices must be stored in {model_dir}/{family}.txt

    Args:
        transitions_dir: The directory with the transitions for which to
            compute the log-likelihood. The transitions for family 'family'
            should be in the file '{family}.txt'
        families: List of families for which to compute the log-likelihood.
        model_dir: The directory containing the rate matrices in the files
            '{family}.txt'
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
        num_processes: How many processes to use to paralellize the likelihood
            evaluation. The parallelization is family-based.
        output_transitions_log_likelihood_dir: Where the log-likelihoods will
            get written. The log-likelihoods for family 'family' will be in
            the file '{family}.txt', with one line per transition.
        output_transitions_log_likelihood_per_site_dir: Where the per-site
            log-likelihoods will get written. The log-likelihoods for family
            'family' will be in the file '{family}.txt', with one line per
            transition.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Going to evaluate site-specific rate matrix model on "
        f"{len(families)} families using {num_processes} processes"
    )

    map_args = [
        [
            transitions_dir,
            utils.get_process_args(process_rank, num_processes, families),
            model_dir,
            output_transitions_log_likelihood_dir,
            output_transitions_log_likelihood_per_site_dir,
            condition_on_non_gap,
        ]
        for process_rank in range(num_processes)
    ]

    map_func = _evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached__map_func
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(
                        map_func,
                        map_args,
                    ),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(
                    map_func,
                    map_args,
                ),
                total=len(map_args),
            )
        )


@pytest.mark.slow
def test_train_site_specific_rate_matrix_model__cached():
    """
    Tests that the VEP scores make sense on a hand-crafted example.
    """
    for use_vectorized_cherryml_implementation in [False, True]:
        families = [f"family_{i}" for i in range(4)]
        alphabet = ["A", "D", "G", "S", "T", "V", "-"]
        with tempfile.TemporaryDirectory() as msa_dir:
            with tempfile.TemporaryDirectory() as tree_dir:
                with tempfile.TemporaryDirectory() as site_rates_dir:
                    with tempfile.NamedTemporaryFile("w") as regularization_rate_matrix_file:
                        with tempfile.TemporaryDirectory() as output_model_dir:
                            regularization_rate_matrix_path = regularization_rate_matrix_file.name
                            for i, family in enumerate(families):
                                tree = _get_test_tree_2(node_prefix="node-", equal_edge_lengths=((i % 2) == 0))
                                msa = _get_test_msa_2(node_prefix="node-")
                                msa = _maximum_parsimony(tree=tree, msa=msa)
                                cherryml_io.write_msa(
                                    msa,
                                    os.path.join(msa_dir, f"{family}.txt")
                                )
                                cherryml_io.write_tree(
                                    tree,
                                    os.path.join(tree_dir, f"{family}.txt")
                                )
                                cherryml_io.write_site_rates(
                                    [2.0, 0.5],
                                    os.path.join(site_rates_dir, f"{family}.txt"),
                                )
                            regularization_rate_matrix = np.zeros(
                                shape=(
                                    len(alphabet),
                                    len(alphabet)
                                )
                            )
                            for i in range(len(alphabet)):
                                regularization_rate_matrix[i, i] = -1
                                for j in range(len(alphabet)):
                                    if j != i:
                                        regularization_rate_matrix[i, j] = 1/(len(alphabet) - 1)
                            cherryml_io.write_rate_matrix(
                                regularization_rate_matrix,
                                alphabet,
                                regularization_rate_matrix_path,
                            )
                            train_site_specific_rate_matrix_model__cached(
                                msa_dir=msa_dir,
                                families=families,
                                regularization_rate_matrix_path=regularization_rate_matrix_path,
                                site_rates_dir=site_rates_dir,
                                tree_dir=tree_dir,
                                regularization_strength=0.5,
                                quantization_grid_center=0.03,
                                quantization_grid_step=1.1,
                                quantization_grid_num_steps=64,
                                optimization_num_epochs=100,
                                transitions_strategy="edges",
                                include_reverse_transitions=True,
                                rate_matrix_parameterization="pande_reversible",
                                alphabet=alphabet,
                                num_processes=4,
                                output_model_dir=output_model_dir,
                                use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
                            )
                            with tempfile.TemporaryDirectory() as test_transitions_dir:
                                for family in families:
                                    expected_orderings = _get_test_expected_orderings(t=1.0)
                                    transitions = sum(expected_orderings, [])
                                    cherryml_io.write_transitions(
                                        transitions,
                                        os.path.join(test_transitions_dir, family + ".txt"),
                                    )
                                with tempfile.TemporaryDirectory() as output_transitions_log_likelihood_dir:
                                    with tempfile.TemporaryDirectory() as output_transitions_log_likelihood_per_site_dir:
                                        evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached(
                                            transitions_dir=test_transitions_dir,
                                            families=families,
                                            model_dir=output_model_dir,
                                            output_transitions_log_likelihood_dir=output_transitions_log_likelihood_dir,
                                            output_transitions_log_likelihood_per_site_dir=output_transitions_log_likelihood_per_site_dir,
                                            num_processes=4,
                                        )
                                        for family_num, family in enumerate(families):
                                            print(f"family = {family}")
                                            tlls = cherryml_io.read_transitions_log_likelihood(
                                                os.path.join(
                                                    output_transitions_log_likelihood_dir,
                                                    family + ".txt"
                                                )
                                            )
                                            tlls_per_site = cherryml_io.read_transitions_log_likelihood_per_site(
                                                os.path.join(
                                                    output_transitions_log_likelihood_per_site_dir,
                                                    family + ".txt"
                                                )
                                            )
                                            assert(len(tlls) == len(transitions))
                                            assert(len(tlls_per_site) == len(transitions))
                                            assert(len(transitions) % 2 == 0)
                                            assert(len(transitions) == 2 * len(expected_orderings))
                                            for i in range(len(expected_orderings)):
                                                transition_1, transition_2 = transitions[2 * i], transitions[2 * i + 1]
                                                assert(abs(sum(tlls_per_site[2 * i]) - tlls[2 * i]) < 1e-6)
                                                assert(abs(sum(tlls_per_site[2 * i + 1]) - tlls[2 * i + 1]) < 1e-6)
                                                if tlls[2 * i] > tlls[2 * i + 1] + 1e-4:
                                                    print(f"PASSED: {(transition_1, tlls_per_site[2 * i])} > {(transition_2,  tlls_per_site[2 * i + 1])}")
                                                else:
                                                    print(f"FAILED: {(transition_1, tlls_per_site[2 * i])} > {(transition_2,  tlls_per_site[2 * i + 1])}")
                                                    if family_num % 2 == 0:
                                                        raise ValueError(
                                                            "This check should have passed!"
                                                        )


@pytest.mark.slow
def test_vectorized_cherryml_implementation_GOAT():
    """
    Best test to check correctness of vectorized implementation:
    Compares the estimates of the vectorized implementation against the original one on real data.
    """
    families = ["1a0b_1_A", "1a2o_1_A", "1a4m_1_A"]
    alphabet = list(utils.amino_acids) + [GAP_CHARACTER]
    msa_dir = "tests/siterm_tests/example_data/train_msas"
    site_rates_dir = "tests/siterm_tests/example_data/train_site_rates_4cat"
    tree_dir = "tests/siterm_tests/example_data/train_trees_fast_tree_wag_4rc"
    regularization_rate_matrix_path = "data/rate_matrices/wag_21_x_21.txt"
    with tempfile.TemporaryDirectory() as mp_msa_dir:
        for family in families:
            tree = cherryml_io.read_tree(os.path.join(tree_dir, family + ".txt"))
            msa = cherryml_io.read_msa(os.path.join(msa_dir, family + ".txt"))
            # mp_msa = _maximum_parsimony(tree=tree, msa=msa)
            mp_msa = msa  # Don't do MP
            cherryml_io.write_msa(
                mp_msa,
                os.path.join(mp_msa_dir, family + ".txt")
            )
        rms = {}
        times = {}
        for use_vectorized_cherryml_implementation in [False, True]:
            with tempfile.TemporaryDirectory() as output_model_dir:
                # output_model_dir = "output_model_dir"
                st = time.time()
                train_site_specific_rate_matrix_model__cached(
                    msa_dir=mp_msa_dir,
                    families=families,
                    regularization_rate_matrix_path=regularization_rate_matrix_path,
                    site_rates_dir=site_rates_dir,
                    tree_dir=tree_dir,
                    regularization_strength=0.5,
                    quantization_grid_center=0.03,
                    quantization_grid_step=1.1 ** 8,
                    quantization_grid_num_steps=8,
                    optimization_num_epochs=100,
                    transitions_strategy="cherry++",
                    include_reverse_transitions=True,
                    rate_matrix_parameterization="pande_reversible",
                    alphabet=alphabet,
                    num_processes=3,
                    output_model_dir=output_model_dir,
                    plot_site_specific_rate_matrices=0,
                    use_vectorized_cherryml_implementation=use_vectorized_cherryml_implementation,
                )
                rms[use_vectorized_cherryml_implementation] = [
                    np.load(os.path.join(output_model_dir, f"{family}.txt"))
                    for family in families
                ]
                times[use_vectorized_cherryml_implementation] = time.time() - st
        errors = [
            rms[False][i] - rms[True][i]
            for i in range(3)
        ]
        for i in range(3):
            print(f"errors = {errors[i][:2, :5, :5]}")
            print(f"rms[False][i] = {rms[False][i][:2, :5, :5]}")
            print(f"rms[True][i] = {rms[True][i][:2, :5, :5]}")
        sum_of_errors = sum([np.mean(np.abs(errors[i])) for i in range(3)])
        print(f"sum_of_errors = {sum_of_errors}")
        print(f"times = {times}")
        assert(sum_of_errors < 0.01)


def get_synthetic_counts_amino_acids(L, B):
    assert(B % 2 == 1)
    Qs_true = np.zeros(shape=(L, 20, 20))
    for l in range(L):
        if l % 2 == 0:
            Qs_true[l, :, :] = cherryml_io.read_rate_matrix(markov_chain.get_equ_path())
        else:
            Qs_true[l, :, :] = cherryml_io.read_rate_matrix(markov_chain.get_wag_path())

    def _matrix_exponential_reversible(
        rate_matrix: np.array,
        exponents: float,
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
            device="cpu",
        )

    counts = np.zeros(shape=(L, B, 20, 20))
    num_steps = int(B/2)
    times = [1.1 ** i for i in range(-num_steps, num_steps+1)]

    for l in range(L):
        counts[l, :, :] = _matrix_exponential_reversible(
            Qs_true[l, :, :],
            times
        )
    times = np.array([[1.1 ** i for i in range(-num_steps, num_steps+1)] for _ in range(L)])

    return Qs_true, counts, times


@pytest.mark.slow
def test_on_synthetic_amino_acid_counts():
    # First test without initialization
    Qs_true, counts, times = get_synthetic_counts_amino_acids(L=10, B=9)
    res_dict = quantized_transitions_mle_vectorized_over_sites(counts, times, num_epochs=100, initialization=None)
    Qs_learnt = res_dict["res"]
    error = np.mean((Qs_true - Qs_learnt) ** 2)
    print(f"mean error = {error}")
    assert(error < 1e-3)

    # Now with GT initialization
    res_dict = quantized_transitions_mle_vectorized_over_sites(counts, times, num_epochs=0, initialization=Qs_true)
    Qs_learnt = res_dict["res"]
    error = np.mean((Qs_true - Qs_learnt) ** 2)
    print(f"mean error = {error}")
    assert(error < 1e-6)


def get_synthetic_counts_DNA(L, B):
    assert(B % 2 == 1)
    Qs_true = np.zeros(shape=(L, 4, 4))

    num_steps = int(B/2)
    times_list = []

    for l in range(L):
        if l % 2 == 0:
            Qs_true[l, :, :] = np.array(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ]
            )
        else:
            Qs_true[l, :, :] = np.array(
                [
                    [-1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, -1.0, 1.0/3.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, -1.0, 1.0/3.0],
                    [1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0],
                ]
            ) * 3.0
        if l % 4 in [0, 1]:
            times_list.append([1.1 ** i for i in range(-num_steps, num_steps+1)])
        else:
            times_list.append([(1.1 ** i)/10.0 for i in range(-num_steps, num_steps+1)])

    def _matrix_exponential_reversible(
        rate_matrix: np.array,
        exponents: float,
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
            device="cpu",
        )

    counts = np.zeros(shape=(L, B, 4, 4))

    for l in range(L):
        counts[l, :, :] = _matrix_exponential_reversible(
            Qs_true[l, :, :],
            times_list[l]
        )

    return Qs_true, counts, times_list


def test_on_synthetic_DNA_counts():
    # First test without initialization
    Qs_true, counts, times = get_synthetic_counts_DNA(L=128, B=11)
    res_dict = quantized_transitions_mle_vectorized_over_sites(counts, times, num_epochs=100, initialization=None)
    Qs_learnt = res_dict["res"]
    error = np.mean((Qs_true - Qs_learnt) ** 2)
    print(f"mean error = {error}")
    assert(error < 1e-3)

    # Now with GT initialization
    res_dict = quantized_transitions_mle_vectorized_over_sites(counts, times, num_epochs=0, initialization=Qs_true)
    Qs_learnt = res_dict["res"]
    error = np.mean((Qs_true - Qs_learnt) ** 2)
    print(f"mean error = {error}")
    assert(error < 1e-6)
