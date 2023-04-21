import itertools
import os
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from parameterized import parameterized

import cherryml
from cherryml.evaluation import compute_log_likelihoods
from cherryml.io import (
    Tree,
    read_log_likelihood,
    read_msa,
    read_site_rates,
    read_tree,
    write_contact_map,
    write_msa,
    write_probability_distribution,
    write_rate_matrix,
    write_site_rates,
    write_tree,
)
from cherryml.markov_chain import (
    FactorizedReversibleModel,
    chain_product,
    compute_stationary_distribution,
    equ_matrix,
    matrix_exponential,
    wag_matrix,
    wag_stationary_distribution,
)
from tests.utils import create_synthetic_contact_map

DATA_DIR = "./tests/evaluation_tests/test_input_data"


def create_fake_msa_and_contact_map_and_site_rates(
    tree: Tree,
    amino_acids: List[str],
    random_seed: int,
    num_rate_categories: int,
) -> Tuple[Dict[str, str], np.array, List[float]]:
    """
    Create fake data for a tree.
    """
    np.random.seed(random_seed)

    num_leaves = sum([tree.is_leaf(v) for v in tree.nodes()])
    single_site_patterns = [
        "".join(pattern)
        for pattern in list(itertools.product(amino_acids, repeat=num_leaves))
    ]
    pair_of_site_patterns = list(
        itertools.product(single_site_patterns, repeat=2)
    )
    # print(f"single_site_patterns = {single_site_patterns}")
    # print(f"pair_of_site_patterns = {pair_of_site_patterns}")
    num_sites = len(single_site_patterns) + 2 * len(pair_of_site_patterns)
    contact_map = create_synthetic_contact_map(
        num_sites=num_sites,
        num_sites_in_contact=2 * len(pair_of_site_patterns),
        random_seed=random_seed,
    )
    contacting_pairs = list(zip(*np.where(contact_map == 1)))
    np.random.shuffle(contacting_pairs)
    contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
    contacting_sites = list(sum(contacting_pairs, ()))
    independent_sites = [
        i for i in range(num_sites) if i not in contacting_sites
    ]
    np.random.shuffle(independent_sites)
    # print(f"contacting_pairs = {contacting_pairs}")
    # print(f"independent_sites = {independent_sites}")

    msa_array = np.zeros(shape=(num_sites, num_leaves), dtype=str)
    for i, site_idx in enumerate(independent_sites):
        for leaf_idx in range(num_leaves):
            msa_array[site_idx, leaf_idx] = single_site_patterns[i][leaf_idx]
    for i, (site_idx_1, site_idx_2) in enumerate(contacting_pairs):
        for leaf_idx in range(num_leaves):
            msa_array[site_idx_1, leaf_idx] = pair_of_site_patterns[i][0][
                leaf_idx
            ]
            msa_array[site_idx_2, leaf_idx] = pair_of_site_patterns[i][1][
                leaf_idx
            ]
    # print(f"msa_array = {msa_array}")
    # for i in range(num_sites):
    #     print(i, msa_array[i])
    msa = {
        leaf: "".join(msa_array[:, i]) for i, leaf in enumerate(tree.leaves())
    }
    # print(f"msa = {msa}")
    site_rates = [0.5 * np.log(2 + i) for i in range(num_rate_categories)] * (
        int(num_sites / num_rate_categories) + 1
    )
    site_rates = site_rates[:num_sites]
    np.random.shuffle(site_rates)
    # print(f"site_rates = {site_rates}")
    return msa, contact_map, site_rates


def likelihood_computation_wrapper(
    tree: Tree,
    msa: Dict[str, str],
    contact_map: np.array,
    site_rates: List[float],
    amino_acids: List[str],
    pi_1: np.array,
    Q_1: np.array,
    reversible_1: bool,
    device_1: str,
    pi_2: np.array,
    Q_2: np.array,
    reversible_2: bool,
    device_2: Optional[str],
    method: str,
    num_processes: int = 1,
    num_families: int = 1,
) -> Tuple[float, List[float]]:
    """
    Compute data loglikelihood by one of several methods

    If device_2 is None, then all the coevolution parameters will be replaced
    by None in the call to compute_log_likelihoods.
    """
    if method == "python" or method == "C++":
        families = [f"fam-{i}" for i in range(num_families)]
        cpp = method == "C++"
        with tempfile.TemporaryDirectory() as tree_dir:
            for family in families:
                tree_path = os.path.join(tree_dir, family + ".txt")
                write_tree(tree, tree_path)
            with tempfile.TemporaryDirectory() as msa_dir:
                for family in families:
                    msa_path = os.path.join(msa_dir, family + ".txt")
                    write_msa(msa, msa_path)
                with tempfile.TemporaryDirectory() as contact_map_dir:
                    for family in families:
                        contact_map_path = os.path.join(
                            contact_map_dir, family + ".txt"
                        )
                        write_contact_map(contact_map, contact_map_path)
                    with tempfile.TemporaryDirectory() as site_rates_dir:
                        for family in families:
                            site_rates_path = os.path.join(
                                site_rates_dir, family + ".txt"
                            )
                            write_site_rates(site_rates, site_rates_path)
                        with tempfile.NamedTemporaryFile("w") as pi_1_file:
                            pi_1_path = pi_1_file.name
                            # pi_1_path = "./pi_1_path.txt"
                            write_probability_distribution(
                                pi_1, amino_acids, pi_1_path
                            )
                            with tempfile.NamedTemporaryFile("w") as Q_1_file:
                                Q_1_path = Q_1_file.name
                                # Q_1_path = "./Q_1_path.txt"
                                write_rate_matrix(Q_1, amino_acids, Q_1_path)
                                with tempfile.NamedTemporaryFile(
                                    "w"
                                ) as pi_2_file:
                                    pi_2_path = pi_2_file.name
                                    # pi_2_path = "./pi_2_path.txt"
                                    amino_acid_pairs = [
                                        aa1 + aa2
                                        for aa1 in amino_acids
                                        for aa2 in amino_acids
                                    ]
                                    write_probability_distribution(
                                        pi_2, amino_acid_pairs, pi_2_path
                                    )
                                    with tempfile.NamedTemporaryFile(
                                        "w"
                                    ) as Q_2_file:
                                        Q_2_path = Q_2_file.name
                                        # Q_2_path = "./Q_2_path.txt"
                                        write_rate_matrix(
                                            Q_2, amino_acid_pairs, Q_2_path
                                        )
                                        need_coevolution = device_2 is not None
                                        with tempfile.TemporaryDirectory() as d:
                                            compute_log_likelihoods(
                                                tree_dir=tree_dir,
                                                msa_dir=msa_dir,
                                                site_rates_dir=site_rates_dir,
                                                contact_map_dir=contact_map_dir
                                                if need_coevolution
                                                else None,
                                                families=families,
                                                amino_acids=amino_acids,
                                                pi_1_path=pi_1_path,
                                                Q_1_path=Q_1_path,
                                                reversible_1=reversible_1,
                                                device_1=device_1,
                                                pi_2_path=pi_2_path
                                                if need_coevolution
                                                else None,
                                                Q_2_path=Q_2_path
                                                if need_coevolution
                                                else None,
                                                reversible_2=reversible_2
                                                if need_coevolution
                                                else None,
                                                device_2=device_2
                                                if need_coevolution
                                                else None,
                                                output_likelihood_dir=d,
                                                num_processes=num_processes,
                                                use_cpp_implementation=cpp,
                                            )
                                            ll_list = []
                                            lls_list = []
                                            for family in families:
                                                log_likelihood_path = (
                                                    os.path.join(
                                                        d,
                                                        family + ".txt",
                                                    )
                                                )
                                                ll, lls = read_log_likelihood(
                                                    log_likelihood_path
                                                )
                                                ll_list.append(ll)
                                                lls_list.append(lls)
                                            if num_families > 1:
                                                return ll_list, lls_list
                                            else:
                                                return ll_list[0], lls_list[0]
    else:
        raise NotImplementedError(f"Unknown method: {method}")


class Test_small_wag_3_seqs(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_3_seqs(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "S",
            "l2": "T",
            "l3": "G",
        }
        contact_map = np.eye(1)
        site_rates = [1.0]
        equ = equ_matrix().to_numpy()
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            reversible_2=reversible,
            device_2=None,  # Passing in None to exercise case of no coevolution
            method="python",
        )
        np.testing.assert_almost_equal(ll, -7.343870, decimal=4)
        np.testing.assert_almost_equal(lls, [-7.343870], decimal=4)


class Test_small_wag_4_seqs_1_site(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_4_seqs_1_site(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2", "l3", "l4"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.121352212),
                ("r", "i1", 1.840784231),
                ("i1", "l3", 1.870540996),
                ("i1", "l4", 2.678783814),
            ]
        )
        msa = {"l1": "S", "l2": "T", "l3": "G", "l4": "D"}
        contact_map = np.eye(1)
        site_rates = [1.0]
        equ = equ_matrix().to_numpy()
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -10.091868, decimal=4)
        np.testing.assert_almost_equal(lls, [-10.091868], decimal=4)


class Test_small_wag_4_seqs_2_sites_and_gaps(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_4_seqs_2_sites_and_gaps(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2", "l3", "l4"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.121562482),
                ("r", "i1", 1.719057732),
                ("i1", "l3", 1.843908633),
                ("i1", "l4", 2.740236263),
            ]
        )
        msa = {"l1": "SS", "l2": "TT", "l3": "GG", "l4": "D-"}
        contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -17.436349, decimal=4)
        np.testing.assert_almost_equal(lls, [-10.092142, -7.344207], decimal=4)


class Test_small_equ_x_equ_3_seqs(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_equ_x_equ_3_seqs(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "SK",
            "l2": "TI",
            "l3": "GL",
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        pi = compute_stationary_distribution(equ)
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=equ,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -9.382765 * 2, decimal=4)
        np.testing.assert_almost_equal(lls, [-9.382765, -9.382765], decimal=4)


class Test_small_equ_x_wag_3_seqs(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_equ_x_wag_3_seqs(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "SK",
            "l2": "TI",
            "l3": "GL",
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        wag = wag_matrix().to_numpy()
        equ_x_wag = chain_product(equ, wag)
        pi_2 = compute_stationary_distribution(equ_x_wag)
        pi = compute_stationary_distribution(equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=equ,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_2,
            Q_2=equ_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        expected_ll = -9.382765 + -9.714873
        np.testing.assert_almost_equal(ll, expected_ll, decimal=4)
        np.testing.assert_almost_equal(
            lls, [expected_ll / 2, expected_ll / 2], decimal=4
        )


class Test_small_wag_x_equ_3_seqs(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_equ_3_seqs(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "KS",
            "l2": "IT",
            "l3": "LG",
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        wag = wag_matrix().to_numpy()
        wag_x_equ = chain_product(wag, equ)
        pi_2 = compute_stationary_distribution(wag_x_equ)
        pi = compute_stationary_distribution(equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_2,
            Q_2=wag_x_equ,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        expected_ll = -9.714873 + -9.382765
        np.testing.assert_almost_equal(ll, expected_ll, decimal=4)
        np.testing.assert_almost_equal(
            lls, [expected_ll / 2, expected_ll / 2], decimal=4
        )


class Test_small_wag_x_wag_3_seqs(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_wag_3_seqs(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "SK",
            "l2": "TI",
            "l3": "GL",
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -7.343870 + -9.714873
        np.testing.assert_almost_equal(ll, ll_expected, decimal=4)
        np.testing.assert_almost_equal(
            lls, [ll_expected / 2, ll_expected / 2], decimal=4
        )


class Test_small_wag_x_wag_3_seqs_many_sites(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_wag_3_seqs_many_sites(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "KSRMFCP",
            "l2": "ITVDQAE",
            "l3": "LGYNGHW",
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        site_rates = [
            1.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
        ]  # I use 2s to make sure site rates are not getting used for coevo
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        lls_expected = [
            -9.714873,
            (-7.343870 + -10.78960) / 2,
            (-10.56782 + -11.85804) / 2,
            (-11.85804 + -10.56782) / 2,
            -11.38148,
            (-10.78960 + -7.343870) / 2,
            -11.31551,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)


class Test_small_wag_x_wag_3_seqs_many_sites_and_gaps(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_wag_3_seqs_many_sites_and_gaps(
        self, reversible, device
    ):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "--RMF--",
            "l2": "-TV-QAE",
            "l3": "-G--G-W",
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]  # I use 2s to make sure site rates are not getting used for coevo
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        lls_expected = [
            0.0,
            (-5.323960 + -2.446133) / 2,
            (-6.953994 + -3.937202) / 2,
            (-3.937202 + -6.953994) / 2,
            -11.38148,
            (-2.446133 + -5.323960) / 2,
            -7.469626,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)


class Test_small_wag_x_wag_2_seqs_many_sites(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_wag_2_seqs_many_sites(self, reversible, device):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2"])
        tree.add_edges(
            [
                ("r", "i1", 0.37),
                ("i1", "l1", 1.1),
                ("i1", "l2", 2.2),
            ]
        )
        msa = {
            "l1": "AGFYLTV",
            "l2": "DPHISKQ",
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
        ]  # I use 2s to make sure site rates are not getting used for coevo
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        lls_expected = [
            -5.301370,
            (-5.790787 + -5.537568) / 2,
            (-6.895662 + -6.497235) / 2,
            (-6.497235 + -6.895662) / 2,
            -5.436122,
            (-5.537568 + -5.790787) / 2,
            -6.212303,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)


class Test_small_wag_x_wag_2_seqs_many_sites_and_gaps(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_small_wag_x_wag_2_seqs_many_sites_and_gaps(
        self, reversible, device
    ):
        """
        This was manually verified with FastTree.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2"])
        tree.add_edges(
            [
                ("r", "i1", 0.37),
                ("i1", "l1", 1.1),
                ("i1", "l2", 2.2),
            ]
        )
        msa = {
            "l1": "----LT-",
            "l2": "-PHI--Q",
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
        ]  # I use 2s to make sure site rates are not getting used for coevo
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        lls_expected = [
            0.0,
            (-3.084277 + -2.796673) / 2,
            (-3.711890 + -3.026892) / 2,
            (-3.026892 + -3.711890) / 2,
            -2.450980,
            (-2.796673 + -3.084277) / 2,
            -3.304213,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)


class Test_real_data_single_site_medium(unittest.TestCase):
    @parameterized.expand(
        [
            ("1_cat", 1, -4649.6146),
            ("2_cat", 2, -4397.8184),
            ("4_cat", 4, -4337.8688),
            ("20_cat", 20, -4307.0638),
        ]
    )
    def test_real_data_single_site_medium(self, name, num_cats, ll_expected):
        """
        Test on family 1a92_1_A using only WAG (no co-evolution model).
        """
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a92_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a92_1_A.txt"))
        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a92_1_A.txt"
            )
        )
        contact_map = np.eye(len(site_rates))

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=True,
            device_1="cpu",
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=True,
            device_2="cpu",
            method="python",
        )
        np.testing.assert_almost_equal(ll, ll_expected, decimal=4)


class Test_real_data_single_site_large(unittest.TestCase):
    @parameterized.expand([("20_cat", 20, -264605.0691)])
    @pytest.mark.slow
    def test_real_data_single_site_large(self, name, num_cats, ll_expected):
        """
        Test on family 13gs_1_A using only WAG (no co-evolution model).
        """
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/13gs_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/13gs_1_A.txt"))
        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/13gs_1_A.txt"
            )
        )
        contact_map = np.eye(len(site_rates))

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=True,
            device_1="cpu",
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=True,
            device_2="cpu",
            method="python",
        )
        np.testing.assert_almost_equal(ll, ll_expected, decimal=2)


class Test_real_data_pair_site_medium(unittest.TestCase):
    @parameterized.expand(
        [
            ("1_cat", 1, -4649.6146),
            ("2_cat", 2, -4397.8184),
            ("4_cat", 4, -4337.8688),
            ("20_cat", 20, -4307.0638),
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_medium(self, name, num_cats, ll_expected):
        """
        Test on family 1a92_1_A using WAG x WAG co-evolution model!
        """
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a92_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a92_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a92_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=True,
            device_1="cpu",
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=True,
            device_2="cpu",
            method="python",
        )
        np.testing.assert_almost_equal(ll, ll_expected, decimal=4)


class Test_real_data_pair_site_large(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_large(self, name, reversible, device):
        """
        Test on family 13gs_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/13gs_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/13gs_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/13gs_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -264605.0691
        np.testing.assert_almost_equal(ll, ll_expected, decimal=2)


class Test_real_data_pair_site_huge(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_huge(self, name, reversible, device):
        """
        Test on family 1a8h_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a8h_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a8h_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a8h_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -561001.0635
        np.testing.assert_almost_equal(ll, ll_expected, decimal=2)


class Test_numerical_stability(unittest.TestCase):
    @parameterized.expand(
        [(False, "cpu"), (False, "cuda"), (True, "cpu"), (True, "cuda")]
    )
    def test_numerical_stability(self, reversible, device):
        """
        Th matrix exponential on very small branches produces near-0 entries
        in the transition matrix, which can cause issues.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            "l1": "AA",
            "l2": "AA",
            "l3": "AA",
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = np.array(
            [
                [-1.0, 1.0],
                [2.0, -2.0],
            ]
        )
        pi = compute_stationary_distribution(equ)
        fact_1 = FactorizedReversibleModel(equ)
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        fact_2 = FactorizedReversibleModel(equ_x_equ)
        np.testing.assert_almost_equal(
            matrix_exponential(
                [1.0],
                equ_x_equ,
                fact=fact_2,
                reversible=reversible,
                device="cpu",
            )[0, 0, 0],
            matrix_exponential(
                [1.0], equ, fact=fact_1, reversible=reversible, device="cpu"
            )[0, 0, 0]
            ** 2,
        )
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=["A", "S"],
            pi_1=pi,
            Q_1=equ,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -1.199186 * 2, decimal=4)
        np.testing.assert_almost_equal(lls, [-1.199186, -1.199186], decimal=4)


class Test_real_data_single_site_medium_multiprocess(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_single_site_medium_multiprocess(
        self, name, reversible, device
    ):
        """
        Test on family 1a92_1_A using only WAG (no co-evolution model).
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a92_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a92_1_A.txt"))
        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a92_1_A.txt"
            )
        )
        contact_map = np.eye(len(site_rates))

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll_list, lls_list = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
            num_processes=3,
            num_families=3,
        )
        for ll in ll_list:
            ll_expected = -4307.0638
            np.testing.assert_almost_equal(ll, ll_expected, decimal=4)


class Test_real_data_pair_site_medium_multiprocess(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_medium_multiprocess(
        self, name, reversible, device
    ):
        """
        Test on family 1a92_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return

        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a92_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a92_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a92_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll_list, lls_list = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
            num_processes=3,
            num_families=3,
        )
        for ll in ll_list:
            ll_expected = -4307.0638
            np.testing.assert_almost_equal(ll, ll_expected, decimal=4)


class Test_real_data_pair_site_large_multiprocess(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_large_multiprocess(
        self, name, reversible, device
    ):
        """
        Test on family 13gs_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/13gs_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/13gs_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/13gs_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll_list, lls_list = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
            num_processes=3,
            num_families=3,
        )
        for ll in ll_list:
            ll_expected = -264605.0691
            np.testing.assert_almost_equal(ll, ll_expected, decimal=2)


class Test_real_data_pair_site_huge_multiprocess(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_huge_multiprocess(
        self, name, reversible, device
    ):
        """
        Test on family 1a8h_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 20
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a8h_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a8h_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a8h_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll_list, lls_list = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
            num_processes=3,
            num_families=3,
        )
        for ll in ll_list:
            ll_expected = -561001.0635
            np.testing.assert_almost_equal(ll, ll_expected, decimal=2)


class Test_real_data_pair_site_huge_1_cat(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_huge_1_cat(self, name, reversible, device):
        """
        Test on family 1a8h_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 1
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a8h_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a8h_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a8h_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -595404.1770
        np.testing.assert_almost_equal(ll, ll_expected, decimal=1)


class Test_real_data_pair_site_huge_2_cat(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_huge_2_cat(self, name, reversible, device):
        """
        Test on family 1a8h_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 2
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a8h_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a8h_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a8h_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -573696.2140
        np.testing.assert_almost_equal(ll, ll_expected, decimal=1)


class Test_real_data_pair_site_huge_4_cat(unittest.TestCase):
    @parameterized.expand(
        [
            ("rev_cuda", True, "cuda"),
            ("rev_cpu", True, "cpu"),
            ("irrev_cuda", False, "cuda"),
            # ("irrev_cpu", False, "cpu"),  # OOM
        ]
    )
    @pytest.mark.slow
    def test_real_data_pair_site_huge_4_cat(self, name, reversible, device):
        """
        Test on family 1a8h_1_A using WAG x WAG co-evolution model!
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        num_cats = 4
        tree = read_tree(
            os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a8h_1_A.txt")
        )
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a8h_1_A.txt"))

        site_rates = read_site_rates(
            os.path.join(
                DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a8h_1_A.txt"
            )
        )
        median_site_rate = np.median(site_rates)
        places_with_median_site_rate = [
            i
            for (i, site_rate) in enumerate(site_rates)
            if site_rate == median_site_rate
        ]
        np.random.seed(1)
        np.random.shuffle(places_with_median_site_rate)

        # Let's make half of these sites evolve coupled
        contact_map = np.eye(len(site_rates))
        for i in range(len(places_with_median_site_rate) // 4):
            j = places_with_median_site_rate[2 * i]
            k = places_with_median_site_rate[2 * i + 1]
            contact_map[j, k] = 1
            contact_map[k, j] = 1

        # Now rescale the rates and the tree, since co-evolution uses a
        # universal rate of 1
        tree_scaled = Tree()
        tree_scaled.add_nodes(tree.nodes())
        for u, v, length in tree.edges():
            tree_scaled.add_edge(u, v, length * median_site_rate)
        site_rates_scaled = [
            site_rate / median_site_rate for site_rate in site_rates
        ]

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree_scaled,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates_scaled,
            amino_acids=cherryml.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            reversible_1=reversible,
            device_1=device,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            reversible_2=reversible,
            device_2=device,
            method="python",
        )
        ll_expected = -565743.9829
        np.testing.assert_almost_equal(ll, ll_expected, decimal=1)
