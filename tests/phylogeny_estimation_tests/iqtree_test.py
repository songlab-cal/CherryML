import os
import tempfile
import unittest

import numpy as np
import pytest
from parameterized import parameterized

from cherryml import caching
from cherryml.io import read_log_likelihood, read_rate_matrix, read_tree
from cherryml.markov_chain import (
    get_equ_halved_path,
    get_equ_path,
    get_jtt_path,
    get_lg_path,
    get_wag_path,
)
from cherryml.phylogeny_estimation import _iqtree, iq_tree


class TestIQTree(unittest.TestCase):
    def test_convert_rate_matrix_to_iqtree_format(self):
        with tempfile.NamedTemporaryFile("w") as output_file:
            output_file_path = output_file.name
            output_file_path = "LG_PAML_div2"
            _iqtree.convert_rate_matrix_to_paml_format(
                rate_matrix=read_rate_matrix(get_lg_path()).to_numpy() / 2,
                output_paml_path=output_file_path,
            )
        # Now try running IQTree on this rate matrix
        # $ iqtree -s ./demo_data/plant_test/100032-100403.txt -m MF -mset LG,LG_PAML -seed 1 -redo -pre model_findel_output -st AA -wsr

    def test_iqtree_from_python_api_with_gamma_model(self):
        for extra_command_line_args in ["", "-fast"]:
            with tempfile.TemporaryDirectory() as cache_dir:
                caching.set_cache_dir(cache_dir)
                output_tree_dirs = iq_tree(
                    msa_dir="./tests/evaluation_tests/a3m_small/",
                    families=["1e7l_1_A", "5a0l_1_A", "6anz_1_B"],
                    rate_matrix_path=get_lg_path(),
                    rate_model="G",
                    num_rate_categories=4,
                    num_processes=3,
                    extra_command_line_args=extra_command_line_args,
                    rate_category_selector="posterior_mean",
                    use_model_finder=False,
                    random_seed=1,
                )
                ll_1, _ = read_log_likelihood(
                    os.path.join(
                        output_tree_dirs["output_likelihood_dir"],
                        "1e7l_1_A.txt",
                    )
                )
                assert abs(ll_1 - -200.0) < 10.0

    def test_model_finder_from_python_api(self):
        for extra_command_line_args in ["", "-fast"]:
            with tempfile.TemporaryDirectory() as cache_dir:
                caching.set_cache_dir(cache_dir)
                output_tree_dirs = iq_tree(
                    msa_dir="./tests/evaluation_tests/a3m_small/",
                    families=["1e7l_1_A", "5a0l_1_A", "6anz_1_B"],
                    rate_matrix_path=get_lg_path(),
                    rate_model=None,
                    num_rate_categories=None,
                    num_processes=3,
                    extra_command_line_args=extra_command_line_args,
                    rate_category_selector="posterior_mean",
                    use_model_finder=True,
                    random_seed=1,
                )
                ll_1, _ = read_log_likelihood(
                    os.path.join(
                        output_tree_dirs["output_likelihood_dir"],
                        "1e7l_1_A.txt",
                    )
                )
                assert abs(ll_1 - -200.0) < 10.0

    def test_model_finder_from_python_api_multi_rate_matrix(self):
        for extra_command_line_args in ["", "-fast"]:
            with tempfile.TemporaryDirectory() as cache_dir:
                caching.set_cache_dir(cache_dir)
                output_tree_dirs = iq_tree(
                    msa_dir="./tests/evaluation_tests/a3m_small/",
                    families=["1e7l_1_A", "5a0l_1_A", "6anz_1_B"],
                    rate_matrix_path=get_jtt_path()
                    + ","
                    + get_wag_path()
                    + ","
                    + get_lg_path(),
                    rate_model=None,
                    num_rate_categories=None,
                    num_processes=3,
                    extra_command_line_args=extra_command_line_args,
                    rate_category_selector="posterior_mean",
                    use_model_finder=True,
                    random_seed=1,
                )
                ll_1, _ = read_log_likelihood(
                    os.path.join(
                        output_tree_dirs["output_likelihood_dir"],
                        "1e7l_1_A.txt",
                    )
                )
                assert abs(ll_1 - -200.0) < 10.0

    def test_iqtree_from_python_api_multi_rate_matrix_errors_out(self):
        with self.assertRaises(ValueError):
            for extra_command_line_args in ["", "-fast"]:
                with tempfile.TemporaryDirectory() as cache_dir:
                    caching.set_cache_dir(cache_dir)
                    output_tree_dirs = iq_tree(
                        msa_dir="./tests/evaluation_tests/a3m_small/",
                        families=["1e7l_1_A", "5a0l_1_A", "6anz_1_B"],
                        rate_matrix_path=get_jtt_path()
                        + ","
                        + get_wag_path()
                        + ","
                        + get_lg_path(),
                        rate_model="G",
                        num_rate_categories=4,
                        num_processes=3,
                        extra_command_line_args=extra_command_line_args,
                        rate_category_selector="posterior_mean",
                        use_model_finder=False,
                        random_seed=1,
                    )
                    ll_1, _ = read_log_likelihood(
                        os.path.join(
                            output_tree_dirs["output_likelihood_dir"],
                            "1e7l_1_A.txt",
                        )
                    )
                    assert abs(ll_1 - -200.0) < 10.0

    def test_iqtree_tree_scaling(self):
        """
        On its own, IQTree is insensitive to scaling of the input rate matrices.
        Here we make sure that halving the rate matrix doubles the branch
        length.
        """
        trees = []
        with tempfile.TemporaryDirectory() as cache_dir:
            for rate_matrix_path in [get_equ_path(), get_equ_halved_path()]:
                caching.set_cache_dir(cache_dir)
                output_tree_dirs = iq_tree(
                    msa_dir="./tests/evaluation_tests/a3m_small/",
                    families=["1e7l_1_A", "5a0l_1_A", "6anz_1_B"],
                    rate_matrix_path=rate_matrix_path,
                    rate_model="G",
                    num_rate_categories=4,
                    num_processes=3,
                    extra_command_line_args="-fast",
                    rate_category_selector="posterior_mean",
                    use_model_finder=False,
                    random_seed=1,
                )
                tree = read_tree(
                    os.path.join(
                        output_tree_dirs["output_tree_dir"],
                        "1e7l_1_A.txt",
                    )
                )
                trees.append(tree)

        def tree_len(tree):
            res = sum([len for (_, _, len) in tree.edges()])
            return res

        self.assertAlmostEqual(2 * tree_len(trees[0]), tree_len(trees[1]))
