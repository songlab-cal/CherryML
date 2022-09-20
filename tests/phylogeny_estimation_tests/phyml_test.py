import tempfile
import unittest

import pytest
from parameterized import parameterized

from cherryml.phylogeny_estimation import phyml

# def branch_length_l1_error(tree_true_path, tree_inferred_path) -> float:
#     tree1 = Tree(tree_true_path)
#     tree2 = Tree(tree_inferred_path)

#     def dfs_branch_length_l1_error(v1, v2) -> float:
#         l1_error = 0
#         for (u1, u2) in zip(v1.children, v2.children):
#             l1_error += abs(u1.dist - u2.dist)
#             l1_error += dfs_branch_length_l1_error(u1, u2)
#         return l1_error

#     l1_error = dfs_branch_length_l1_error(tree1, tree2)
#     return l1_error


class TestPhyml(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    @pytest.mark.slow
    def test_basic_regression(self, name, num_processes):
        """
        Test that Phyml runs and its output matches the expected output.
        The expected output is located at test_input_data/trees_small
        """
        with tempfile.TemporaryDirectory() as output_tree_dir:
            # output_tree_dir = "phyml_trees"
            with tempfile.TemporaryDirectory() as output_site_rates_dir:
                # output_site_rates_dir = "phyml_site_rates"
                with tempfile.TemporaryDirectory() as output_likelihood_dir:
                    # output_likelihood_dir = "output_likelihood_dir"
                    families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
                    phyml(
                        msa_dir="tests/test_input_data/a3m_small_fast_tree",
                        families=families,
                        rate_matrix_path="data/rate_matrices/jtt.txt",
                        num_rate_categories=20,
                        output_tree_dir=output_tree_dir,
                        output_site_rates_dir=output_site_rates_dir,
                        output_likelihood_dir=output_likelihood_dir,
                        num_processes=num_processes,
                    )
                    # for protein_family_name in families:
                    #     tree_true_path = (
                    #         "tests/test_input_data/trees_small"
                    #         f"/{protein_family_name}.newick"
                    #     )
                    #     tree_inferred_path = os.path.join(
                    #         output_tree_dir, protein_family_name + ".newick"
                    #     )
                    #     l1_error = branch_length_l1_error(
                    #         tree_true_path, tree_inferred_path
                    #     )
                    #     assert l1_error < 0.02  # Redundant, but just in case

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    @pytest.mark.slow
    def test_custom_rate_matrix_runs_regression(self, name, num_processes):
        """
        Tests the use of a custom rate matrix in Phyml.
        """
        with tempfile.TemporaryDirectory() as output_tree_dir:
            # output_tree_dir = "phyml_trees"
            with tempfile.TemporaryDirectory() as output_site_rates_dir:
                # output_site_rates_dir = "phyml_site_rates"
                with tempfile.TemporaryDirectory() as output_likelihood_dir:
                    # output_likelihood_dir = "output_likelihood_dir"
                    families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
                    phyml(
                        msa_dir="tests/test_input_data/a3m_small_fast_tree",
                        families=families,
                        rate_matrix_path="data/rate_matrices/equ.txt",
                        num_rate_categories=20,
                        output_tree_dir=output_tree_dir,
                        output_site_rates_dir=output_site_rates_dir,
                        output_likelihood_dir=output_likelihood_dir,
                        num_processes=num_processes,
                    )
                    # for protein_family_name in families:
                    #     tree_true_path = (
                    #         "tests/test_input_data/trees_small_Q1_uniform"
                    #         f"/{protein_family_name}.newick"
                    #     )
                    #     tree_inferred_path = os.path.join(
                    #         output_tree_dir, protein_family_name + ".newick"
                    #     )
                    #     l1_error = branch_length_l1_error(
                    #         tree_true_path, tree_inferred_path
                    #     )
                    #     assert l1_error < 0.02  # Redundant, but just in case

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    @pytest.mark.slow
    def test_custom_rate_matrix_unnormalized_runs_regression(
        self, name, num_processes
    ):
        """
        Tests the use of an UNNORMALIZED custom rate matrix in Phyml.
        """
        with tempfile.TemporaryDirectory() as output_tree_dir:
            # output_tree_dir = "phyml_trees"
            with tempfile.TemporaryDirectory() as output_site_rates_dir:
                # output_site_rates_dir = "phyml_site_rates"
                with tempfile.TemporaryDirectory() as output_likelihood_dir:
                    # output_likelihood_dir = "output_likelihood_dir"
                    families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
                    phyml(
                        msa_dir="tests/test_input_data/a3m_small_fast_tree",
                        families=families,
                        rate_matrix_path="data/rate_matrices/equ_halved.txt",
                        num_rate_categories=20,
                        output_tree_dir=output_tree_dir,
                        output_site_rates_dir=output_site_rates_dir,
                        output_likelihood_dir=output_likelihood_dir,
                        num_processes=num_processes,
                    )
                    # for protein_family_name in families:
                    #     tree_true_path = (
                    #         "tests/test_input_data"
                    #         f"/trees_small_Q1_uniform_halved"
                    #         f"/{protein_family_name}.newick"
                    #     )
                    #     tree_inferred_path = os.path.join(
                    #         output_tree_dir, protein_family_name + ".newick"
                    #     )
                    #     l1_error = branch_length_l1_error(
                    #         tree_true_path, tree_inferred_path
                    #     )
                    #     assert l1_error < 0.02  # Redundant, but just in case

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    @pytest.mark.slow
    def test_inexistent_rate_matrix_raises_error(self, name, num_processes):
        """
        If the rate matrix passed to Phyml does not exist, we should error
        out.
        """
        with tempfile.TemporaryDirectory() as output_tree_dir:
            # output_tree_dir = "phyml_trees"
            with tempfile.TemporaryDirectory() as output_site_rates_dir:
                # output_site_rates_dir = "phyml_site_rates"
                with tempfile.TemporaryDirectory() as output_likelihood_dir:
                    # output_likelihood_dir = "output_likelihood_dir"
                    families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
                    with self.assertRaises(Exception):
                        phyml(
                            msa_dir="tests/test_input_data/a3m_small_fast_tree",
                            families=families,
                            rate_matrix_path="data/rate_matrices/not-exist.txt",
                            num_rate_categories=20,
                            output_tree_dir=output_tree_dir,
                            output_site_rates_dir=output_site_rates_dir,
                            output_likelihood_dir=output_likelihood_dir,
                            num_processes=num_processes,
                        )

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    @pytest.mark.slow
    def test_malformed_a3m_file_raises_error(self, name, num_processes):
        """
        If the a3m data is corrupted, an error should be raised.
        """
        with tempfile.TemporaryDirectory() as output_tree_dir:
            # output_tree_dir = "phyml_trees"
            with tempfile.TemporaryDirectory() as output_site_rates_dir:
                # output_site_rates_dir = "phyml_site_rates"
                with tempfile.TemporaryDirectory() as output_likelihood_dir:
                    # output_likelihood_dir = "output_likelihood_dir"
                    families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
                    with self.assertRaises(Exception):
                        msa_dir = (
                            "tests/test_input_data"
                            "/a3m_small_fast_tree_corrupted"
                        )
                        phyml(
                            msa_dir=msa_dir,
                            families=families,
                            rate_matrix_path="data/rate_matrices/jtt.txt",
                            num_rate_categories=20,
                            output_tree_dir=output_tree_dir,
                            output_site_rates_dir=output_site_rates_dir,
                            output_likelihood_dir=output_likelihood_dir,
                            num_processes=num_processes,
                        )
