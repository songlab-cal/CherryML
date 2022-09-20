import os
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from cherryml.estimation import quantized_transitions_mle
from cherryml.io import read_mask_matrix, read_rate_matrix


class TestQuantizedTransitionsMLE(unittest.TestCase):
    @parameterized.expand([("cpu",), ("cuda",)])
    def test_smoke_toy_matrix(self, device):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            initialization_path = (
                "tests/test_input_data"
                "/3x3_pande_reversible_initialization.txt"
            )
            quantized_transitions_mle(
                count_matrices_path="tests/test_input_data/matrices_toy.txt",
                initialization_path=initialization_path,
                mask_path=None,
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device=device,
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )

    @parameterized.expand([("cpu",), ("cuda",)])
    def test_smoke_toy_matrix_raises_if_mask_and_initialization_incompatible(
        self, device
    ):
        """
        Test that RateMatrixLearner raises error if mask and
        initialization are incompatible.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            with self.assertRaises(ValueError):
                count_matrices_path = "tests/test_input_data/matrices_toy.txt"
                initialization_path = (
                    "tests/test_input_data"
                    "/3x3_pande_reversible_initialization.txt"
                )
                quantized_transitions_mle(
                    count_matrices_path=count_matrices_path,
                    initialization_path=initialization_path,
                    mask_path="tests/test_input_data/3x3_mask.txt",
                    output_rate_matrix_dir=output_rate_matrix_dir,
                    stationary_distribution_path=None,
                    rate_matrix_parameterization="pande_reversible",
                    device=device,
                    learning_rate=1e-1,
                    num_epochs=3,
                    do_adam=True,
                )

    @parameterized.expand([("cpu",), ("cuda",)])
    def test_smoke_toy_matrix_mask(self, device):
        """
        Test that RateMatrixLearner runs on a very small input dataset,
        using masking.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            initialization_path = (
                "tests/test_input_data"
                "/3x3_pande_reversible_initialization_mask.txt"
            )
            quantized_transitions_mle(
                count_matrices_path="tests/test_input_data/matrices_toy.txt",
                initialization_path=initialization_path,
                mask_path="tests/test_input_data/3x3_mask.txt",
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device=device,
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Check that the learned rate matrix has the right masking
            # structure
            mask = read_mask_matrix(
                "tests/test_input_data/3x3_mask.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )

    @parameterized.expand([("cpu",), ("cuda",)])
    def test_smoke_large_matrix(self, device):
        """
        Test that RateMatrixLearner runs on a large input dataset.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            count_matrices_path = (
                "tests/test_input_data/matrices_small"
                "/matrices_by_quantized_branch_length.txt"
            )
            quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=None,
                mask_path="tests/test_input_data/20x20_random_mask.txt",
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device=device,
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Test that the masking worked.
            mask = read_mask_matrix(
                "tests/test_input_data/20x20_random_mask.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )

    @parameterized.expand([("cpu",), ("cuda",)])
    def test_smoke_huge_matrix(self, device):
        """
        Test that RateMatrixLearner runs on a huge input dataset.
        """
        if device == "cuda" and not torch.cuda.is_available():
            return
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            count_matrices_path = (
                "tests/test_input_data/co_matrices_small"
                "/matrices_by_quantized_branch_length.txt"
            )
            mask_path = (
                "tests/test_input_data/synthetic_rate_matrices" "/mask_Q2.txt"
            )
            quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=None,
                mask_path=mask_path,
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device=device,
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Test that the masking worked.
            mask = read_mask_matrix(
                "tests/test_input_data/synthetic_rate_matrices/mask_Q2.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )
