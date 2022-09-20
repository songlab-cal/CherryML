import os
import tempfile
import unittest

import numpy as np

from cherryml.estimation import jtt_ipw
from cherryml.io import read_rate_matrix


class TestJTT(unittest.TestCase):
    def test_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        Output verified manually.
        """
        for use_ipw in [True, False]:
            with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
                for use_cached in [False, True]:
                    count_matrices_path = (
                        "tests/test_input_data" "/matrices_toy.txt"
                    )
                    jtt_ipw(
                        count_matrices_path=count_matrices_path,
                        mask_path=None,
                        use_ipw=use_ipw,
                        output_rate_matrix_dir=output_rate_matrix_dir,
                    )

                    ipw_str = "-IPW" if use_ipw else ""
                    expected_rate_matrix_path = (
                        "tests/test_input_data"
                        f"/Q1_JTT{ipw_str}_on_toy_matrix/learned_matrix.txt"
                    )
                    learned_rate_matrix_path = os.path.join(
                        output_rate_matrix_dir, "result.txt"
                    )
                    np.testing.assert_almost_equal(
                        np.loadtxt(expected_rate_matrix_path),
                        read_rate_matrix(learned_rate_matrix_path),
                    )

    def test_toy_matrix_masking(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset,
        using masking.
        Output verified manually.
        """
        for use_ipw in [True, False]:
            with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
                for use_cached in [False, True]:
                    count_matrices_path = (
                        "tests/test_input_data" "/matrices_toy.txt"
                    )
                    jtt_ipw(
                        count_matrices_path=count_matrices_path,
                        mask_path="tests/test_input_data/3x3_mask.txt",
                        use_ipw=use_ipw,
                        output_rate_matrix_dir=output_rate_matrix_dir,
                    )

                    ipw_str = "-IPW" if use_ipw else ""
                    expected_rate_matrix_path = (
                        "tests/test_input_data"
                        f"/Q1_JTT{ipw_str}_on_toy_matrix_mask"
                        "/learned_matrix.txt"
                    )
                    learned_rate_matrix_path = os.path.join(
                        output_rate_matrix_dir, "result.txt"
                    )
                    np.testing.assert_almost_equal(
                        np.loadtxt(expected_rate_matrix_path),
                        read_rate_matrix(learned_rate_matrix_path),
                    )
