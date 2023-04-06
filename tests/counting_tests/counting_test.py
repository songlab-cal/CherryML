import os
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized

from cherryml.counting import count_co_transitions, count_transitions
from cherryml.io import read_count_matrices
from cherryml.utils import amino_acids

DATA_DIR = "./tests/counting_tests/test_input_data"
families_medium = [
    "13gs_1_A",
    "19hc_1_A",
    "1a0b_1_A",
    "1a0p_1_A",
    "1a0t_1_A",
    "1a12_1_A",
    "1a17_1_A",
    "1a27_1_A",
    "1a2o_1_A",
    "1a2t_1_A",
    "1a3a_1_A",
    "1a40_1_A",
    "1a41_1_A",
    "1a45_1_A",
    "1a4a_1_A",
    "1a4m_1_A",
    "1a4p_1_A",
    "1a5k_1_B",
    "1a5t_1_B",
    "1a62_1_A",
    "1a64_1_A",
    "1a6l_1_A",
    "1a6q_1_A",
    "1a79_1_A",
    "1a7j_1_A",
    "1a82_1_A",
    "1a8h_1_A",
    "1a8l_1_A",
    "1a8m_1_A",
    "1a92_1_A",
    "1a95_1_B",
    "1a9y_1_A",
]


def check_count_matrices_are_equal(
    count_matrices_1: List[Tuple[float, pd.DataFrame]],
    count_matrices_2: List[Tuple[float, pd.DataFrame]],
) -> None:
    qs_1 = np.array([q for (q, _) in count_matrices_1])
    qs_2 = np.array([q for (q, _) in count_matrices_2])
    if not np.all(np.isclose(qs_1, qs_2)):
        raise Exception(
            f"Quantization values are different:\nExpected: "
            f"{qs_1}\nvs\nObtained: {qs_2}"
        )
    for q_idx in range(len(qs_1)):
        count_matrix_1 = count_matrices_1[q_idx][1]
        count_matrix_2 = count_matrices_2[q_idx][1]
        if list(count_matrix_1.columns) != list(count_matrix_2.columns):
            raise Exception(
                f"Count matrix columns differ:\nExpected:\n"
                f"{count_matrix_1.columns}\nvs\nObtained:\n"
                f"{count_matrix_2.columns}"
            )
        if list(count_matrix_1.index) != list(count_matrix_2.index):
            raise Exception(
                f"Count matrix indices differ:\nExpected:\n"
                f"{count_matrix_1.index}\nvs\nObtained:\n{count_matrix_2.index}"
            )

        if (
            not (np.abs(count_matrix_1.to_numpy() - count_matrix_2.to_numpy()))
            .sum()
            .sum()
            < 0.1
        ):
            raise Exception(
                f"Count matrix contents differ:\nExpected:\n{count_matrix_1}\n"
                f"vs\nObtained:\n{count_matrix_2}"
            )
    return True


class TestCountTransitionsTiny(unittest.TestCase):
    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_edges")
            count_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                site_rates_dir=f"{DATA_DIR}/tiny/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 5.01],
                edge_or_cherry="edge",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_cherries")
            count_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                site_rates_dir=f"{DATA_DIR}/tiny/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_transitions_cherries_plus_plus(self, name, num_processes):
        """
        Not a very interesting test, since there are no more cherries to pop
        except the original onces. Thus, just check that cherry++ works under
        this border case.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_cherries")
            count_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                site_rates_dir=f"{DATA_DIR}/tiny/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry++",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_transitions_cherries_plus_plus_v2(self, name, num_processes):
        """
        Basically I just took the original test for cherries and "doubled" each
        cherry. The output count matrices should thus just double.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(
                root_dir, "count_matrices_dir_cherries_plus_plus"
            )
            count_transitions(
                tree_dir=f"{DATA_DIR}/tiny_2/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny_2/msa_dir",
                site_rates_dir=f"{DATA_DIR}/tiny_2/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry++",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny_2/count_matrices_dir_cherries_plus_plus/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_co_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_edges")
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                contact_map_dir=f"{DATA_DIR}/tiny/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 5.01],
                edge_or_cherry="edge",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_co_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_co_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_cherries")
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                contact_map_dir=f"{DATA_DIR}/tiny/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_co_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_co_transitions_cherries_plus_plus(self, name, num_processes):
        """
        Not a very interesting test, since there are no more cherries to pop
        except the original onces. Thus, just check that cherry++ works under
        this border case.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_cherries")
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/tiny/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny/msa_dir",
                contact_map_dir=f"{DATA_DIR}/tiny/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry++",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny/count_co_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_count_co_transitions_cherries_plus_plus_2(
        self, name, num_processes
    ):
        """
        Basically I just took the original test for cherries and "doubled" each
        cherry. The output count matrices should thus just double.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(
                root_dir, "count_co_matrices_dir_cherries_plus_plus"
            )
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/tiny_2/tree_dir",
                msa_dir=f"{DATA_DIR}/tiny_2/msa_dir",
                contact_map_dir=f"{DATA_DIR}/tiny_2/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry++",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/tiny_2/count_co_matrices_dir_cherries_plus_plus/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )


class TestCountTransitionsMedium(unittest.TestCase):
    @parameterized.expand([("3 processes", 3)])
    @pytest.mark.slow
    def test_count_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_edges")
            count_transitions(
                tree_dir=f"{DATA_DIR}/medium/tree_dir",
                msa_dir=f"{DATA_DIR}/medium/msa_with_anc_dir",
                site_rates_dir=f"{DATA_DIR}/medium/site_rates_dir",
                families=families_medium,
                amino_acids=amino_acids,
                quantization_points=[
                    0.06 * 1.1**i for i in range(-51, 51, 1)
                ],
                edge_or_cherry="edge",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/medium/count_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("3 processes", 3)])
    @pytest.mark.slow
    def test_count_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_cherries")
            count_transitions(
                tree_dir=f"{DATA_DIR}/medium/tree_dir",
                msa_dir=f"{DATA_DIR}/medium/msa_dir",
                site_rates_dir=f"{DATA_DIR}/medium/site_rates_dir",
                families=families_medium,
                amino_acids=amino_acids,
                quantization_points=[
                    0.06 * 1.1**i for i in range(-51, 51, 1)
                ],
                edge_or_cherry="cherry",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/medium/count_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("3 processes", 3)])
    @pytest.mark.slow
    def test_count_co_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_edges")
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/medium/tree_dir",
                msa_dir=f"{DATA_DIR}/medium/msa_with_anc_dir",
                contact_map_dir=f"{DATA_DIR}/medium/contact_map_dir",
                families=families_medium,
                amino_acids=amino_acids,
                quantization_points=[0.06 * 2.0**i for i in range(-5, 5, 1)],
                edge_or_cherry="edge",
                minimum_distance_for_nontrivial_contact=7,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/medium/count_co_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("3 processes", 3)])
    @pytest.mark.slow
    def test_count_co_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_cherries")
            count_co_transitions(
                tree_dir=f"{DATA_DIR}/medium/tree_dir",
                msa_dir=f"{DATA_DIR}/medium/msa_dir",
                contact_map_dir=f"{DATA_DIR}/medium/contact_map_dir",
                families=families_medium,
                amino_acids=amino_acids,
                quantization_points=[0.06 * 2.0**i for i in range(-5, 5, 1)],
                edge_or_cherry="cherry",
                minimum_distance_for_nontrivial_contact=7,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                f"{DATA_DIR}/medium/count_co_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )
