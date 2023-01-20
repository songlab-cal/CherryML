import filecmp
import os
import tempfile
import unittest

import numpy as np

from cherryml.estimation import em_lg_xrate
from cherryml.estimation._em_lg import _translate_tree_and_msa_to_stock_format
from cherryml.estimation._em_lg_xrate import (
    _install_xrate,
    _translate_rate_matrix_from_xrate_format,
    _translate_rate_matrix_to_xrate_format,
)
from cherryml.io import read_rate_matrix
from cherryml.markov_chain import get_lg_path
from cherryml.utils import get_amino_acids

DATA_DIR = "./tests/estimation_tests/test_input_data"


class TestEMLG_XRATE(unittest.TestCase):
    def test_installation(self):
        """
        Test that XRATE is installed
        """
        _install_xrate()

    def test_translate_tree_and_msa_to_stock_format(self):
        """
        The expected output is at ./test_input_data/stock_dir/fam1_{i}.txt
        """
        with tempfile.TemporaryDirectory() as stock_dir:
            res = _translate_tree_and_msa_to_stock_format(
                "fam1",
                f"{DATA_DIR}/tree_dir",
                f"{DATA_DIR}/msa_dir",
                f"{DATA_DIR}/site_rates_dir",
                get_amino_acids(),
                stock_dir,
                missing_data_character=".",
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir_xrate/fam1_{i}.txt"
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert filecmp.cmp(filepath_1, filepath_2)

    def test_translate_tree_and_msa_to_stock_format_with_trifurcations(
        self,
    ):  # noqa
        """
        The expected output is at
        ./test_input_data/stock_dir_trifurcation/fam1_{i}.txt
        """
        with tempfile.TemporaryDirectory() as stock_dir:
            res = _translate_tree_and_msa_to_stock_format(
                "fam1",
                f"{DATA_DIR}/tree_dir_trifurcation",
                f"{DATA_DIR}/msa_dir",
                f"{DATA_DIR}/site_rates_dir",
                get_amino_acids(),
                stock_dir,
                missing_data_character=".",
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = (
                    f"{DATA_DIR}/stock_dir_trifurcation_xrate/fam1_{i}.txt"
                )
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert filecmp.cmp(filepath_1, filepath_2)

    def test_translate_rate_matrix_to_xrate_format(self):
        """
        Expected output is at ./test_input_data/xrate_init.txt
        """
        with tempfile.NamedTemporaryFile("w") as xrate_init_file:
            xrate_init_path = xrate_init_file.name
            _translate_rate_matrix_to_xrate_format(
                initialization_rate_matrix_path=get_lg_path(),
                xrate_init_path=xrate_init_path,
            )
            filepath_1 = f"{DATA_DIR}/xrate_init.txt"
            filepath_2 = xrate_init_path
            file_1_lines = open(filepath_1).read().split("\n")
            file_2_lines = open(filepath_2).read().split("\n")
            for line_1, line_2 in zip(file_1_lines, file_2_lines):
                tokens_1, tokens_2 = line_1.split(), line_2.split()
                for token_1, token_2 in zip(tokens_1, tokens_2):
                    try:
                        np.testing.assert_almost_equal(
                            float(token_1.strip(",").replace(")", "")),
                            float(token_2.strip(",").replace(")", "")),
                        )
                    except Exception:
                        self.assertEqual(token_1, token_2)

    def test_run_xrate_from_CLI(self):
        """
        Run XRATE from CLI
        """
        with tempfile.NamedTemporaryFile("w") as xrate_learned_rate_matrix_file:
            xrate_learned_rate_matrix_path = xrate_learned_rate_matrix_file.name
            command = (
                "cherryml/estimation/xrate/bin/xrate "
                + "".join(
                    [
                        f" {DATA_DIR}/stock_dir_xrate/fam1_{i}.txt"
                        for i in range(3)
                    ]
                )
                + f" -g {DATA_DIR}/xrate_init_small.txt"
                + f" -t {xrate_learned_rate_matrix_path}"
                + f" -log 6 -f 3 -mi 0.001"
            )
            print(f"Going to run: {command}")
            os.system(command)
            with open(xrate_learned_rate_matrix_path, "r") as xrate_output_file:
                xrate_file_contents = xrate_output_file.read()
                assert "(mutate (from (" in xrate_file_contents

    def test_translate_rate_matrix_from_xrate_format(self):
        with tempfile.NamedTemporaryFile("w") as learned_rate_matrix_file:
            learned_rate_matrix_path = learned_rate_matrix_file.name
            _translate_rate_matrix_from_xrate_format(
                f"{DATA_DIR}/xrate_learned_rate_matrix.txt",
                alphabet=["A", "R", "N", "Q"],
                learned_rate_matrix_path=learned_rate_matrix_path,
            )
            filepath_1 = f"{DATA_DIR}/learned_rate_matrix_xrate.txt"
            filepath_2 = learned_rate_matrix_path
            assert filecmp.cmp(filepath_1, filepath_2)

    def test_run_xrate_from_python_api(self):
        """
        Run XRATE from our CLI
        """
        with tempfile.TemporaryDirectory() as learned_rate_matrix_dir:
            em_lg_xrate(
                tree_dir=f"{DATA_DIR}/tree_dir",
                msa_dir=f"{DATA_DIR}/msa_dir",
                site_rates_dir=f"{DATA_DIR}/site_rates_dir",
                families=["fam1"],
                initialization_rate_matrix_path=f"{DATA_DIR}/historian_init_small.txt",  # noqa
                output_rate_matrix_dir=learned_rate_matrix_dir,
                extra_command_line_args="-log 6 -f 3 -mi 0.001",
            )
            learned_rate_matrix = read_rate_matrix(
                os.path.join(learned_rate_matrix_dir, "result.txt")
            )
            np.testing.assert_almost_equal(
                learned_rate_matrix.to_numpy(),
                read_rate_matrix(
                    f"{DATA_DIR}/learned_rate_matrix_xrate.txt"
                ).to_numpy(),
            )
