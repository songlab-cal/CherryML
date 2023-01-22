import filecmp
import json
import os
import tempfile
import unittest

import numpy as np

from cherryml.estimation import em_lg
from cherryml.estimation._em_lg import (
    _install_historian,
    _translate_rate_matrix_from_historian_format,
    _translate_rate_matrix_to_historian_format,
    _translate_tree_and_msa_to_stock_format,
)
from cherryml.io import read_rate_matrix
from cherryml.markov_chain import get_lg_path
from cherryml.utils import get_amino_acids

DATA_DIR = "./tests/estimation_tests/test_input_data"


class TestEMLG(unittest.TestCase):
    def test_installation(self):
        """
        Test that Historian is installed
        """
        _install_historian()

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
                missing_data_character="x",
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir/fam1_{i}.txt"
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
                missing_data_character="x",
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir_trifurcation/fam1_{i}.txt"
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert filecmp.cmp(filepath_1, filepath_2)

    def test_translate_rate_matrix_to_historian_format(self):
        """
        Expected output is at ./test_input_data/historian_init.json
        """
        with tempfile.NamedTemporaryFile("w") as historian_init_file:
            historian_init_path = historian_init_file.name
            _translate_rate_matrix_to_historian_format(
                initialization_rate_matrix_path=get_lg_path(),
                historian_init_path=historian_init_path,
                missing_data_character="x",
            )
            filepath_1 = f"{DATA_DIR}/historian_init.json"
            filepath_2 = historian_init_path
            file_1_lines = open(filepath_1).read().split("\n")
            file_2_lines = open(filepath_2).read().split("\n")
            for line_1, line_2 in zip(file_1_lines, file_2_lines):
                tokens_1, tokens_2 = line_1.split(), line_2.split()
                for token_1, token_2 in zip(tokens_1, tokens_2):
                    try:
                        np.testing.assert_almost_equal(
                            float(token_1.strip(",")), float(token_2.strip(","))
                        )
                    except Exception:
                        self.assertEqual(token_1, token_2)

    def test_run_historian_from_CLI(self):
        """
        Run Historian from CLI
        """
        with tempfile.NamedTemporaryFile(
            "w"
        ) as historian_learned_rate_matrix_file:
            historian_learned_rate_matrix_path = (
                historian_learned_rate_matrix_file.name
            )
            command = (
                "cherryml/estimation/historian/bin/historian fit"
                + "".join(
                    [f" {DATA_DIR}/stock_dir/fam1_{i}.txt" for i in range(3)]
                )
                + f" -model {DATA_DIR}/historian_init_small.json"
                + " -band 0"
                + f" -fixgaprates > {historian_learned_rate_matrix_path} -v2"
            )
            print(f"Going to run: {command}")
            os.system(command)
            with open(historian_learned_rate_matrix_path) as json_file:
                learned_rate_matrix_json = json.load(json_file)
                assert "subrate" in learned_rate_matrix_json.keys()

    def test_translate_rate_matrix_from_historian_format(self):
        with tempfile.NamedTemporaryFile("w") as learned_rate_matrix_file:
            learned_rate_matrix_path = learned_rate_matrix_file.name
            _translate_rate_matrix_from_historian_format(
                f"{DATA_DIR}/historian_learned_rate_matrix.json",
                alphabet=["A", "R", "N", "Q"],
                learned_rate_matrix_path=learned_rate_matrix_path,
            )
            filepath_1 = f"{DATA_DIR}/learned_rate_matrix.txt"
            filepath_2 = learned_rate_matrix_path
            assert filecmp.cmp(filepath_1, filepath_2)

    def test_run_historian_from_python_api(self):
        """
        Run Historian from our CLI
        """
        with tempfile.TemporaryDirectory() as learned_rate_matrix_dir:
            em_lg(
                tree_dir=f"{DATA_DIR}/tree_dir",
                msa_dir=f"{DATA_DIR}/msa_dir",
                site_rates_dir=f"{DATA_DIR}/site_rates_dir",
                families=["fam1"],
                initialization_rate_matrix_path=f"{DATA_DIR}/historian_init_small.txt",  # noqa
                output_rate_matrix_dir=learned_rate_matrix_dir,
                extra_command_line_args="-band 0 -fixgaprates",
            )
            learned_rate_matrix = read_rate_matrix(
                os.path.join(learned_rate_matrix_dir, "result.txt")
            )
            np.testing.assert_almost_equal(
                learned_rate_matrix.to_numpy(),
                read_rate_matrix(
                    f"{DATA_DIR}/learned_rate_matrix.txt"
                ).to_numpy(),
            )
