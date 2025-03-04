import os
import pytest
import tempfile
import unittest
from typing import List

from cherryml.markov_chain import get_lg_path

DATA_DIR = "./tests/evaluation_tests/"


def run_command(
    cache_dir: str,
    families: List[str] = [],
    extra_command_line_args: str = "",
    tree_estimator_name: str = "FastTree",
) -> float:
    """
    Run evaluation command from command line.
    """
    with tempfile.NamedTemporaryFile("w") as output_file:
        output_path = output_file.name
        command = (
            "python -m cherryml.evaluation"
            f" --output_path {output_path}"
            f" --rate_matrix_path {get_lg_path()}"
            f" --msa_dir {os.path.join(DATA_DIR, 'a3m_small')}"
            f" --cache_dir {cache_dir}"
            f" --num_processes_tree_estimation {3}"
            f" --num_rate_categories {4}"
        )
        if len(families):
            command += f" --families {' '.join(families)}"
        command += f" --tree_estimator_name {tree_estimator_name}"
        if extra_command_line_args != "":
            command += f" --extra_command_line_args='{extra_command_line_args}'"
        print(f"Going to run: {command}")
        os.system(command)
        return float(open(output_path).read().split("\n")[0].split(" ")[-1])


class TestEvaluationPublicAPI(unittest.TestCase):
    @pytest.mark.slow
    def test_with_fast_tree_from_CLI(self):
        """
        Run evaluation from CLI, using FastTree
        """
        with tempfile.TemporaryDirectory() as cache_dir:
            family_names = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
            ll_all_fams = run_command(
                cache_dir=cache_dir, families=family_names
            )
            ll_all_fams_2 = run_command(cache_dir=cache_dir, families=[])
            self.assertEqual(ll_all_fams, ll_all_fams_2)
            ll_fam_i = [
                run_command(cache_dir=cache_dir, families=[family_name])
                for family_name in family_names
            ]
            self.assertAlmostEqual(sum(ll_fam_i), ll_all_fams)

            ll_all_fams_with_gamma = run_command(
                cache_dir=cache_dir,
                families=family_names,
                extra_command_line_args="-gamma",
            )
            ll_fam_i_with_gamma = [
                run_command(
                    cache_dir=cache_dir,
                    families=[family_name],
                    extra_command_line_args="-gamma",
                )
                for family_name in family_names
            ]
            self.assertAlmostEqual(
                sum(ll_fam_i_with_gamma), ll_all_fams_with_gamma
            )
            # Gamma likelihoods should be smaller than w/o Gamma.
            for ll_wo_gamma, ll_w_gamma in zip(ll_fam_i, ll_fam_i_with_gamma):
                self.assertGreater(ll_wo_gamma, ll_w_gamma)

    @pytest.mark.slow
    def test_with_phyml_from_CLI(self):
        """
        Run evaluation from CLI, using PhyML
        """
        with tempfile.TemporaryDirectory() as cache_dir:
            family_names = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
            ll_all_fams = run_command(
                cache_dir=cache_dir,
                families=family_names,
                tree_estimator_name="PhyML",
            )
            ll_all_fams_2 = run_command(
                cache_dir=cache_dir, families=[], tree_estimator_name="PhyML"
            )
            self.assertEqual(ll_all_fams, ll_all_fams_2)
            ll_fam_i = [
                run_command(
                    cache_dir=cache_dir,
                    families=[family_name],
                    tree_estimator_name="PhyML",
                )
                for family_name in family_names
            ]
            self.assertAlmostEqual(sum(ll_fam_i), ll_all_fams)
