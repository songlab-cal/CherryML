import os
import tempfile
import unittest
from collections import defaultdict

import numpy as np
import pandas as pd
from parameterized import parameterized

from cherryml.io import (
    read_msa,
    read_probability_distribution,
    read_tree,
    write_contact_map,
    write_site_rates,
)
from cherryml.simulation import simulate_msas
from tests.utils import create_synthetic_contact_map

DATA_DIR = "./tests/simulation_tests/test_input_data"


def check_empirical_counts(
    empirical_counts,  # defaultdict of Dict[str, int]
    expected_distribution: pd.DataFrame,
    rel_error_tolerance: float,
):
    """
    Raises Exception if the deviation is too large.
    """
    sample_size = sum(empirical_counts.values())
    for state in expected_distribution.index:
        expected_counts = sample_size * float(expected_distribution.loc[state])
        abs_error = empirical_counts[state] - expected_counts
        rel_error = abs(abs_error / expected_counts)
        if rel_error > rel_error_tolerance:
            raise Exception(
                f"Expected state {state} to appear approximately "
                f"{expected_counts} times, but found {empirical_counts[state]}."
            )


class TestSimulation(unittest.TestCase):
    @parameterized.expand(
        [("3 processes", 3)]  # , ("2 processes", 2), ("serial", 1)]
    )
    def test_simulate_msas_extreme_model(self, name, num_processes):
        families = ["fam1", "fam2", "fam3"]
        tree_dir = "./tests/simulation_tests/test_input_data/tree_dir"
        with tempfile.TemporaryDirectory() as synthetic_contact_map_dir:
            # synthetic_contact_map_dir = "tests/simulation_tests/test_input_data/synthetic_contact_map_dir"  # noqa
            # Create synthetic contact maps
            contact_maps = {}
            for i, family in enumerate(families):
                num_sites = 100
                num_sites_in_contact = 50
                contact_map = create_synthetic_contact_map(
                    num_sites=num_sites,
                    num_sites_in_contact=num_sites_in_contact,
                    random_seed=i,
                )
                contact_map_path = os.path.join(
                    synthetic_contact_map_dir, family + ".txt"
                )
                write_contact_map(contact_map, contact_map_path)
                contact_maps[family] = contact_map

            with tempfile.TemporaryDirectory() as synthetic_site_rates_dir:
                # synthetic_site_rates_dir = "tests/simulation_tests/test_input_data/synthetic_site_rates_dir"  # noqa
                for i, family in enumerate(families):
                    site_rates = [1.0 * np.log(1 + i) for i in range(num_sites)]
                    site_rates_path = os.path.join(
                        synthetic_site_rates_dir, family + ".txt"
                    )
                    write_site_rates(site_rates, site_rates_path)

                with tempfile.TemporaryDirectory() as simulated_msa_dir:
                    # simulated_msa_dir = "tests/simulation_tests/test_input_data/simulated_msa_dir"  # noqa
                    simulate_msas(
                        tree_dir=tree_dir,
                        site_rates_dir=synthetic_site_rates_dir,
                        contact_map_dir=synthetic_contact_map_dir,
                        families=families,
                        amino_acids=["S", "T"],
                        pi_1_path=f"{DATA_DIR}/extreme_model/pi_1.txt",
                        Q_1_path=f"{DATA_DIR}/extreme_model/Q_1.txt",
                        pi_2_path=f"{DATA_DIR}/extreme_model/pi_2.txt",
                        Q_2_path=f"{DATA_DIR}/extreme_model/Q_2.txt",
                        strategy="all_transitions",
                        output_msa_dir=simulated_msa_dir,
                        random_seed=0,
                        num_processes=num_processes,
                    )
                    # Check that the distribution of the endings states matches
                    # the stationary distribution
                    C_1 = defaultdict(int)  # single states
                    C_2 = defaultdict(int)  # co-evolving pairs
                    for family in families:
                        tree_path = os.path.join(tree_dir, family + ".txt")
                        tree = read_tree(
                            tree_path=tree_path,
                        )
                        msa = read_msa(
                            os.path.join(simulated_msa_dir, family + ".txt")
                        )
                        contact_map = contact_maps[family]
                        contacting_pairs = list(
                            zip(*np.where(contact_map == 1))
                        )
                        contacting_pairs = [
                            (i, j) for (i, j) in contacting_pairs if i < j
                        ]
                        contacting_sites = list(sum(contacting_pairs, ()))
                        sites_indep = [
                            i
                            for i in range(num_sites)
                            if i not in contacting_sites
                        ]
                        for node in tree.nodes():
                            if node not in msa:
                                raise Exception(
                                    f"Missing sequence for node: {node}"
                                )
                            if tree.is_leaf(node):
                                seq = msa[node]
                                for i in sites_indep:
                                    state = seq[i]
                                    C_1[state] += 1
                                for (i, j) in contacting_pairs:
                                    state = seq[i] + seq[j]
                                    C_2[state] += 1

                    # Check that almost all single sites are 'S' and pair of
                    # sites are 'TT'.
                    if C_1["S"] / sum(C_1.values()) < 0.95:
                        raise Exception(
                            "Almost all the single-site leaf states should be "
                            f"'S', but found: {dict(C_1)}"
                        )
                    if C_2["TT"] / sum(C_2.values()) < 0.95:
                        raise Exception(
                            "Almost all the co-evolution leaf states should be "
                            f"'TT', but found: {dict(C_2)}"
                        )

    @parameterized.expand(
        [("3 processes", 3)]  # , ("2 processes", 2), ("serial", 1)]
    )
    def test_simulate_msas_normal_model(self, name, num_processes):
        families = ["fam1", "fam2", "fam3"]
        tree_dir = "./tests/simulation_tests/test_input_data/tree_dir"
        with tempfile.TemporaryDirectory() as synthetic_contact_map_dir:
            # synthetic_contact_map_dir = "tests/simulation_tests/test_input_data/synthetic_contact_map_dir"  # noqa
            contact_maps = {}
            for i, family in enumerate(families):
                num_sites = 1000
                num_sites_in_contact = 500
                contact_map = create_synthetic_contact_map(
                    num_sites=num_sites,
                    num_sites_in_contact=num_sites_in_contact,
                    random_seed=i,
                )
                contact_map_path = os.path.join(
                    synthetic_contact_map_dir, family + ".txt"
                )
                write_contact_map(contact_map, contact_map_path)
                contact_maps[family] = contact_map

            with tempfile.TemporaryDirectory() as synthetic_site_rates_dir:
                # synthetic_site_rates_dir = "tests/simulation_tests/test_input_data/synthetic_site_rates_dir"  # noqa
                for i, family in enumerate(families):
                    site_rates = [1.0 * np.log(1 + i) for i in range(num_sites)]
                    site_rates_path = os.path.join(
                        synthetic_site_rates_dir, family + ".txt"
                    )
                    write_site_rates(site_rates, site_rates_path)

                with tempfile.TemporaryDirectory() as simulated_msa_dir:
                    # simulated_msa_dir = "tests/simulation_tests/test_input_data/simulated_msa_dir"  # noqa
                    simulate_msas(
                        tree_dir=tree_dir,
                        site_rates_dir=synthetic_site_rates_dir,
                        contact_map_dir=synthetic_contact_map_dir,
                        families=families,
                        amino_acids=["S", "T"],
                        pi_1_path=f"{DATA_DIR}/normal_model/pi_1.txt",
                        Q_1_path=f"{DATA_DIR}/normal_model/Q_1.txt",
                        pi_2_path=f"{DATA_DIR}/normal_model/pi_2.txt",
                        Q_2_path=f"{DATA_DIR}/normal_model/Q_2.txt",
                        strategy="all_transitions",
                        output_msa_dir=simulated_msa_dir,
                        random_seed=0,
                        num_processes=num_processes,
                    )
                    # Check that the distribution of the endings states matches
                    # the stationary distribution
                    C_1 = defaultdict(int)  # single states
                    C_2 = defaultdict(int)  # co-evolving pairs
                    for family in families:
                        tree_path = os.path.join(tree_dir, family + ".txt")
                        tree = read_tree(
                            tree_path=tree_path,
                        )
                        msa = read_msa(
                            os.path.join(simulated_msa_dir, family + ".txt")
                        )
                        contact_map = contact_maps[family]
                        contacting_pairs = list(
                            zip(*np.where(contact_map == 1))
                        )
                        contacting_pairs = [
                            (i, j) for (i, j) in contacting_pairs if i < j
                        ]
                        contacting_sites = list(sum(contacting_pairs, ()))
                        sites_indep = [
                            i
                            for i in range(num_sites)
                            if i not in contacting_sites
                        ]
                        for node in tree.nodes():
                            if node not in msa:
                                raise Exception(
                                    f"Missing sequence for node: {node}"
                                )
                            if tree.is_leaf(node):
                                seq = msa[node]
                                for i in sites_indep:
                                    state = seq[i]
                                    C_1[state] += 1
                                for (i, j) in contacting_pairs:
                                    state = seq[i] + seq[j]
                                    C_2[state] += 1

                    pi_1 = read_probability_distribution(
                        f"{DATA_DIR}/normal_model/pi_1.txt"
                    )
                    pi_2 = read_probability_distribution(
                        f"{DATA_DIR}/normal_model/pi_2.txt"
                    )
                    check_empirical_counts(C_1, pi_1, rel_error_tolerance=0.10)
                    check_empirical_counts(C_2, pi_2, rel_error_tolerance=0.10)
