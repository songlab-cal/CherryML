"""
Classical EM for the LG model, using Historian.
"""
import json
import logging
import os
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from cherryml import caching
from cherryml.io import (
    read_msa,
    read_rate_matrix,
    read_site_rates,
    read_tree,
    write_rate_matrix,
)
from cherryml.markov_chain import compute_stationary_distribution
from cherryml.utils import pushd


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _historian_is_installed_on_system() -> bool:
    """
    Check whether `historian` program is installed on the system.
    """
    res = os.popen("which historian").read()
    if len(res) > 0:
        # is installed
        return True
    else:
        # is not installed
        return False


def _install_historian() -> str:
    """
    Makes sure that Historian is installed on the system, and if not, installs
    it.

    Returns the path to the Historian binary.
    """
    if _historian_is_installed_on_system():
        return "historian"
    logger = logging.getLogger(__name__)
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "historian"
    )
    bin_path = os.path.join(dir_path, "bin/historian")
    if not os.path.exists(dir_path):
        git_clone_command = (
            f"git clone https://github.com/evoldoers/historian {dir_path}"
        )
        logger.info(f"Going to run: {git_clone_command}")
        os.system(git_clone_command)
    if not os.path.exists(bin_path):
        with pushd(dir_path):
            compile_command = "make"
            logger.info(f"Building Historian with:\n{compile_command}")
            # See https://github.com/evoldoers/historian
            os.system(compile_command)
            if not os.path.exists(bin_path):
                raise Exception("Was not able to build Historian")
    return bin_path


def _translate_tree_and_msa_to_stock_format(
    family: str,
    input_tree_dir: str,
    input_msa_dir: str,
    input_site_rates_dir: str,
    alphabet: List[str],
    output_stock_dir: str,
    missing_data_character: str,
) -> List[str]:
    """
    Translate tree and MSA to the Stockholm format.

    One Stockholm file is created for each site rate category.

    Returns the new fake protein families.
    """
    if not os.path.exists(output_stock_dir):
        os.makedirs(output_stock_dir)
    input_tree_path = os.path.join(input_tree_dir, family + ".txt")
    input_msa_path = os.path.join(input_msa_dir, family + ".txt")
    input_site_rates_path = os.path.join(input_site_rates_dir, family + ".txt")
    msa_orig = read_msa(input_msa_path)
    # Replace characters out of alphabet with missing data character

    def apply_missing_data_character(seq: str):
        new_seq = []
        alphabet_set = set(alphabet)
        for state in seq:
            if state in alphabet_set:
                new_seq.append(state)
            else:
                new_seq.append(missing_data_character)
        return new_seq

    msa = {
        seq_name: apply_missing_data_character(seq)
        for (seq_name, seq) in msa_orig.items()
    }

    site_rates = read_site_rates(input_site_rates_path)
    rate_categories = sorted(list(set(site_rates)))
    res = []
    for i, rate_category in enumerate(rate_categories):
        # Write out tree scaled by this rate category, and sites in the MSA
        fake_family = family + "_" + str(i)
        res.append(fake_family)

        stock_str = "# STOCKHOLM 1.0\n"

        tree = read_tree(input_tree_path)
        tree = tree.scaled(rate_category, node_name_prefix=fake_family + "-")
        stock_str += (
            "#=GF NH "
            + tree.to_newick_resolve_root_trifurcation(format=3)
            + "\n"
        )

        sites_with_this_rate_category = [
            i for i in range(len(site_rates)) if site_rates[i] == rate_category
        ]
        msa_str = ""
        for seq_name, seq in msa.items():
            msa_str += (
                fake_family
                + "-"
                + seq_name
                + " "
                + "".join([seq[i] for i in sites_with_this_rate_category])
                + "\n"
            )
        stock_str += msa_str

        with open(
            os.path.join(output_stock_dir, fake_family + ".txt"), "w"
        ) as output_stock_file:
            output_stock_file.write(stock_str)
    return res


def _translate_trees_and_msas_to_stock_format(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    output_stock_dir: str,
    alphabet: List[str],
    families: List[str],
    missing_data_character: str,
) -> List[str]:
    """
    Translate trees and MSAs to the Stockholm format.

    One Stockholm file is created for each site rate category.

    Returns the new fake protein families.
    """
    res = []
    for family in families:
        res += _translate_tree_and_msa_to_stock_format(
            family,
            tree_dir,
            msa_dir,
            site_rates_dir,
            alphabet,
            output_stock_dir,
            missing_data_character=missing_data_character,
        )
    return res


def _translate_rate_matrix_from_historian_format(
    historian_learned_rate_matrix_path: str,
    alphabet: List[str],
    learned_rate_matrix_path: str,
) -> None:
    with open(historian_learned_rate_matrix_path) as json_file:
        learned_rate_matrix_json = json.load(json_file)
    res = pd.DataFrame(
        np.zeros(shape=(len(alphabet), len(alphabet))),
        index=alphabet,
        columns=alphabet,
    )
    for state_1 in alphabet:
        for state_2 in alphabet:
            if state_1 != state_2:
                res.loc[state_1, state_2] = learned_rate_matrix_json["subrate"][
                    state_1
                ][state_2]
    for state_1 in alphabet:
        res.loc[state_1, state_1] = -res.loc[state_1, :].sum()
    write_rate_matrix(res, alphabet, learned_rate_matrix_path)


def _translate_rate_matrix_to_historian_format(
    initialization_rate_matrix_path: str,
    historian_init_path: str,
    missing_data_character: str,
):
    rate_matrix = read_rate_matrix(initialization_rate_matrix_path)
    alphabet = list(rate_matrix.columns)
    pi = compute_stationary_distribution(rate_matrix.to_numpy())
    res = {
        "insrate": 0.0,
        "delrate": 0.0,
        "insextprob": 0.0,
        "delextprob": 0.0,
        "alphabet": "".join(alphabet),
        "wildcard": missing_data_character,
    }
    res["rootprob"] = {state: pi[i] for (i, state) in enumerate(alphabet)}
    res["subrate"] = {}
    for state_1 in alphabet:
        res["subrate"][state_1] = {
            state_2: rate_matrix.loc[state_1, state_2]
            for state_2 in alphabet
            if state_2 != state_1
        }
    json_str = json.dumps(res, indent=4)
    with open(historian_init_path, "w") as historian_init_file:
        historian_init_file.write(json_str)


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
    exclude_args=[],
    write_extra_log_files=True,
)
def em_lg(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    families: List[str],
    initialization_rate_matrix_path: str,
    output_rate_matrix_dir: Optional[str] = None,
    extra_command_line_args: str = "-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",  # noqa
):
    """
    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves.
        families: The protein families for which to perform the computation.
        initialization_rate_matrix_path: Rate matrix used to initialize EM
            optimizer.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Going to run on {len(families)} families, with output at:"
        f"{output_rate_matrix_dir}"
    )

    if not os.path.exists(output_rate_matrix_dir):
        os.makedirs(output_rate_matrix_dir)

    historian_bin = _install_historian()
    alphabet = list(read_rate_matrix(initialization_rate_matrix_path).index)

    with tempfile.TemporaryDirectory() as stock_dir:
        with tempfile.NamedTemporaryFile("w") as historian_init_file:
            historian_init_path = historian_init_file.name
            with tempfile.NamedTemporaryFile(
                "w"
            ) as historian_learned_rate_matrix_file:
                historian_learned_rate_matrix_path = (
                    historian_learned_rate_matrix_file.name
                )

                # Translate data from friendly format to historian format.
                new_families = _translate_trees_and_msas_to_stock_format(
                    tree_dir,
                    msa_dir,
                    site_rates_dir,
                    stock_dir,
                    alphabet,
                    families,
                    missing_data_character="x",
                )
                _translate_rate_matrix_to_historian_format(
                    initialization_rate_matrix_path,
                    historian_init_path,
                    missing_data_character="x",
                )

                # Run Historian
                historian_command = (
                    f"{historian_bin}"
                    + " fit "
                    + " ".join(
                        [
                            os.path.join(stock_dir, family + ".txt")
                            for family in new_families
                        ]
                    )
                    + f" -model {historian_init_path} "
                    + " "
                    + extra_command_line_args
                    + f" > {historian_learned_rate_matrix_path}"
                )
                logger.info(f"Going to run command: {historian_command}")
                st = time.time()
                os.system(historian_command)
                et = time.time()

                # Translate results back
                learned_rate_matrix_path = os.path.join(
                    output_rate_matrix_dir,
                    "result.txt",
                )
                _translate_rate_matrix_from_historian_format(
                    historian_learned_rate_matrix_path,
                    alphabet,
                    learned_rate_matrix_path,
                )

                # Write profiling information
                profiling_path = os.path.join(
                    output_rate_matrix_dir,
                    "profiling.txt",
                )
                with open(profiling_path, "w") as profiling_file:
                    profiling_file.write(f"Total time: {et - st} s")
