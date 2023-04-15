"""
Classical EM for the LG model, using XRATE.
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from cherryml import caching
from cherryml.io import read_rate_matrix, write_rate_matrix
from cherryml.markov_chain import compute_stationary_distribution
from cherryml.utils import pushd

from ._em_lg import _translate_trees_and_msas_to_stock_format


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _xrate_is_installed_on_system() -> bool:
    """
    Check whether `xrate` program is installed on the system.
    """
    res = os.popen("which xrate").read()
    if len(res) > 0:
        # is installed
        return True
    else:
        # is not installed
        return False


def _install_xrate() -> str:
    """
    Makes sure that XRATE is installed on the system, and if not, installs
    it.

    Returns the path to the XRATE binary.
    """
    if _xrate_is_installed_on_system():
        return "xrate"
    logger = logging.getLogger(__name__)
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "xrate"
    )
    bin_path = os.path.join(dir_path, "bin/xrate")
    if not os.path.exists(dir_path):
        git_clone_command = f"git clone https://github.com/ihh/dart {dir_path}"
        logger.info(f"Going to run: {git_clone_command}")
        os.system(git_clone_command)
    if not os.path.exists(bin_path):
        with pushd(dir_path):
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info("./configure --without-guile ...")
            os.system("./configure --without-guile")
            logger.info("make xrate ...")
            os.system("make xrate")
            logger.info("Done!")
            if not os.path.exists(bin_path):
                raise Exception("Was not able to build XRATE")
    return bin_path


def _translate_rate_matrix_from_xrate_format(
    xrate_learned_rate_matrix_path: str,
    alphabet: List[str],
    learned_rate_matrix_path: str,
) -> None:
    """
    Given the xrate_learned_rate_matrix_path and the alphabet, translates the
    rate matrix to our format, writing it to learned_rate_matrix_path.
    """
    res_df = pd.DataFrame(
        np.zeros(shape=(len(alphabet), len(alphabet))),
        index=alphabet,
        columns=alphabet,
    )
    with open(xrate_learned_rate_matrix_path, "r") as file:
        lines = list(file)
        for line in lines:
            if line.startswith("  (mutate (from (") and "rate" in line:
                aa1 = line[17].upper()
                aa2 = line[26].upper()
                rate = float(line.replace(")", "").split(" ")[-1])
                res_df.loc[aa1, aa2] = rate
                res_df.loc[aa1, aa1] -= rate
    rate_matrix = res_df.to_numpy()
    write_rate_matrix(rate_matrix, alphabet, learned_rate_matrix_path)


def _translate_rate_matrix_to_xrate_format(
    initialization_rate_matrix_path: str,
    xrate_init_path: str,
) -> None:
    """
    Given the rate matrix used to initialize XRATE (in our CherryML format),
    we convert it into the XRATE grammar format, written out to xrate_init_path.
    """
    Q_df = read_rate_matrix(initialization_rate_matrix_path)
    Q = Q_df.to_numpy()
    alphabet = list(Q_df.index)
    res = """;; Grammar nullprot
;;
(grammar
 (name nullprot)
 (update-rates 1)
 (update-rules 1)

 ;; Transformation rules for grammar symbols

 ;; State Start
 ;;
 (transform (from (Start)) (to (S0)) (prob 0.5))
 (transform (from (Start)) (to ()) (prob 0.5))

 ;; State S0
 ;;
 (transform (from (S0)) (to (A0 S0*)) (gaps-ok)
  (minlen 1))
 (transform (from (S0*)) (to ()) (prob 0.5))
 (transform (from (S0*)) (to (S0)) (prob 0.5))

 ;; Markov chain substitution models

 (chain
  (update-policy rev)
  (terminal (A0))

  ;; initial probability distribution
"""
    pi = compute_stationary_distribution(Q)
    for i, aa in enumerate(alphabet):
        res += f"  (initial (state ({aa.lower()})) (prob {pi[i]}))\n"
    res += "\n"
    res += "  ;; mutation rates\n"
    for i, aa1 in enumerate(alphabet):
        for j, aa2 in enumerate(alphabet):
            if i != j:
                res += f"  (mutate (from ({aa1.lower()})) (to ({aa2.lower()})) (rate {Q[i, j]}))\n"  # noqa
    res += """ )  ;; end chain A0

)  ;; end grammar nullprot

;; Alphabet Protein
;;
(alphabet
 (name Protein)
"""
    res += " (token (" + " ".join([aa1.lower() for aa1 in alphabet]) + "))\n"
    res += """ (wildcard *)
)  ;; end alphabet Protein

"""
    with open(xrate_init_path, "w") as outfile:
        outfile.write(res)


def run_xrate(
    xrate_bin_path: str,
    stock_input_paths: List[str],
    xrate_grammar: Optional[str],
    extra_command_line_args: str,
    output_path: str,
    logfile: Optional[str] = None,
    estimate_trees: bool = False,
):
    """
    Run XRATE on the stock_input_paths with the initialization xrate_grammar.
    """
    logger = logging.getLogger(__name__)

    # We create symlinks to the MSAs to reduce the length of the command,
    # since there is a command line maximum length imposed by the OS.
    with tempfile.TemporaryDirectory() as tmpdir:
        stock_input_paths_symlinks = []
        for i, stock_input_path in enumerate(stock_input_paths):
            stock_input_path_symlink = os.path.join(tmpdir, f"{i}.stock")
            os.symlink(
                os.path.abspath(stock_input_path), stock_input_path_symlink
            )
            stock_input_paths_symlinks.append(stock_input_path_symlink)

        def get_command(stock_filepaths: List[str]) -> str:
            """
            Given the stock_filepaths, returns the command for running
            XRATE on those stock_filepaths. The outdir, etc. are
            taken from the current context. This is just used to
            easily avoid code duplication.
            """
            if estimate_trees:
                cmd = f"{xrate_bin_path} {' '.join(stock_filepaths)} -e {xrate_grammar} -g {xrate_grammar} -t {output_path} {extra_command_line_args}"  # noqa
            else:
                cmd = f"{xrate_bin_path} {' '.join(stock_filepaths)} -g {xrate_grammar} -t {output_path} {extra_command_line_args}"  # noqa
            if logfile is not None:
                cmd += f" 2>&1 | tee {logfile}"
            return cmd

        cmd = get_command(stock_input_paths)
        cmd_with_symlinks = get_command(stock_input_paths_symlinks)

        # Write the command to a file and run it from there with bash because
        # running directly subprocess.run(cmd_with_symlinks) fails due to
        # command length limit.
        bash_script_filepath = os.path.join(tmpdir, "run_xrate.sh")
        with open(bash_script_filepath, "w") as bash_script_file:
            bash_script_file.write(cmd_with_symlinks)
            # This is key, or else the call below will fail!
            bash_script_file.flush()
            # We log the command w/o symlinks to be able to run it manually for
            # debugging.
            logger.info(f"Original command:\n{cmd}")
            logger.info(
                f"Running original command with symlinks:\n{cmd_with_symlinks}"
            )
            st = time.time()
            subprocess.run(
                f"bash {bash_script_filepath}", shell=True, check=True
            )
            return time.time() - st


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
    exclude_args=[],
    write_extra_log_files=True,
)
def em_lg_xrate(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    families: List[str],
    initialization_rate_matrix_path: str,
    output_rate_matrix_dir: Optional[str] = None,
    extra_command_line_args: str = "-log 6 -f 3 -mi 0.000001",  # noqa
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

    xrate_bin_path = _install_xrate()
    alphabet = list(read_rate_matrix(initialization_rate_matrix_path).index)

    with tempfile.TemporaryDirectory() as stock_dir:
        with tempfile.NamedTemporaryFile("w") as xrate_init_file:
            xrate_init_path = xrate_init_file.name
            with tempfile.NamedTemporaryFile(
                "w"
            ) as xrate_learned_rate_matrix_file:
                xrate_learned_rate_matrix_path = (
                    xrate_learned_rate_matrix_file.name
                )

                # Translate data from friendly format to historian format.
                new_families = _translate_trees_and_msas_to_stock_format(
                    tree_dir,
                    msa_dir,
                    site_rates_dir,
                    stock_dir,
                    alphabet,
                    families,
                    missing_data_character=".",
                )
                _translate_rate_matrix_to_xrate_format(
                    initialization_rate_matrix_path,
                    xrate_init_path,
                )

                # Run XRATE
                runtime = run_xrate(
                    xrate_bin_path=xrate_bin_path,
                    stock_input_paths=[
                        os.path.join(stock_dir, family + ".txt")
                        for family in new_families
                    ],
                    xrate_grammar=xrate_init_path,
                    extra_command_line_args=extra_command_line_args,
                    output_path=xrate_learned_rate_matrix_path,
                    logfile=os.path.join(
                        output_rate_matrix_dir, "xrate_log.txt"
                    ),
                    estimate_trees=False,
                )

                # Translate results back
                learned_rate_matrix_path = os.path.join(
                    output_rate_matrix_dir,
                    "result.txt",
                )
                _translate_rate_matrix_from_xrate_format(
                    xrate_learned_rate_matrix_path,
                    alphabet,
                    learned_rate_matrix_path,
                )

                # Write profiling information
                profiling_path = os.path.join(
                    output_rate_matrix_dir,
                    "profiling.txt",
                )
                with open(profiling_path, "w") as profiling_file:
                    profiling_file.write(f"Total time: {runtime} s")
