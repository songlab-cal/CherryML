import logging
import multiprocessing
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from ete3 import Tree as TreeETE

from cherryml.caching import cached_parallel_computation, secure_parallel_output
from cherryml.io import read_msa, read_rate_matrix, write_tree
from cherryml.markov_chain import compute_stationary_distribution
from cherryml.utils import get_process_args, pushd

from ._common import name_internal_nodes, translate_tree


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _phyml_is_installed_on_system() -> bool:
    """
    Check whether `phyml` program is installed on the system.
    """
    res = os.popen("which phyml").read()
    if len(res) > 0:
        # is installed
        return True
    else:
        # is not installed
        return False


def _install_phyml() -> str:
    """
    Makes sure that PhyML is installed on the system, and if not, installs it.

    Returns the path to the PhyML binary.

    See https://github.com/stephaneguindon/phyml for installation details.
    """
    if _phyml_is_installed_on_system():
        return "phyml"
    logger = logging.getLogger(__name__)
    logger.info("Checking for PhyML ...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    phyml_path = os.path.join(dir_path, "phyml_github")
    phyml_bin_path = os.path.join(dir_path, "bin/phyml")
    if not os.path.exists(phyml_bin_path):
        logger.info(
            f"git clone https://github.com/stephaneguindon/phyml {phyml_path}"
        )
        os.system(
            f"git clone https://github.com/stephaneguindon/phyml {phyml_path}"
        )
        with pushd(phyml_path):
            commands = [
                "bash ./autogen.sh",
                f"./configure --enable-phyml --prefix={dir_path}",
                "make",
                "make install",
            ]
            for command in commands:
                logger.info(command)
                os.system(command)
            logger.info("Done!")
    if not os.path.exists(phyml_bin_path):
        raise Exception("Failed to install PhyML")
    return phyml_bin_path


def _to_paml_format(
    input_rate_matrix_path: str,
    output_rate_matrix_path: str,
) -> None:
    """
    Convert a rate matrix into the PAML format required to run PhyML.
    """
    Q = read_rate_matrix(input_rate_matrix_path).to_numpy()
    pi = compute_stationary_distribution(Q)
    E, F = Q / pi, np.diag(pi)
    res = ""
    n = Q.shape[0]
    for i in range(n):
        for j in range(i):
            res += "%.6f " % E[i, j]
        res += "\n"
    res += "\n"
    for i in range(n):
        res += "%.6f " % F[i, i]
    with open(output_rate_matrix_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def write_msa_to_phylip_format(
    msa: Dict[str, str],
    output_msa_phylip_path: str,
) -> None:
    """
    Write an MSA in phylip format.
    """
    num_sequences = len(msa)
    num_sites = len(next(iter(msa.values())))
    res = f"{num_sequences} {num_sites}\n"
    for seq_name, seq in msa.items():
        res += f"{seq_name} {seq}\n"
    open(output_msa_phylip_path, "w").write(res)


def get_site_rates_and_site_ll(
    phyml_site_ll_path: str,
) -> Tuple[List[float], List[float]]:
    df = pd.read_csv(
        phyml_site_ll_path,
        skiprows=9,
        delim_whitespace=True,
    )

    df.rename(
        columns={
            "Posterior": "Posterior mean",
            "mean": "P(D|M,rr[0]=0)",
            "P(D|M,rr[0]=0)": "NDistinctStates",
            "NDistinctStates": "drop",
        },
        inplace=True,
    )
    df.drop(columns=["drop"], inplace=True)
    site_rates = list(df["Posterior mean"])
    site_ll = list(np.log(df["P(D|M)"]))
    return site_rates, site_ll


def get_ll(phyml_stats_path: str) -> float:
    lines = open(phyml_stats_path, "r").read().strip().split("\n")
    for line in lines:
        if line.startswith(". Log-likelihood: "):
            ll = float(line.split(" ")[2])
    return ll


def _map_func(args: List):
    msa_dir = args[0]
    families = args[1]
    rate_matrix_path = args[2]
    num_rate_categories = args[3]
    output_tree_dir = args[4]
    output_site_rates_dir = args[5]
    output_likelihood_dir = args[6]
    extra_command_line_args = args[7]
    phyml_bin_path = args[8]

    for family in families:
        st = time.time()
        input_msa_path = os.path.join(msa_dir, family + ".txt")
        phyml_log_path = os.path.join(output_tree_dir, family + ".phyml_log")
        with pushd(output_tree_dir):
            # I translate the MSA to the output directory because:
            # (1) phyml uses the phylip MSA format, and
            # (2) phyml generates a bunch of output files in the input
            # directory, thus we want the input directory to be the output
            # directory too.
            msa = read_msa(input_msa_path)
            input_msa_phylip_path = os.path.join(
                output_tree_dir, family + ".phylip"
            )
            write_msa_to_phylip_format(msa, input_msa_phylip_path)

            rate_matrix_paml_path = os.path.join(
                output_tree_dir, family + ".paml"
            )
            _to_paml_format(
                input_rate_matrix_path=rate_matrix_path,
                output_rate_matrix_path=rate_matrix_paml_path,
            )

            command = (
                f"{phyml_bin_path} "
                f"--input {input_msa_phylip_path} "
                f"--nclasses {num_rate_categories} "
                f"--model custom "
                f"--aa_rate_file {rate_matrix_paml_path} "
            )
            command += extra_command_line_args + " "
            command += f"> {phyml_log_path}"
            os.system(command)
        phyml_stats_path = os.path.join(
            output_tree_dir, family + ".phylip_phyml_stats.txt"
        )
        phyml_site_ll_path = os.path.join(
            output_tree_dir, family + ".phylip_phyml_lk.txt"
        )
        phyml_tree_path = os.path.join(
            output_tree_dir, family + ".phylip_phyml_tree.txt"
        )
        if (
            not os.path.exists(phyml_stats_path)
            or not os.path.exists(phyml_site_ll_path)
            or not os.path.exists(phyml_tree_path)
        ):
            raise Exception(
                f"PhyML failed to run. Files:\n{phyml_stats_path}\nAnd"
                f"\n{phyml_site_ll_path}\nAnd\n{phyml_tree_path}\n"
                f"do not all exist.\nCommand:\n{command}\n"
            )

        # Translate the tree
        tree_ete = TreeETE(phyml_tree_path)
        name_internal_nodes(tree_ete)
        tree_ete.write(
            format=3,
            outfile=os.path.join(output_tree_dir, family + ".newick"),
        )
        open(os.path.join(output_tree_dir, family + ".command"), "w").write(
            command
        )
        tree = translate_tree(tree_ete)
        write_tree(tree, os.path.join(output_tree_dir, family + ".txt"))
        secure_parallel_output(output_tree_dir, family)

        site_rates, site_ll = get_site_rates_and_site_ll(
            phyml_site_ll_path=phyml_site_ll_path
        )
        ll = get_ll(phyml_stats_path=phyml_stats_path)

        ll_file_contents = (
            f"{ll}\n{len(site_ll)} sites\n{' '.join(map(str, site_ll))}\n"
        )
        open(os.path.join(output_likelihood_dir, family + ".txt"), "w").write(
            ll_file_contents
        )
        secure_parallel_output(output_likelihood_dir, family)

        site_rates_file_contents = (
            f"{len(site_rates)} sites\n{' '.join(map(str, site_rates))}\n"
        )
        open(os.path.join(output_site_rates_dir, family + ".txt"), "w").write(
            site_rates_file_contents
        )
        secure_parallel_output(output_site_rates_dir, family)

        open(os.path.join(output_tree_dir, family + ".profiling"), "w").write(
            f"Total time: {time.time() - st}\n"
        )


def get_phyml_default_extra_command_line_args() -> str:
    return (
        "--datatype aa --pinv e --r_seed 0 --bootstrap 0 -f m "
        "--alpha e --print_site_lnl"
    )


@cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=[
        "output_tree_dir",
        "output_site_rates_dir",
        "output_likelihood_dir",
    ],
    write_extra_log_files=True,
)
def phyml(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_rate_categories: int,
    num_processes: int,
    extra_command_line_args: str = get_phyml_default_extra_command_line_args(),
    output_tree_dir: Optional[str] = None,
    output_site_rates_dir: Optional[str] = None,
    output_likelihood_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)

    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
    if not os.path.exists(output_site_rates_dir):
        os.makedirs(output_site_rates_dir)
    if not os.path.exists(output_likelihood_dir):
        os.makedirs(output_likelihood_dir)

    phyml_bin_path = _install_phyml()

    msa_dir = os.path.abspath(msa_dir)
    rate_matrix_path = os.path.abspath(rate_matrix_path)
    output_tree_dir = os.path.abspath(output_tree_dir)
    output_site_rates_dir = os.path.abspath(output_site_rates_dir)
    output_likelihood_dir = os.path.abspath(output_likelihood_dir)

    map_args = [
        [
            msa_dir,
            get_process_args(process_rank, num_processes, families),
            rate_matrix_path,
            num_rate_categories,
            output_tree_dir,
            output_site_rates_dir,
            output_likelihood_dir,
            extra_command_line_args,
            phyml_bin_path,
        ]
        for process_rank in range(num_processes)
    ]

    logger.info(
        f"Going to run on {len(families)} families using {num_processes} "
        "processes"
    )

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
