import logging
import multiprocessing
import os
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np
import tqdm
from ete3 import Tree as TreeETE

from cherryml.caching import cached_parallel_computation, secure_parallel_output
from cherryml.io import (
    get_msa_num_sites,
    read_rate_matrix,
    write_float,
    write_site_rates,
    write_tree,
)
from cherryml.markov_chain import compute_stationary_distribution
from cherryml.utils import get_process_args

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


def _iq_tree_is_installed_on_system() -> bool:
    """
    Check whether `iqtree` program is installed on the system.
    """
    res = os.popen("which iqtree").read()
    if len(res) > 0:
        # is installed
        return True
    else:
        # is not installed
        return False


def convert_rate_matrix_to_paml_format(
    rate_matrix: np.array,
    output_paml_path: str,
) -> None:
    """
    Convert a rate matrix into the PAML format

    The IQTree format consists of the exchangeabilities, followed by the
    stationary distribution.
    """
    pi = compute_stationary_distribution(rate_matrix)
    exchangeabilities = np.zeros(shape=rate_matrix.shape)
    n = rate_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            exchangeabilities[i, j] = rate_matrix[i, j] / pi[j]
    for i in range(n):
        for j in range(n):
            delta = exchangeabilities[i, j] - exchangeabilities[j, i]
            if abs(delta) > 1e-3:
                raise ValueError(
                    "Rate matrix does not appear to be reversible. Difference "
                    "in exchangeability coefficients (i,j) and (j,i) is: "
                    f"{abs(delta)}"
                )
    res = ""
    for i in range(1, n):
        res += " ".join([str(exchangeabilities[i, j]) for j in range(i)])
        res += "\n"
    res += "\n"
    res += " ".join([str(pi[i]) for i in range(n)])
    res += "\n"
    with open(output_paml_path, "w") as output_file:
        output_file.write(res)


def translate_site_rates(
    iq_tree_output_dir: str,
    family: str,
    rate_category_selector: str,
    o_site_rates_dir: str,
    num_sites: int,
) -> None:
    """
    Translate IQTree site rates to our format
    """
    file_contents = open(
        os.path.join(iq_tree_output_dir, "output.rate"), "r"
    ).read()
    lines = file_contents.split("\n")
    lines = lines[9:]
    assert lines[-1] == ""
    lines = lines[:-1]
    rates = []
    for (j, line) in enumerate(lines):
        site_str, posterior_mean_rate_str, cat_str, map_rate_str = line.split()
        site = int(site_str)
        posterior_mean_rate = float(posterior_mean_rate_str)
        cat = int(cat_str)
        del cat
        map_rate = float(map_rate_str)
        if site != j + 1:
            raise Exception(
                f"Error parsing IQTree site rate file. Line: '{line}'"
                f" should have started with line number '{j + 1}'"
            )
        if rate_category_selector == "MAP":
            rates.append(map_rate)
        elif rate_category_selector == "posterior_mean":
            rates.append(posterior_mean_rate)
        else:
            raise ValueError(
                f"Unknown rate_category_selector: '{rate_category_selector}'."
                f" Allowed values: 'MAP', 'posterior_mean'."
            )
    if len(rates) != num_sites:
        # Means this was a model without site rate variation (e.g. JTT instead
        # of JTT+G4), so rates should be all 1s.
        rates = [1.0] * num_sites
    write_site_rates(
        rates,
        os.path.join(o_site_rates_dir, family + ".txt"),
    )
    secure_parallel_output(o_site_rates_dir, family)


def extract_log_likelihood(
    iq_tree_output_dir: str,
    family: str,
    o_likelihood_dir: str,
) -> None:
    """
    Use the .iqtree file to parse out the LL and the AIC, cAIC, BIC
    (write them to [family].bic, etc. Only write the ll to family.result,
    to conform to the API.)
    """
    lines = (
        open(os.path.join(iq_tree_output_dir, "output.iqtree"), "r")
        .read()
        .split("\n")
    )
    for line in lines:
        if line.startswith("Log-likelihood of the tree:"):
            ll = float(line.split(" ")[4])
            write_float(
                val=ll,
                float_path=os.path.join(o_likelihood_dir, family + ".txt"),
            )
            write_float(
                val=ll,
                float_path=os.path.join(o_likelihood_dir, family + ".ll"),
            )
        elif line.startswith("Unconstrained log-likelihood (without tree):"):
            ull = float(line.split(" ")[-1])
            write_float(
                val=ull,
                float_path=os.path.join(o_likelihood_dir, family + ".ull"),
            )
        # elif line.startswith("Number of free parameters (#branches + #model parameters):"):
        #     fp = float(line.split(' ')[-1])
        #     write_float(
        #         val=fp,
        #         float_path=os.path.join(o_likelihood_dir, family + ".fp")
        #     )
        elif line.startswith("Akaike information criterion (AIC) score:"):
            aic = float(line.split(" ")[-1])
            write_float(
                val=aic,
                float_path=os.path.join(o_likelihood_dir, family + ".aic"),
            )
        elif line.startswith(
            "Corrected Akaike information criterion (AICc) score:"
        ):
            aicc = float(line.split(" ")[-1])
            write_float(
                val=aicc,
                float_path=os.path.join(o_likelihood_dir, family + ".aicc"),
            )
        elif line.startswith("Bayesian information criterion (BIC) score:"):
            bic = float(line.split(" ")[-1])
            write_float(
                val=bic,
                float_path=os.path.join(o_likelihood_dir, family + ".bic"),
            )
        # elif line.startswith("Total tree length (sum of branch lengths):"):
        #     tree_len = float(line.split(' ')[-1])
        #     write_float(
        #         val=tree_len,
        #         float_path=os.path.join(o_likelihood_dir, family + ".tree_len")
        #     )
        # elif line.startswith("Sum of internal branch lengths:"):
        #     internal_bl = float(line.split(' ')[-1])
        #     write_float(
        #         val=internal_bl,
        #         float_path=os.path.join(o_likelihood_dir, family + ".internal_bl")
        #     )

    secure_parallel_output(o_likelihood_dir, family)


def extract_model_of_substitution(
    iq_tree_output_dir: str,
    family: str,
    o_likelihood_dir: str,
    scaled_rate_matrices_filenames: List[str],
    rate_matrices_paths: List[str],
) -> None:
    lines = (
        open(os.path.join(iq_tree_output_dir, "output.iqtree"), "r")
        .read()
        .split("\n")
    )
    for line in lines:
        if line.startswith("Model of substitution:"):
            scaled_rate_matrix_path_w_rate_model = line.split(" ")[3]
            scaled_rate_matrix_path = (
                scaled_rate_matrix_path_w_rate_model.split("+")[0]
            )
            rate_model = scaled_rate_matrix_path_w_rate_model[
                len(scaled_rate_matrix_path) :
            ]
            if scaled_rate_matrix_path not in scaled_rate_matrices_filenames:
                raise Exception(
                    f"Selected model is {scaled_rate_matrix_path} but is not "
                    "present in scaled_rate_matrices_filenames="
                    f"{scaled_rate_matrices_filenames}"
                )
            rate_matrix_path = None
            for (x, y) in zip(
                rate_matrices_paths, scaled_rate_matrices_filenames
            ):
                if y == scaled_rate_matrix_path:
                    rate_matrix_path = x
                    break
            assert rate_matrix_path is not None
            break
    assert rate_matrix_path is not None
    with open(
        os.path.join(o_likelihood_dir, family + ".model"), "w"
    ) as output_file:
        output_file.write(rate_matrix_path)
    with open(
        os.path.join(o_likelihood_dir, family + ".model_full"), "w"
    ) as output_file:
        output_file.write(rate_matrix_path + rate_model)


def run_iq_tree(
    msa_path: str,
    family: str,
    rate_matrix_path: str,
    rate_model: str,
    num_rate_categories: int,
    use_model_finder: bool,
    output_tree_dir: str,
    output_site_rates_dir: str,
    output_likelihood_dir: str,
    extra_command_line_args: str,
    rate_category_selector: str,
    random_seed: int,
    iq_tree_bin: str,
) -> str:
    r"""
    This wrapper deals with the fact that IQTree normalizes input rate
    matrices. Therefore, to run IQTree with an arbitrary rate matrix,
    we first normalize it. After inference with IQTree, we have to
    'de-normalize' the branch lengths to put them in the same time units as the
    original rate matrix.
    """
    rate_matrices_paths = rate_matrix_path.split(",")
    del rate_matrix_path
    if len(rate_matrices_paths) > 1 and not use_model_finder:
        raise ValueError(
            "Trying to run IQTree with more than one rate matrix. This is only "
            "allowed when using ModelFinder, i.e. use_model_finder = True. You "
            f"provided use_model_finder = '{use_model_finder}'"
        )
    with tempfile.TemporaryDirectory() as iq_tree_output_dir:
        with tempfile.TemporaryDirectory() as scaled_rate_matrices_dir:
            scaled_rate_matrices_filenames = [
                os.path.join(scaled_rate_matrices_dir, f"rate_matrix_{i}.txt")
                for i in range(len(rate_matrices_paths))
            ]
            for rate_matrix_path, scaled_rate_matrix_filename in zip(
                rate_matrices_paths, scaled_rate_matrices_filenames
            ):
                Q_df = read_rate_matrix(rate_matrix_path)
                if not (Q_df.shape == (20, 20)):
                    raise ValueError(
                        f"The rate matrix {rate_matrix_path} does not have "
                        "dimension 20 x 20."
                    )
                Q = np.array(Q_df)
                pi = compute_stationary_distribution(Q)
                # Check that rows (originally columns) of Q add to 0
                if not np.sum(np.abs(Q.sum(axis=1))) < 0.01:
                    raise ValueError(
                        f"Custom rate matrix {rate_matrix_path} doesn't have "
                        "columns that add up to 0."
                    )
                # Check that the stationary distro is correct
                if not np.sum(np.abs(pi @ Q)) < 0.01:
                    raise ValueError(
                        f"Custom rate matrix {rate_matrix_path} doesn't have "
                        "the stationary distribution."
                    )
                # Compute the mutation rate.
                mutation_rate = pi @ -np.diag(Q)
                # Normalize Q
                Q_normalized = Q / mutation_rate
                # Write out Q_normalized in PAML format, for use in IQTree
                convert_rate_matrix_to_paml_format(
                    Q_normalized,
                    output_paml_path=scaled_rate_matrix_filename,
                )
            # Run IQTree!
            command = f"{iq_tree_bin}"
            command += f" -s {msa_path}"
            if use_model_finder:
                # ModelFinder will search for the optimal site rate model.
                command += (
                    f" -m MF"
                    f" -mset {','.join(scaled_rate_matrices_filenames)}"
                )
            else:
                # Just standard IQTree with a fixed site rate model.
                command += (
                    f" -m {scaled_rate_matrix_filename}"
                    f"+{rate_model}{num_rate_categories}"  # e.g. [...]+R4
                )
            command += (
                f" -seed {random_seed}"
                f" -redo"
                f" -pre {iq_tree_output_dir}/output"  # (output prefix)
                f" -st AA"
                f" -wsr"  # (write site rates to *.rate file)
                f" -nt 1"  # (1 thread)
                f" -quiet"  # silent!
                f" {extra_command_line_args}"
            )
            st = time.time()
            os.system(command)
            et = time.time()
            open(
                os.path.join(output_tree_dir, family + ".profiling"), "w"
            ).write(f"time_iq_tree: {et - st}")
            # De-normalize the branch lengths of the tree
            tree_ete = TreeETE(
                os.path.join(f"{iq_tree_output_dir}/output.treefile")
            )

            def dfs_scale_tree(v) -> None:
                for u in v.get_children():
                    u.dist = u.dist / mutation_rate
                    dfs_scale_tree(u)

            dfs_scale_tree(tree_ete)

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

            num_sites = get_msa_num_sites(msa_path=msa_path)
            translate_site_rates(
                iq_tree_output_dir=iq_tree_output_dir,
                family=family,
                rate_category_selector=rate_category_selector,
                o_site_rates_dir=output_site_rates_dir,
                num_sites=num_sites,
            )

            extract_log_likelihood(
                iq_tree_output_dir=iq_tree_output_dir,
                family=family,
                o_likelihood_dir=output_likelihood_dir,
            )

            extract_model_of_substitution(
                iq_tree_output_dir=iq_tree_output_dir,
                family=family,
                o_likelihood_dir=output_likelihood_dir,
                scaled_rate_matrices_filenames=scaled_rate_matrices_filenames,
                rate_matrices_paths=rate_matrices_paths,
            )


def _map_func(args: List):
    msa_dir = args[0]
    families = args[1]
    rate_matrix_path = args[2]
    rate_model = args[3]
    num_rate_categories = args[4]
    output_tree_dir = args[5]
    output_site_rates_dir = args[6]
    output_likelihood_dir = args[7]
    extra_command_line_args = args[8]
    rate_category_selector = args[9]
    use_model_finder = args[10]
    random_seed = args[11]
    iq_tree_bin = args[12]

    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        run_iq_tree(
            msa_path=msa_path,
            family=family,
            rate_matrix_path=rate_matrix_path,
            rate_model=rate_model,
            num_rate_categories=num_rate_categories,
            output_tree_dir=output_tree_dir,
            output_site_rates_dir=output_site_rates_dir,
            output_likelihood_dir=output_likelihood_dir,
            extra_command_line_args=extra_command_line_args,
            rate_category_selector=rate_category_selector,
            use_model_finder=use_model_finder,
            random_seed=random_seed,
            iq_tree_bin=iq_tree_bin,
        )


@cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=[
        "output_tree_dir",
        "output_site_rates_dir",
        "output_likelihood_dir",
    ],
)
def iq_tree(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    rate_model: Optional[str],  # "G" or "R"
    num_rate_categories: Optional[int],
    num_processes: int,
    extra_command_line_args: str = "",
    rate_category_selector: str = "MAP",  # "MAP" or "posterior_mean"
    use_model_finder: bool = "False",
    random_seed: int = 1,
    output_tree_dir: Optional[str] = None,
    output_site_rates_dir: Optional[str] = None,
    output_likelihood_dir: Optional[str] = None,
) -> None:
    """
    Run IQTree to find the best model given a rate matrix and infer the
    corresponding tree.
    """
    if use_model_finder:
        if num_rate_categories is not None:
            raise ValueError(
                "You are attempting to run IQTree with ModelFinder. "
                "You cannot specify the number of rate categories "
                "since it will be searched over"
            )
        if rate_model is not None:
            raise ValueError(
                "You are attempting to run IQTree with a rate_model. You cannot"
                " specify the rate_model since it will be searched over"
            )
    logger = logging.getLogger(__name__)
    logger.info(
        f"Going to run on {len(families)} families using {num_processes} "
        "processes"
    )

    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
    if not os.path.exists(output_site_rates_dir):
        os.makedirs(output_site_rates_dir)
    if not os.path.exists(output_likelihood_dir):
        os.makedirs(output_likelihood_dir)

    iq_tree_bin = "iqtree"

    if not _iq_tree_is_installed_on_system():
        raise Exception(
            "iqtree is not installed on system (or is not on PATH)."
        )

    map_args = [
        [
            msa_dir,
            get_process_args(process_rank, num_processes, families),
            rate_matrix_path,
            rate_model,
            num_rate_categories,
            output_tree_dir,
            output_site_rates_dir,
            output_likelihood_dir,
            extra_command_line_args,
            rate_category_selector,
            use_model_finder,
            random_seed,
            iq_tree_bin,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))

    logger.info("Done!")
