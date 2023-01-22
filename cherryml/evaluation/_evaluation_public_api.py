"""
Public API for log-likelihood evaluation, useful for model selection.
"""
import logging
import os
import sys
import tempfile
from functools import partial
from typing import List, Optional

from cherryml import caching, utils
from cherryml.io import read_log_likelihood, read_site_rates
from cherryml.phylogeny_estimation import fast_tree, phyml


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()
logger = logging.getLogger(__name__)


def evaluation_public_api(
    output_path: str,
    rate_matrix_path: str,
    msa_dir: str,
    cache_dir: Optional[str] = None,
    num_processes_tree_estimation: int = 4,
    num_rate_categories: int = 20,
    families: Optional[List[str]] = None,
    tree_estimator_name: str = "FastTree",
    extra_command_line_args: Optional[str] = None,
) -> str:
    """
    Log-likelihood evaluation.

    This minimalist API should suit most use cases. The log-likelihood will be
    written to `output_path`.

    The directory `msa_dir` should contain one file per family, named
    `[family_name].txt`, where `family_name` is the name of the family.

    The files in `msa_dir` should list the protein sequences in each family
    following the format in the following example:
    ```
    >seq1
    TTLLS
    >seq2
    TTIIS
    >seq3
    SSIIS
    ```
    All sequences should have the same length.

    Log-likelihood is computed by running the tree estimator given by
    `tree_estimator_name`. Tree estimation is parallelized using the number
    of processes given by `num_processes_tree_estimation`. The number of
    rate categories used is `num_rate_categories`.

    To compute log-likelihood on only a subset of the alignments in `msa_dir`,
    provide the list of families with the `families` argument (this is useful
    for having training and testing alignments in the same directory, without
    explicit separation in disk).

    Extra command line arguments can be provided to the tree estimator via
    `extra_command_line_args`, for example, `-gamma` to run FastTree with the
    Gamma model rather than MAP rate categories (this produces lower
    likelihoods, since it accounts for the possibility that the rate could have
    been something other than the MAP category; see FastTree documentation
    online for more information).

    Args:
        See description of arguments in https://github.com/songlab-cal/CherryML
    """
    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory()
        logger.info(
            "Cache directory not provided. Will use temporary directory "
            f"{cache_dir} to cache computations."
        )

    caching.set_cache_dir(cache_dir)

    if families is not None:
        for family in families:
            if not os.path.exists(os.path.join(msa_dir, family + ".txt")):
                raise ValueError(
                    f"MSA for family {family} not found in {msa_dir}. "
                    f"Was expecting file {msa_dir}.txt to contain the MSA, "
                    "but it does not exist."
                )

    if families is None:
        families = utils.get_families(msa_dir)

    if tree_estimator_name == "FastTree":
        tree_estimator = fast_tree
    elif tree_estimator_name == "PhyML":
        tree_estimator = phyml
    else:
        raise ValueError(
            f"Unknown tree_estimator_name: {tree_estimator_name}."
            " Available tree estimators: 'FastTree', 'PhyML'."
        )
    tree_estimator = partial(
        tree_estimator,
        num_rate_categories=num_rate_categories,
    )
    if extra_command_line_args is not None:
        tree_estimator = partial(
            tree_estimator,
            extra_command_line_args=extra_command_line_args,
        )

    tree_estimator_output_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_processes=num_processes_tree_estimation,
    )

    lls = []
    num_sites = []
    tot_ll = 0.0
    tot_num_sites = 0
    for family in families:
        ll_path = os.path.join(
            tree_estimator_output_dirs["output_likelihood_dir"], family + ".txt"
        )
        ll, _ = read_log_likelihood(ll_path)
        lls.append(ll)
        tot_ll += ll

        site_rates_path = os.path.join(
            tree_estimator_output_dirs["output_site_rates_dir"], family + ".txt"
        )
        site_rates = read_site_rates(site_rates_path)
        num_sites.append(len(site_rates))
        tot_num_sites += len(site_rates)

    with open(output_path, "w") as outfile:
        outfile.write(
            f"Total log-likelihood: {tot_ll}\n"
            f"Total number of sites: {tot_num_sites}\n"
            f"Average log-likelihood per site: {tot_ll/tot_num_sites}\n"
            f"Families: {' '.join(families)}\n"
            f"Log-likelihood per family: {' '.join(map(str, lls))}\n"
            f"Sites per family: {' '.join(map(str, num_sites))}\n"
        )
