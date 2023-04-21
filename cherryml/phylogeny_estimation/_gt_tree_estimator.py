import logging
import multiprocessing
import os
import sys
from typing import List, Optional

import tqdm

from cherryml.caching import cached_parallel_computation, secure_parallel_output
from cherryml.io import (
    read_log_likelihood,
    read_site_rates,
    read_tree,
    write_log_likelihood,
    write_site_rates,
    write_tree,
)
from cherryml.utils import get_process_args


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _map_func(args: List):
    gt_tree_dir = args[0]
    gt_site_rates_dir = args[1]
    gt_likelihood_dir = args[2]
    families = args[3]
    output_tree_dir = args[4]
    output_site_rates_dir = args[5]
    output_likelihood_dir = args[6]

    for family in families:
        gt_tree = read_tree(os.path.join(gt_tree_dir, family + ".txt"))
        write_tree(gt_tree, os.path.join(output_tree_dir, family + ".txt"))
        secure_parallel_output(output_tree_dir, family)

        gt_site_rates = read_site_rates(
            os.path.join(gt_site_rates_dir, family + ".txt")
        )
        write_site_rates(
            gt_site_rates, os.path.join(output_site_rates_dir, family + ".txt")
        )
        secure_parallel_output(output_site_rates_dir, family)

        gt_likelihood = read_log_likelihood(
            os.path.join(gt_likelihood_dir, family + ".txt")
        )
        write_log_likelihood(
            gt_likelihood, os.path.join(output_likelihood_dir, family + ".txt")
        )
        secure_parallel_output(output_likelihood_dir, family)

        open(os.path.join(output_tree_dir, family + ".profiling"), "w").write(
            f"time_gt_tree_estimator: {0}"
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
def gt_tree_estimator(
    gt_tree_dir: str,
    gt_site_rates_dir: str,
    gt_likelihood_dir: str,
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_rate_categories: int,
    num_processes: int,
    output_tree_dir: Optional[str] = None,
    output_site_rates_dir: Optional[str] = None,
    output_likelihood_dir: Optional[str] = None,
) -> None:
    """
    Return the ground truth tree.
    """
    logger = logging.getLogger(__name__)

    map_args = [
        [
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
            get_process_args(process_rank, num_processes, families),
            output_tree_dir,
            output_site_rates_dir,
            output_likelihood_dir,
        ]
        for process_rank in range(num_processes)
    ]

    logger.info(
        f"Going to run on {len(families)} families using {num_processes} "
        "processes."
    )

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
