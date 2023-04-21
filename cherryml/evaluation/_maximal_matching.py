import logging
import multiprocessing
import os
import sys
from typing import List, Optional

import networkx as nx
import numpy as np
import tqdm

from cherryml.caching import cached_parallel_computation, secure_parallel_output
from cherryml.io import read_contact_map, write_contact_map
from cherryml.utils import get_process_args


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


init_logger()
logger = logging.getLogger(__name__)


@cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=["o_contact_map_dir"],
    write_extra_log_files=True,
)
def create_maximal_matching_contact_map(
    i_contact_map_dir: str,
    families: List[str],
    minimum_distance_for_nontrivial_contact: int,
    num_processes: int,
    o_contact_map_dir: Optional[str] = None,
) -> None:
    map_args = [
        [
            i_contact_map_dir,
            get_process_args(process_rank, num_processes, families),
            minimum_distance_for_nontrivial_contact,
            o_contact_map_dir,
        ]
        for process_rank in range(num_processes)
    ]

    logger.info(
        f"Going to run on {len(families)} families using {num_processes} "
        "processes"
    )

    # Map step (distribute families among processes)
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))

    logger.info("Done!")


def _map_func(args: List):
    i_contact_map_dir = args[0]
    families = args[1]
    minimum_distance_for_nontrivial_contact = args[2]
    o_contact_map_dir = args[3]
    for family in families:
        topology = nx.Graph()
        i_contact_map_path = os.path.join(i_contact_map_dir, family + ".txt")
        contact_map = read_contact_map(contact_map_path=i_contact_map_path)
        num_sites = contact_map.shape[0]
        topology.add_nodes_from([i for i in range(num_sites)])
        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if i < j and abs(i - j) >= minimum_distance_for_nontrivial_contact
        ]
        topology.add_edges_from(contacting_pairs)
        match = nx.maximal_matching(topology)
        res = np.zeros(shape=contact_map.shape)
        for u, v in match:
            res[u, v] = res[v, u] = 1
        o_contact_map_path = os.path.join(o_contact_map_dir, family + ".txt")
        write_contact_map(res, o_contact_map_path)
        secure_parallel_output(o_contact_map_dir, family)
