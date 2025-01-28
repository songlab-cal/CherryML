from typing import List, Optional
import logging
import sys
import tempfile
import os
from ete3 import Tree as TreeETE
import shutil
from ._common import name_internal_nodes, translate_tree
from cherryml.io import read_rate_matrix, write_tree, read_msa, \
    read_computed_cherries_from_file
from cherryml.caching import cached_parallel_computation, secure_parallel_output
import pandas as pd
from cherryml import markov_chain
import tqdm
from cherryml.utils import pushd
import time
import random
import multiprocessing
from cherryml.config import Config
import numpy as np
from cherryml.utils import quantization_idx
def write_file_paths_to_list(file_paths, output_file):
    with open(output_file, 'w') as file_list:
        file_list.write(str(len(file_paths)) + '\n')
        file_list.write('\n'.join(file_paths))

def _make_fast_cherries_and_return_bin_path(remake=False) -> str:
    """
    makes the binary and returns it.
    """
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "FastCherries"
    )
    bin_path = os.path.join(dir_path, "build/fast_cherries")
    if remake or not os.path.exists(bin_path):
        with pushd(dir_path):
            os.system("make clean && make main")
            if not os.path.exists(bin_path):
                raise Exception("Was not able to build FastCherries")
    return bin_path

def extract_log_likelihood(
    family: str,
    o_likelihood_dir: str,
) -> None:
    # hardcoded log likelihood to conform to phylogeny estimator pattern
    file = open(os.path.join(o_likelihood_dir, family + ".txt"), "w+")
    file.write(str(0.0))
    file.close()

    
def _map_func(args: List):
    families = args[0]                    
    msa_paths = args[1]
    fast_cherries_output_paths = args[2]
    output_tree_dir = args[3]
    output_site_rates_dir = args[4]
    output_likelihood_dir = args[5]
    quantization_grid_center = args[6]
    quantization_grid_step = args[7]
    quantization_grid_num_steps = args[8]
    rate_matrix_path = args[9]
    alphabet_path = args[10]
    verbose = args[11]
    bin_path = args[12]
    seed = args[13]
    num_rate_categories = args[14]
    max_iters = args[15]

    st = time.time()
    if not families:
        return
    with tempfile.NamedTemporaryFile("w") as msa_paths_file:
        with tempfile.NamedTemporaryFile("w") as output_paths_file:
            with tempfile.NamedTemporaryFile("w") as fast_cherries_output_paths_file:
                with tempfile.NamedTemporaryFile("w") as site_rate_paths_file:
                    profiling_paths = [os.path.join(output_tree_dir, family + ".profiling") for family in families]
                    site_rate_paths = [os.path.join(output_site_rates_dir, family + ".txt") for family in families]
                    write_file_paths_to_list(msa_paths, msa_paths_file.name)
                    write_file_paths_to_list(profiling_paths, output_paths_file.name)
                    write_file_paths_to_list(site_rate_paths, site_rate_paths_file.name)
                    write_file_paths_to_list(fast_cherries_output_paths, fast_cherries_output_paths_file.name)
                    if verbose:
                        logger = logging.getLogger(__name__)
                        logger.info(f"beginning c++ fast cherries")
                    command = bin_path + \
                        " -seed " + str(seed) + \
                        " -quantization_grid_center " + str(quantization_grid_center) + \
                        " -quantization_grid_step " + str(quantization_grid_step) + \
                        " -quantization_grid_num_steps " + str(quantization_grid_num_steps) + \
                        " -output_list_path " + fast_cherries_output_paths_file.name + \
                        " -rate_matrix_path " + rate_matrix_path + \
                        " -msa_list_path " + msa_paths_file.name + \
                        " -profiling_list_path " + output_paths_file.name + \
                        " -site_rate_list_path " + site_rate_paths_file.name + \
                        " -num_rate_categories_ble " + str(num_rate_categories) + \
                        " -max_iters_ble " + str(max_iters) + \
                        " -alphabet_path " + alphabet_path
                    if verbose:
                        logger.info(command)
                    os.system(command)
                    if verbose:
                        logger.info(f"c++ fast cherries finished in {time.time()-st} seconds")

                    et = time.time()
                    python_time = (et - st)

                    cpp_times = []
                    # post processing, get the output of c++ program and reformat it
                    for family, msa_path, fast_cherries_output_path in zip(
                        families, msa_paths, fast_cherries_output_paths
                    ):
                        cherries, distances = read_computed_cherries_from_file(fast_cherries_output_path)
                        msa = read_msa(msa_path)

                        all_seq = set(msa.keys())
                        in_cherries = set([seq for cherry in cherries for seq in cherry])
                        missing = list(all_seq.difference(in_cherries))

                        # create the tree
                        tree_ete = TreeETE(name="root")
                        for i in range(len(cherries)):
                            child = tree_ete.add_child(name="internal-"+str(i))
                            child.add_child(name=cherries[i][0])
                            child.add_child(name=cherries[i][1])
                            child.children[0].dist = distances[i]/2.0
                            child.children[1].dist = distances[i]/2.0
                        if len(missing) & 1 == 1:
                            tree_ete.add_child(name=missing[-1])

                        tree_ete.write(
                            format=3,
                            outfile=os.path.join(output_tree_dir, family + ".newick"),
                        )
                        tree = translate_tree(tree_ete)
                        write_tree(tree, os.path.join(output_tree_dir, family + ".txt"))

                        extract_log_likelihood(
                            family=family,
                            o_likelihood_dir=output_likelihood_dir
                        )
                        with open(
                            os.path.join(output_tree_dir, family + ".profiling"), "r"
                        ) as f:
                            lines = f.readlines()
                            cpp_times.append(float(lines[2].split()[1]))
                        
                    total_cpp_time = sum(cpp_times)
                    amortized = (python_time - total_cpp_time)/len(families)
                    for i,family in enumerate(families):
                        amortized_cpp_time = amortized + cpp_times[i]
                        with open(
                            os.path.join(output_tree_dir, family + ".profiling"), "a"
                        ) as f:
                            f.write(f"total_time: {amortized_cpp_time}")


def get_process_args(
    process_rank: int, num_processes: int, all_args: List
) -> List:
    process_args = [
        all_args[i]
        for i in range(len(all_args))
        if i % num_processes == process_rank
    ]
    return process_args

def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

_init_logger()

@cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    exclude_args_if_default=["_version"],
    output_dirs=[
        "output_tree_dir",
        "output_site_rates_dir",
        "output_likelihood_dir",
    ],
    write_extra_log_files=True,
)
def fast_cherries(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_rate_categories: int,
    max_iters: int,
    num_processes: int,
    _version = "2",
    output_tree_dir: Optional[str] = None,
    output_site_rates_dir: Optional[str] = None,
    output_likelihood_dir: Optional[str] = None,
    remake=False,
    quantization_grid_center = 0.03,
    quantization_grid_step = 1.1,
    quantization_grid_num_steps = 64,
    verbose = True,
    seed = 1234,
) -> None:
    logger = logging.getLogger(__name__)
    if verbose:
        logger.info(
            f"Going to run on {len(families)} families using {num_processes} "
            "processes"
        )
        logger.info(output_tree_dir)
    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
    if not os.path.exists(output_site_rates_dir):
        os.makedirs(output_site_rates_dir)
    if not os.path.exists(output_likelihood_dir):
        os.makedirs(output_likelihood_dir)
    
    rate_matrix = read_rate_matrix(rate_matrix_path)

    bin_path = _make_fast_cherries_and_return_bin_path(remake)

    alphabet = list(rate_matrix.columns)
    with tempfile.NamedTemporaryFile("w") as rate_matrix_file:
        with tempfile.TemporaryDirectory() as fast_cherries_output_dir:
            with tempfile.NamedTemporaryFile("w") as alphabet_file:
                quantization_points = [
                    (quantization_grid_center * quantization_grid_step**i)
                    for i in range(
                        -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
                    )
                ]
                msa_paths = [os.path.join(msa_dir, family + ".txt") for family in families]
                profiling_paths = [os.path.join(output_tree_dir, family + ".profiling") for family in families]
                fast_cherries_output_paths = [os.path.join(fast_cherries_output_dir, family + ".output") for family in families]
                with open(alphabet_file.name, "w") as f:
                    f.write(str(len(alphabet)) + " " + " ".join(alphabet))
                        
                with open(rate_matrix_path, 'r') as source_file:
                    lines = source_file.readlines()
                    modified_lines = [line[1:] for line in lines[1:]]
                
                with open(rate_matrix_file.name, 'w') as destination_file:
                    destination_file.writelines(modified_lines)

                map_args = [
                    [
                        get_process_args(process_rank, num_processes, families),
                        get_process_args(process_rank, num_processes, msa_paths),
                        get_process_args(process_rank, num_processes, fast_cherries_output_paths),
                        output_tree_dir,
                        output_site_rates_dir,
                        output_likelihood_dir,
                        quantization_grid_center,
                        quantization_grid_step,
                        quantization_grid_num_steps,
                        rate_matrix_file.name,
                        alphabet_file.name,
                        verbose,
                        bin_path,
                        seed,
                        num_rate_categories,
                        max_iters 
                    ]
                    for process_rank in range(num_processes)
                ]

                if num_processes > 1:
                    with multiprocessing.Pool(num_processes) as pool:
                        list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
                else:
                    if verbose:
                        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
                    else:
                        list(map(_map_func, map_args))
    if verbose:
        logger.info("Done!")    
