import logging
import multiprocessing
import os
import sys
from typing import Dict, List, Optional

import tqdm

from cherryml import caching
from cherryml.counting import count_co_transitions, count_transitions
from cherryml.estimation import jtt_ipw, quantized_transitions_mle
from cherryml.evaluation import create_maximal_matching_contact_map
from cherryml.io import (
    read_msa,
    read_site_rates,
    read_sites_subset,
    write_msa,
    write_site_rates,
)
from cherryml.markov_chain import get_equ_path, get_equ_x_equ_path
from cherryml.types import PhylogenyEstimatorType
from cherryml.utils import get_amino_acids, get_process_args

CHERRYML_TYPE = "cherry++"


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _map_func_subset_data_to_sites_subset(args: List) -> None:
    sites_subset_dir = args[0]
    msa_dir = args[1]
    site_rates_dir = args[2]
    families = args[3]
    output_msa_dir = args[4]
    output_site_rates_dir = args[5]

    for family in families:
        sites_subset_path = os.path.join(sites_subset_dir, family + ".txt")
        sites_subset = read_sites_subset(sites_subset_path)
        # This exception below is not needed because downstream code (counting
        # transitions) has no issue with this border case.
        # if len(sites_subset) == 0:
        #     raise Exception(
        #         f"Family {family} has no nontrivial contacting sites. "
        #         "This would lead to an empty MSA."
        #     )

        input_msa_path = os.path.join(msa_dir, family + ".txt")
        msa = read_msa(input_msa_path)

        input_site_rates_path = os.path.join(site_rates_dir, family + ".txt")
        site_rates = read_site_rates(input_site_rates_path)

        new_msa = {
            sequence_name: "".join([sequence[site] for site in sites_subset])
            for (sequence_name, sequence) in msa.items()
        }
        write_msa(new_msa, os.path.join(output_msa_dir, family + ".txt"))

        new_site_rates = [site_rates[site] for site in sites_subset]
        write_site_rates(
            new_site_rates, os.path.join(output_site_rates_dir, family + ".txt")
        )

        caching.secure_parallel_output(output_msa_dir, family)
        caching.secure_parallel_output(output_site_rates_dir, family)


def is_pairer(tree_estimator_output_dirs: str, families: List[str]) -> float:
    """
    checks if the phylogeny estimator is a pairer by checking that the pairing time is in the profiling file
    """
    profiling_file_path = os.path.join(
        tree_estimator_output_dirs["output_tree_dir"],
        families[0] + ".profiling",
    )
    if not os.path.exists(profiling_file_path):
        return False
    with open(profiling_file_path, "r") as profiling_file:
        profiling_file_contents = profiling_file.read()
        return "pairing" in profiling_file_contents

@caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_msa_dir", "output_site_rates_dir"],
    write_extra_log_files=True,
)
def _subset_data_to_sites_subset(
    sites_subset_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    families: List[str],
    num_processes: int = 1,
    output_msa_dir: Optional[str] = None,
    output_site_rates_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info(
        f"Subsetting data to sites subset on {len(families)} families "
        f"using {num_processes} processes. output_msa_dir: {output_msa_dir} ; "
        f"output_site_rates_dir: {output_site_rates_dir}"
    )

    map_args = [
        [
            sites_subset_dir,
            msa_dir,
            site_rates_dir,
            get_process_args(process_rank, num_processes, families),
            output_msa_dir,
            output_site_rates_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_subset_data_to_sites_subset, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_subset_data_to_sites_subset, map_args),
                total=len(map_args),
            )
        )

    logger.info("Subsetting data to sites subset done!")


def _get_runtime_from_profiling_file(profiling_file_path: str) -> float:
    """
    TODO: Define a serialization protocol for profiling files and move to IO
    """
    with open(profiling_file_path, "r") as profiling_file:
        profiling_file_contents = profiling_file.read()
        res = float(profiling_file_contents.split()[2])
        return res

def _get_runtime_from_tree_estimator_profiling_file(
    profiling_file_path: str, attribute: str
) -> float:
    """
    TODO: Define a serialization protocol for profiling files and move to IO
    """
    if not os.path.exists(profiling_file_path):
        return 0  # No profiling information was provided (e.g. GT tree dir)
    index = -1
    if attribute == "pairing":
        index = 1
    elif attribute == "ble":
        index = 3
    with open(profiling_file_path, "r") as profiling_file:
        profiling_file_contents = profiling_file.read()
        res = float(profiling_file_contents.split()[index])
        return res

def _get_tree_estimation_runtime(
    tree_estimator_output_dirs: str, families: List[str], attribute: str=""
) -> float:
    """
    Total runtime used to estimate trees
    TODO: Define a serialization protocol for profiling files and move to IO
    """
    res = 0
    for family in families:
        profiling_file_path = os.path.join(
            tree_estimator_output_dirs["output_tree_dir"],
            family + ".profiling",
        )
        res += _get_runtime_from_tree_estimator_profiling_file(
            profiling_file_path, attribute
        )
    return res

def _get_all_site_rates(
    tree_estimator_output_dirs: str, families: List[str]
) -> List[List[float]]:
    """
    list of list of site rates.
    res[i][j] is the site rate of the j-th site for the i-th family
    """
    res = []
    for family in sorted(families):
        profiling_file_path = os.path.join(
            tree_estimator_output_dirs["output_site_rates_dir"],
            family + ".txt",
        )
        res.append(read_site_rates(profiling_file_path))
    return res

def lg_end_to_end_with_cherryml_optimizer(
    msa_dir: str,
    families: List[str],
    tree_estimator: PhylogenyEstimatorType,
    initial_tree_estimator_rate_matrix_path: str,
    num_iterations: Optional[int] = 1,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    optimizer_device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 2000,
    do_adam: bool = True,
    edge_or_cherry: str = CHERRYML_TYPE,
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    num_processes_tree_estimation: int = 8,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 2,
    optimizer_initialization: str = "jtt-ipw",
    sites_subset_dir: Optional[str] = None,
    tree_dir: Optional[str] = None,
    site_rates_dir: Optional[str] = None,
    alphabet: List[str] = get_amino_acids(),
) -> Dict:
    """
    LG pipeline with CherryML optimizer.

    Returns a dictionary with the directories to the intermediate outputs. In
    particular, the learned rate matrix is indexed by
    "learned_rate_matrix_path".

    One can train a model on only a subset of the sites by specifying
    sites_subset_dir. This is a file containing the indices of the sites used
    for training. Note that ALL the sites will the used when fitting the trees.

    You can provide the trees and site rates for the first round of estimation
    with tree_dir and site_rates_dir. Otherwise, the provided tree_estimator
    will be used to estimate trees and site rates.
    """
    if sites_subset_dir is not None and num_iterations > 1:
        raise ValueError(
            "You are using more than 1 iteration while learning a model only"
            "on a subset of sites. This is most certainly a usage error."
        )

    if (tree_dir is None and site_rates_dir is not None) or (
        tree_dir is not None and site_rates_dir is None
    ):
        raise ValueError(
            "tree_dir and site_rates_dir must be either both provided or none "
            f"provided. You provided: tree_dir={tree_dir} ; "
            f"site_rates_dir={site_rates_dir}"
        )

    res = {}

    quantization_points = [
        ("%.8f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    res["quantization_points"] = quantization_points

    time_tree_estimation = 0
    time_pairing = 0
    time_ble = 0
    time_counting = 0
    time_jtt_ipw = 0
    time_optimization = 0
    is_a_pairer = False
    
    current_estimate_rate_matrix_path = initial_tree_estimator_rate_matrix_path
    for iteration in range(num_iterations):
        if (
            iteration == 0
            and tree_dir is not None
            and site_rates_dir is not None
        ):
            tree_estimator_output_dirs = {
                "output_tree_dir": tree_dir,
                "output_site_rates_dir": site_rates_dir,
            }
        else:
            tree_estimator_output_dirs = tree_estimator(
                msa_dir=msa_dir,
                families=families,
                rate_matrix_path=current_estimate_rate_matrix_path,
                num_processes=num_processes_tree_estimation,
            )
        res[
            f"tree_estimator_output_dirs_{iteration}"
        ] = tree_estimator_output_dirs

        time_tree_estimation += _get_tree_estimation_runtime(
            tree_estimator_output_dirs, families, "total"
        )
        if is_a_pairer or is_pairer(tree_estimator_output_dirs=tree_estimator_output_dirs, families=families):
            is_a_pairer = True
            time_pairing += _get_tree_estimation_runtime(
                tree_estimator_output_dirs, families, "pairing"
            )
            time_ble += _get_tree_estimation_runtime(
                tree_estimator_output_dirs, families, "ble"
            )

        if sites_subset_dir is not None:
            res_dict = _subset_data_to_sites_subset(
                sites_subset_dir=sites_subset_dir,
                msa_dir=msa_dir,
                site_rates_dir=tree_estimator_output_dirs[
                    "output_site_rates_dir"
                ],
                families=families,
                num_processes=num_processes_counting,
            )
            msa_dir = res_dict["output_msa_dir"]
            tree_estimator_output_dirs["output_site_rates_dir"] = res_dict[
                "output_site_rates_dir"
            ]
            del res_dict

        count_matrices_dir = count_transitions(
            tree_dir=tree_estimator_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            site_rates_dir=tree_estimator_output_dirs["output_site_rates_dir"],
            families=families,
            amino_acids=alphabet[:],
            quantization_points=quantization_points,
            edge_or_cherry=edge_or_cherry,
            num_processes=num_processes_counting,
            use_cpp_implementation=use_cpp_counting_implementation,
            cpp_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_command_line_suffix=cpp_counting_command_line_suffix,
        )["output_count_matrices_dir"]

        res[f"count_matrices_dir_{iteration}"] = count_matrices_dir
        time_counting += _get_runtime_from_profiling_file(
            os.path.join(count_matrices_dir, "profiling.txt")
        )

        jtt_ipw_dir = jtt_ipw(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            mask_path=None,
            use_ipw=True,
            normalize=False,
        )["output_rate_matrix_dir"]

        res[f"jtt_ipw_dir_{iteration}"] = jtt_ipw_dir
        time_jtt_ipw += _get_runtime_from_profiling_file(
            os.path.join(jtt_ipw_dir, "profiling.txt")
        )

        initialization_path = None
        if optimizer_initialization == "jtt-ipw":
            initialization_path = os.path.join(jtt_ipw_dir, "result.txt")
        elif optimizer_initialization == "equ":
            initialization_path = get_equ_path()
        elif optimizer_initialization == "random":
            initialization_path = None
        else:
            raise ValueError(
                f"Unknown optimizer_initialization = {optimizer_initialization}"
            )

        rate_matrix_dir = quantized_transitions_mle(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            initialization_path=initialization_path,
            mask_path=None,
            stationary_distribution_path=None,
            rate_matrix_parameterization="pande_reversible",
            device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            OMP_NUM_THREADS=num_processes_optimization,
            OPENBLAS_NUM_THREADS=num_processes_optimization,
        )["output_rate_matrix_dir"]
        time_optimization += _get_runtime_from_profiling_file(
            os.path.join(rate_matrix_dir, "profiling.txt")
        )

        res[f"rate_matrix_dir_{iteration}"] = rate_matrix_dir

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )

    res["learned_rate_matrix_path"] = current_estimate_rate_matrix_path

    res["all_site_rates"] = _get_all_site_rates(tree_estimator_output_dirs=tree_estimator_output_dirs,
                                                families=families)
    res["time_tree_estimation"] = time_tree_estimation

    if is_a_pairer:
        res["time_pairing"] = time_pairing
        res["time_ble"] = time_ble

    res["time_counting"] = time_counting
    res["time_jtt_ipw"] = time_jtt_ipw
    res["time_optimization"] = time_optimization
    res["total_cpu_time"] = (
        time_tree_estimation + time_counting + time_jtt_ipw + time_optimization
    )


    profiling_str = (
        f"CherryML runtimes:\n"
        "time_tree_estimation (without parallelization): "
        f"{res['time_tree_estimation']}\n"
        f"time_counting: {res['time_counting']}\n"
        f"time_jtt_ipw: {res['time_jtt_ipw']}\n"
        f"time_optimization: {res['time_optimization']}\n"
        f"total_cpu_time: {res['total_cpu_time']}\n"
    )
    if is_a_pairer:
        profiling_str += (
            f"time_pairing {res['time_pairing']}\n"
            f"time_ble {res['time_ble']}"
        )
    res["profiling_str"] = profiling_str

    # write dfs function to pop cherries from the tree and compute distance of cherries, 
    # store a list of all the cherry distance in res, order doesn't matter, we only care 
    # about the distribution of branch legnths
    #res["cherry_distances"] = _get_all_cherry_distances(
    #    tree_estimator_output_dirs=tree_estimator_output_dirs,
    #    families=families
    #)
    #res["cherry_composite_likelihoods"] = _get_cherry_likelihoods(
    #    tree_estimator_output_dirs=tree_estimator_output_dirs,
    #    families=families
    #)
    return res



def coevolution_end_to_end_with_cherryml_optimizer(
    msa_dir: str,
    contact_map_dir: str,
    minimum_distance_for_nontrivial_contact: int,
    coevolution_mask_path: Optional[str],
    families: List[str],
    tree_estimator: PhylogenyEstimatorType,
    initial_tree_estimator_rate_matrix_path: str,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    optimizer_device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 500,
    do_adam: bool = True,
    edge_or_cherry: str = CHERRYML_TYPE,
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    num_processes_tree_estimation: int = 8,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 8,
    optimizer_initialization: str = "jtt-ipw",
    use_maximal_matching: bool = True,
    tree_dir: Optional[str] = None,
    alphabet: List[str] = get_amino_acids(),
) -> Dict:
    """
    Cherry estimator for coevolution.

    Returns a dictionary with the directories to the intermediate outputs. In
    particular, the learned coevolution rate matrix is indexed by
    "learned_rate_matrix_path"

    You can provide the trees with tree_dir . Otherwise, the provided
    tree_estimator will be used to estimate trees.
    """
    res = {}

    quantization_points = [
        ("%.8f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    res["quantization_points"] = quantization_points

    current_estimate_rate_matrix_path = initial_tree_estimator_rate_matrix_path
    for iteration in range(1):  # There is no repeated iteration in this case.
        if iteration == 0 and tree_dir is not None:
            tree_estimator_output_dirs = {"output_tree_dir": tree_dir}
        else:
            tree_estimator_output_dirs = tree_estimator(
                msa_dir=msa_dir,
                families=families,
                rate_matrix_path=current_estimate_rate_matrix_path,
                num_processes=num_processes_tree_estimation,
            )
        res[
            f"tree_estimator_output_dirs_{iteration}"
        ] = tree_estimator_output_dirs

        mdnc = minimum_distance_for_nontrivial_contact

        if use_maximal_matching:
            # Need to compute a maximal matching instead of using the whole
            # contact maps
            contact_map_dir = create_maximal_matching_contact_map(
                i_contact_map_dir=contact_map_dir,
                families=families,
                minimum_distance_for_nontrivial_contact=mdnc,
                num_processes=num_processes_counting,
            )["o_contact_map_dir"]

        count_matrices_dir = count_co_transitions(
            tree_dir=tree_estimator_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            families=families,
            amino_acids=alphabet[:],
            quantization_points=quantization_points,
            edge_or_cherry=edge_or_cherry,
            minimum_distance_for_nontrivial_contact=mdnc,
            num_processes=num_processes_counting,
            use_cpp_implementation=use_cpp_counting_implementation,
            cpp_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_command_line_suffix=cpp_counting_command_line_suffix,
        )["output_count_matrices_dir"]

        res[f"count_matrices_dir_{iteration}"] = count_matrices_dir

        jtt_ipw_dir = jtt_ipw(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            mask_path=coevolution_mask_path,
            use_ipw=True,
            normalize=False,
        )["output_rate_matrix_dir"]

        res[f"jtt_ipw_dir_{iteration}"] = jtt_ipw_dir

        initialization_path = None
        if optimizer_initialization == "jtt-ipw":
            initialization_path = os.path.join(jtt_ipw_dir, "result.txt")
        elif optimizer_initialization == "equ_x_equ":
            initialization_path = get_equ_x_equ_path()
        elif optimizer_initialization == "random":
            initialization_path = None
        else:
            raise ValueError(
                f"Unknown optimizer_initialization = {optimizer_initialization}"
            )

        rate_matrix_dir = quantized_transitions_mle(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            initialization_path=initialization_path,
            mask_path=coevolution_mask_path,
            stationary_distribution_path=None,
            rate_matrix_parameterization="pande_reversible",
            device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            OMP_NUM_THREADS=num_processes_optimization,
            OPENBLAS_NUM_THREADS=num_processes_optimization,
        )["output_rate_matrix_dir"]

        res[f"rate_matrix_dir_{iteration}"] = rate_matrix_dir

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )

    res["learned_rate_matrix_path"] = current_estimate_rate_matrix_path

    return res
