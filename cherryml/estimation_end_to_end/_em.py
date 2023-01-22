import logging
import os
import sys
from typing import Dict, List, Optional

from cherryml.counting import count_transitions
from cherryml.estimation import em_lg, em_lg_xrate, jtt_ipw
from cherryml.estimation_end_to_end._cherry import _subset_data_to_sites_subset
from cherryml.markov_chain import get_equ_path
from cherryml.types import PhylogenyEstimatorType
from cherryml.utils import get_amino_acids

from ._cherry import (
    _get_runtime_from_profiling_file,
    _get_tree_estimation_runtime,
)


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def lg_end_to_end_with_em_optimizer(
    msa_dir: str,
    families: List[str],
    tree_estimator: PhylogenyEstimatorType,
    initial_tree_estimator_rate_matrix_path: str,
    num_iterations: Optional[int] = 1,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    extra_em_command_line_args: str = "-log 6 -f 3 -mi 0.000001",
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    num_processes_tree_estimation: int = 8,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 2,
    optimizer_initialization: str = "jtt-ipw",
    sites_subset_dir: Optional[str] = None,
    em_backend: str = "xrate",
) -> Dict:
    if sites_subset_dir is not None and num_iterations > 1:
        raise Exception(
            "You are doing more than 1 iteration while learning a model only"
            "on a subset of sites. This is most certainly a usage error."
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
    time_counting = 0
    time_jtt_ipw = 0
    time_optimization = 0

    current_estimate_rate_matrix_path = initial_tree_estimator_rate_matrix_path
    for iteration in range(num_iterations):
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
            tree_estimator_output_dirs, families
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
            amino_acids=get_amino_acids(),
            quantization_points=quantization_points,
            edge_or_cherry="cherry",
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
        elif optimizer_initialization.endswith(".txt"):
            initialization_path = optimizer_initialization
        else:
            raise ValueError(
                f"Unknown optimizer_initialization = {optimizer_initialization}"
            )

        if em_backend == "historian":
            em_backend_fn = em_lg
        elif em_backend == "xrate":
            em_backend_fn = em_lg_xrate
        else:
            raise ValueError(
                f"Unknown EM backend: {em_backend}. Allowed: 'historian', "
                "'xrate'."
            )

        rate_matrix_dir = em_backend_fn(
            tree_dir=tree_estimator_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            site_rates_dir=tree_estimator_output_dirs["output_site_rates_dir"],
            families=families,
            initialization_rate_matrix_path=initialization_path,
            extra_command_line_args=extra_em_command_line_args,
        )["output_rate_matrix_dir"]
        time_optimization += _get_runtime_from_profiling_file(
            os.path.join(rate_matrix_dir, "profiling.txt")
        )

        res[f"rate_matrix_dir_{iteration}"] = rate_matrix_dir

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )

    res["learned_rate_matrix_path"] = current_estimate_rate_matrix_path

    res["time_tree_estimation"] = time_tree_estimation
    res["time_counting"] = time_counting
    res["time_jtt_ipw"] = time_jtt_ipw
    res["time_optimization"] = time_optimization
    res["total_cpu_time"] = (
        time_tree_estimation + time_counting + time_jtt_ipw + time_optimization
    )

    profiling_str = (
        f"EM runtimes:\n"
        "time_tree_estimation (without parallelization): "
        f"{res['time_tree_estimation']}\n"
        f"time_counting: {res['time_counting']}\n"
        f"time_jtt_ipw: {res['time_jtt_ipw']}\n"
        f"time_optimization: {res['time_optimization']}\n"
        f"total_cpu_time: {res['total_cpu_time']}\n"
    )
    res["profiling_str"] = profiling_str

    return res
