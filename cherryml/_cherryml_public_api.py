"""
Public API for CherryML as applied to the LG model and the co-evolution model.
"""
import logging
import os
import sys
import tempfile
from functools import partial
from typing import List, Optional

from cherryml import caching, utils
from cherryml.estimation_end_to_end import (
    coevolution_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_cherryml_optimizer,
)
from cherryml.io import read_rate_matrix, write_rate_matrix
from cherryml.markov_chain import get_lg_path
from cherryml.phylogeny_estimation import fast_tree, phyml, fast_cherries


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


def cherryml_public_api(
    output_path: str,
    model_name: str,
    msa_dir: str,
    contact_map_dir: Optional[str] = None,
    tree_dir: Optional[str] = None,
    site_rates_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_processes_tree_estimation: int = 32,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 2,
    num_rate_categories: int = 20,
    initial_tree_estimator_rate_matrix_path: str = get_lg_path(),
    num_iterations: int = 1,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    optimizer_device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 500,
    minimum_distance_for_nontrivial_contact: int = 7,
    do_adam: bool = True,
    cherryml_type: str = "cherry++",
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    optimizer_initialization: str = "jtt-ipw",
    sites_subset_dir: Optional[str] = None,
    coevolution_mask_path: Optional[str] = None,
    use_maximal_matching: bool = True,
    families: Optional[List[str]] = None,
    tree_estimator_name: str = "FastTree",
) -> str:
    """
    CherryML method applied to the LG model and the co-evolution model.

    This minimalist API should suit most use cases. The learnt rate matrix
    will be written to `output_path`.

    In the simplest case, to learn a 20 x 20 rate matrix with the LG model,
    just provide a `msa_dir` directory with one MSA file per family, and
    specify `model_name="LG"`.

    To learn a 400 x 400 co-evolution rate matrix, in the simplest case, just
    provide a `msa_dir` directory with one MSA file per family, a
    `contact_map_dir` directory with one contact map per family, and specify
    `model_name="co-evolution"`.

    If you already have estimated trees and site rates for each family, you can
    provide these directories with `tree_dir` and `site_rates_dir` respectively.
    Note that site rates are not used to learn the co-evolution model.

    The directories `msa_dir`, `contact_map_dir`, `tree_dir`, `site_rates_dir` -
    if provided - should contain one file per family, named `[family_name].txt`,
    where `family_name` is the name of the family.

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

    The files in `contact_map_dir` should list the contact map for each family
    following the format in the following example:
    ```
    5 sites
    10101
    01110
    11110
    01111
    10011
    ```

    The files in `tree_dir` should list the trees for in each family following
    the format in the following example:
    ```
    6 nodes
    internal-0
    internal-1
    internal-2
    seq1
    seq2
    seq3
    5 edges
    internal-0 internal-1 1.0
    internal-1 internal-2 2.0
    internal-2 seq1 3.0
    internal-2 seq2 4.0
    internal-1 seq3 5.0
    ```
    This format is intended to be easier to parse than the newick format. It
    first lists the node in the tree, and then the edges with their length.

    The files in `site_rates_dir` should list the site rates for in each family
    following the format in the following example:
    ```
    5 sites
    1.0 0.8 1.2 0.7 1.05
    ```

    Args:
        See description of arguments in https://github.com/songlab-cal/CherryML
    """
    if model_name not in ["LG", "co-evolution"]:
        raise ValueError('model_name should be either "LG" or "co-evolution".')

    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory()
        logger.info(
            "Cache directory not provided. Will use temporary directory "
            f"{cache_dir} to cache computations."
        )

    caching.set_cache_dir(cache_dir)

    if families is None:
        families = utils.get_families(msa_dir)

    if tree_estimator_name == "FastTree":
        tree_estimator = fast_tree
    elif tree_estimator_name == "PhyML":
        tree_estimator = phyml
    elif tree_estimator_name == "FastCherries":
        tree_estimator = partial(fast_cherries, max_iters=50)
    else:
        raise ValueError(f"Unknown tree_estimator_name: {tree_estimator_name}")

    if model_name == "LG":
        outputs = lg_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_dir,
            families=families,
            tree_estimator=partial(
                tree_estimator,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=initial_tree_estimator_rate_matrix_path,  # noqa
            num_iterations=num_iterations,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_cpp_counting_implementation=use_cpp_counting_implementation,
            optimizer_device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            edge_or_cherry=cherryml_type,
            cpp_counting_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_counting_command_line_suffix=cpp_counting_command_line_suffix,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
            optimizer_initialization=optimizer_initialization,
            sites_subset_dir=sites_subset_dir,
            tree_dir=tree_dir,
            site_rates_dir=site_rates_dir,
        )
        learned_rate_matrix = read_rate_matrix(
            os.path.join(outputs["learned_rate_matrix_path"])
        )
        write_rate_matrix(
            learned_rate_matrix.to_numpy(),
            list(learned_rate_matrix.columns),
            output_path,
        )

    elif model_name == "co-evolution":
        if num_iterations > 1:
            raise ValueError(
                "Iteration is not used for learning a coevolution model. "
                f"You provided: num_iterations={num_iterations}. Set this "
                "argument to 1 and retry."
            )
        outputs = coevolution_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,  # noqa
            coevolution_mask_path=coevolution_mask_path,
            families=families,
            tree_estimator=partial(
                tree_estimator,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=initial_tree_estimator_rate_matrix_path,  # noqa
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_cpp_counting_implementation=use_cpp_counting_implementation,
            optimizer_device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            edge_or_cherry=cherryml_type,
            cpp_counting_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_counting_command_line_suffix=cpp_counting_command_line_suffix,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
            optimizer_initialization=optimizer_initialization,
            use_maximal_matching=use_maximal_matching,
            tree_dir=tree_dir,
        )
        learned_rate_matrix = read_rate_matrix(
            os.path.join(outputs["learned_rate_matrix_path"])
        )
        write_rate_matrix(
            learned_rate_matrix.to_numpy(),
            list(learned_rate_matrix.columns),
            output_path,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
