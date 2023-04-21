import hashlib
import logging
import multiprocessing
import os
import sys
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from cherryml import caching
from cherryml.caching import secure_parallel_output
from cherryml.evaluation import create_maximal_matching_contact_map
from cherryml.global_vars import TITLES
from cherryml.io import read_msa, write_contact_map, write_msa
from cherryml.markov_chain import (
    get_lg_path,
    get_lg_stationary_path,
    get_lg_x_lg_path,
    get_lg_x_lg_stationary_path,
)
from cherryml.phylogeny_estimation import fast_tree
from cherryml.simulation import simulate_msas
from cherryml.utils import get_amino_acids, get_process_args

from ._contact_generation.ContactMatrix import ContactMatrix


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _compute_contact_map(
    pfam_15k_pdb_dir: str,
    family: str,
    angstrom_cutoff: float,
    output_contact_map_dir: str,
) -> None:
    output_contact_map_path = os.path.join(
        output_contact_map_dir, family + ".txt"
    )
    contact_matrix = ContactMatrix(
        pdb_dir=pfam_15k_pdb_dir,
        protein_family_name=family,
        angstrom_cutoff=angstrom_cutoff,
    )
    contact_matrix.write_to_file(output_contact_map_path)
    secure_parallel_output(output_contact_map_dir, family)


def _map_func_compute_contact_maps(args: List) -> None:
    pfam_15k_pdb_dir = args[0]
    families = args[1]
    angstrom_cutoff = args[2]
    output_contact_map_dir = args[3]

    for family in families:
        _compute_contact_map(
            pfam_15k_pdb_dir=pfam_15k_pdb_dir,
            family=family,
            angstrom_cutoff=angstrom_cutoff,
            output_contact_map_dir=output_contact_map_dir,
        )


@caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_contact_map_dir"],
    write_extra_log_files=True,
)
def compute_contact_maps(
    pfam_15k_pdb_dir: str,
    families: List[str],
    angstrom_cutoff: float,
    num_processes: int,
    output_contact_map_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Going to compute contact maps for {len(families)} families")

    if not os.path.exists(pfam_15k_pdb_dir):
        raise ValueError(f"Could not find pfam_15k_pdb_dir {pfam_15k_pdb_dir}")

    map_args = [
        [
            pfam_15k_pdb_dir,
            get_process_args(process_rank, num_processes, families),
            angstrom_cutoff,
            output_contact_map_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_compute_contact_maps, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_compute_contact_maps, map_args),
                total=len(map_args),
            )
        )


def _subsample_pfam_15k_msa(
    pfam_15k_msa_path: str,
    num_sequences: Optional[int],
    output_msa_dir: str,
    family: str,
):
    if not os.path.exists(pfam_15k_msa_path):
        raise FileNotFoundError(f"MSA file {pfam_15k_msa_path} does not exist!")

    # Read MSA
    msa = []  # type: List[Tuple[str, str]]
    with open(pfam_15k_msa_path) as file:
        lines = list(file)
        n_lines = len(lines)
        for i in range(0, n_lines, 2):
            if not lines[i][0] == ">":
                raise Exception("Protein name line should start with '>'")
            protein_name = lines[i][1:].strip()
            protein_seq = lines[i + 1].strip()
            # Lowercase amino acids in the sequence are repetitive
            # sequences and should be ignored.
            protein_seq = "".join([c for c in protein_seq if not c.islower()])
            msa.append((protein_name, protein_seq))
        # Check that all sequences in the MSA have the same length.
        for i in range(len(msa) - 1):
            if len(msa[i][1]) != len(msa[i + 1][1]):
                raise Exception(
                    f"Sequence\n{msa[i][1]}\nand\n{msa[i + 1][1]}\nin the "
                    f"MSA do not have the same length! ({len(msa[i][1])} vs"
                    f" {len(msa[i + 1][1])})"
                )

    # Subsample MSA
    family_int_hash = (
        int(
            hashlib.sha512(
                (family + "-_subsample_pfam_15k_msa").encode("utf-8")
            ).hexdigest(),
            16,
        )
        % 10**8
    )
    rng = np.random.default_rng(family_int_hash)
    nseqs = len(msa)
    if num_sequences is not None:
        max_seqs = min(nseqs, num_sequences)
        seqs_to_keep = [0] + list(
            rng.choice(range(1, nseqs, 1), size=max_seqs - 1, replace=False)
        )
        seqs_to_keep = sorted(seqs_to_keep)
        msa = [msa[i] for i in seqs_to_keep]
    msa_dict = dict(msa)
    write_msa(
        msa=msa_dict, msa_path=os.path.join(output_msa_dir, family + ".txt")
    )
    secure_parallel_output(output_msa_dir, family)


def _map_func_subsample_pfam_15k_msas(args: List):
    pfam_15k_msa_dir = args[0]
    num_sequences = args[1]
    families = args[2]
    output_msa_dir = args[3]
    for family in families:
        _subsample_pfam_15k_msa(
            pfam_15k_msa_path=os.path.join(pfam_15k_msa_dir, family + ".a3m"),
            num_sequences=num_sequences,
            output_msa_dir=output_msa_dir,
            family=family,
        )


@caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def subsample_pfam_15k_msas(
    pfam_15k_msa_dir: str,
    num_sequences: int,
    families: List[str],
    output_msa_dir: str,
    num_processes: int,
):
    map_args = [
        [
            pfam_15k_msa_dir,
            num_sequences,
            get_process_args(process_rank, num_processes, families),
            output_msa_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_subsample_pfam_15k_msas, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_subsample_pfam_15k_msas, map_args),
                total=len(map_args),
            )
        )


@caching.cached()
def get_families(
    pfam_15k_msa_dir: str,
) -> List[str]:
    """
    Get the list of protein families names.

    Args:
        pfam_15k_msa_dir: Directory with the MSA files. There should be one
            file with name family.txt for each protein family.

    Returns:
        The list of protein family names in the provided directory.
    """
    families = sorted(list(os.listdir(pfam_15k_msa_dir)))
    families = [x.split(".")[0] for x in families if x.endswith(".a3m")]
    return families


@caching.cached()
def get_family_sizes(
    pfam_15k_msa_dir: str,
) -> pd.DataFrame:
    """
    Get the size of each protein family.

    By 'size' we mean the number of sequences and the number of sites. These
    are returned in a Pandas DataFrame object, with one row per family.

    Args:
        pfam_15k_msa_dir: Directory with the MSA files. There should be one
            file with name family.a3m for each protein family.

    Returns:
        A Pandas DataFrame with one row per family, containing the num_sequences
        and num_sites.
    """
    families = get_families(pfam_15k_msa_dir=pfam_15k_msa_dir)
    family_size_tuples = []
    for family in families:
        for i, line in enumerate(
            open(os.path.join(pfam_15k_msa_dir, f"{family}.a3m"), "r")
        ):
            if i == 1:
                num_sites = len(line.strip())
        num_lines = (
            open(os.path.join(pfam_15k_msa_dir, f"{family}.a3m"), "r")
            .read()
            .strip()
            .count("\n")
            + 1
        )
        assert num_lines % 2 == 0
        num_sequences = num_lines // 2
        family_size_tuples.append((family, num_sequences, num_sites))
    family_size_df = pd.DataFrame(
        family_size_tuples, columns=["family", "num_sequences", "num_sites"]
    )
    return family_size_df


@caching.cached()
def get_families_within_cutoff(
    pfam_15k_msa_dir: str,
    min_num_sites: Optional[int] = 0,
    max_num_sites: Optional[int] = 1000000,
    min_num_sequences: Optional[int] = 0,
    max_num_sequences: Optional[int] = 1000000,
) -> List[str]:
    family_size_df = get_family_sizes(
        pfam_15k_msa_dir=pfam_15k_msa_dir,
    )
    families = family_size_df.family[
        (family_size_df.num_sites >= min_num_sites)
        & (family_size_df.num_sites <= max_num_sites)
        & (family_size_df.num_sequences >= min_num_sequences)
        & (family_size_df.num_sequences <= max_num_sequences)
    ]
    families = list(families)
    return families


def fig_family_sizes(
    msa_dir: str,
    max_families: Optional[int] = None,
) -> None:
    """
    Histograms of num_sequences and num_sites.
    """
    family_size_df = get_family_sizes(
        msa_dir=msa_dir,
        max_families=max_families,
    )
    if TITLES:
        plt.title("Distribution of family num_sequences")
    plt.hist(family_size_df.num_sequences)
    plt.show()

    if TITLES:
        plt.title("Distribution of family num_sites")
    print(f"median num_sites = {family_size_df.num_sites.median()}")
    print(f"mode num_sites = {family_size_df.num_sites.mode()}")
    print(f"mean num_sites = {family_size_df.num_sites.mean()}")
    plt.hist(family_size_df.num_sites, bins=100)
    plt.show()

    plt.xlabel("num_sequences")
    plt.ylabel("num_sites")
    plt.scatter(
        family_size_df.num_sequences, family_size_df.num_sites, alpha=0.3
    )
    plt.show()


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_contact_map_dir"],
    write_extra_log_files=True,
)
def create_trivial_contact_maps(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_contact_map_dir: Optional[str] = None,
):
    for family in families:
        st = time.time()
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        num_sites = len(next(iter(msa.values())))
        contact_map = np.zeros(shape=(num_sites, num_sites), dtype=int)
        write_contact_map(
            contact_map, os.path.join(output_contact_map_dir, family + ".txt")
        )
        et = time.time()
        open(
            os.path.join(output_contact_map_dir, family + ".profiling"), "w"
        ).write(f"Total time: {et - st}\n")
        secure_parallel_output(output_contact_map_dir, family)


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def subset_msa_to_leaf_nodes(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_msa_dir: Optional[str] = None,
):
    """
    An internal node is anyone that starts with 'internal-'.
    """
    for family in families:
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        msa_subset = {
            seq_name: seq
            for (seq_name, seq) in msa.items()
            if not seq_name.startswith("internal-")
        }
        write_msa(msa_subset, os.path.join(output_msa_dir, family + ".txt"))
        secure_parallel_output(output_msa_dir, family)


@caching.cached()
def simulate_ground_truth_data_single_site(
    pfam_15k_msa_dir: str,
    families: List[str],
    num_sequences: int,
    num_rate_categories: int,
    num_processes: int,
    random_seed: int,
    use_cpp_simulation_implementation: bool = True,
):
    """
    Simulate ground truth MSAs with LG.
    """
    real_msa_dir = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=pfam_15k_msa_dir,
        num_sequences=num_sequences,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    # contact_map_dir = compute_contact_maps(
    #     pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
    #     families=families,
    #     angstrom_cutoff=8.0,
    #     num_processes=num_processes,
    # )["output_contact_map_dir"]

    fast_tree_output = fast_tree(
        msa_dir=real_msa_dir,
        families=families,
        rate_matrix_path=get_lg_path(),
        num_rate_categories=num_rate_categories,
        num_processes=num_processes,
    )

    gt_trees, gt_site_rates, gt_likelihood_dir = (
        fast_tree_output["output_tree_dir"],
        fast_tree_output["output_site_rates_dir"],
        fast_tree_output["output_likelihood_dir"],
    )

    # We only investigate single-site model here.
    contact_map_dir = create_trivial_contact_maps(
        msa_dir=real_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_contact_map_dir"]

    # Now we simulate MSAs
    gt_msa_dir = simulate_msas(
        tree_dir=gt_trees,
        site_rates_dir=gt_site_rates,
        contact_map_dir=contact_map_dir,
        families=families,
        amino_acids=get_amino_acids(),
        pi_1_path=get_lg_stationary_path(),
        Q_1_path=get_lg_path(),
        pi_2_path=get_lg_x_lg_stationary_path(),
        Q_2_path=get_lg_x_lg_path(),
        strategy="all_transitions",
        random_seed=random_seed,
        num_processes=num_processes,
        use_cpp_implementation=use_cpp_simulation_implementation,
    )["output_msa_dir"]

    # Now subset the MSAs to only the leaf nodes.
    msa_dir = subset_msa_to_leaf_nodes(
        msa_dir=gt_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_msa_dir"]

    return (
        msa_dir,
        contact_map_dir,
        gt_msa_dir,
        gt_trees,
        gt_site_rates,
        gt_likelihood_dir,
    )


@caching.cached(
    exclude_if_default=[
        "pi_2_path",
        "Q_2_path",
    ],
)
def simulate_ground_truth_data_coevolution(
    pfam_15k_msa_dir: str,
    pfam_15k_pdb_dir: str,
    minimum_distance_for_nontrivial_contact: int,
    angstrom_cutoff: float,
    families: List[str],
    num_sequences: int,
    num_rate_categories: int,
    num_processes: int,
    random_seed: int,
    use_cpp_simulation_implementation: bool,
    pi_2_path: str = get_lg_x_lg_stationary_path(),
    Q_2_path: str = get_lg_x_lg_path(),
):
    """
    Simulate ground truth MSAs with LG x LG.
    """
    real_msa_dir = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=pfam_15k_msa_dir,
        num_sequences=num_sequences,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    contact_map_dir = compute_contact_maps(
        pfam_15k_pdb_dir=pfam_15k_pdb_dir,
        families=families,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes,
    )["output_contact_map_dir"]

    mdnc = minimum_distance_for_nontrivial_contact
    contact_map_dir = create_maximal_matching_contact_map(
        i_contact_map_dir=contact_map_dir,
        families=families,
        minimum_distance_for_nontrivial_contact=mdnc,
        num_processes=num_processes,
    )["o_contact_map_dir"]

    fast_tree_output = fast_tree(
        msa_dir=real_msa_dir,
        families=families,
        rate_matrix_path=get_lg_path(),
        num_rate_categories=num_rate_categories,
        num_processes=num_processes,
    )

    gt_trees, gt_site_rates, gt_likelihood_dir = (
        fast_tree_output["output_tree_dir"],
        fast_tree_output["output_site_rates_dir"],
        fast_tree_output["output_likelihood_dir"],
    )

    # Now we simulate MSAs
    gt_msa_dir = simulate_msas(
        tree_dir=gt_trees,
        site_rates_dir=gt_site_rates,
        contact_map_dir=contact_map_dir,
        families=families,
        amino_acids=get_amino_acids(),
        pi_1_path=get_lg_stationary_path(),
        Q_1_path=get_lg_path(),
        pi_2_path=pi_2_path,
        Q_2_path=Q_2_path,
        strategy="all_transitions",
        random_seed=random_seed,
        num_processes=num_processes,
        use_cpp_implementation=use_cpp_simulation_implementation,
    )["output_msa_dir"]

    # Now subset the MSAs to only the leaf nodes.
    msa_dir = subset_msa_to_leaf_nodes(
        msa_dir=gt_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_msa_dir"]

    return (
        msa_dir,
        contact_map_dir,
        gt_msa_dir,
        gt_trees,
        gt_site_rates,
        gt_likelihood_dir,
    )
