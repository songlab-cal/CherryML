import hashlib
import logging
import multiprocessing
import os
import random
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import tqdm

from cherryml.caching import cached_parallel_computation, secure_parallel_output
from cherryml.io import (
    read_contact_map,
    read_probability_distribution,
    read_rate_matrix,
    read_site_rates,
    read_tree,
    write_msa,
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


def sample(probability_distribution: np.array) -> int:
    return np.random.choice(
        range(len(probability_distribution)), p=probability_distribution
    )


def sample_transition(
    starting_state: int,
    rate_matrix: np.array,
    elapsed_time: float,
    strategy: str,
) -> int:
    """
    Sample the ending state of the Markov chain.

    Args:
        starting_state: The starting state (an integer)
        rate_matrix: The rate matrix
        elapsed_time: The amount of time that the chain is run.
        strategy: Either "all_transitions" or "node_states".
            The "node_states" strategy uses the matrix exponential to figure
            out how to change state from node to node directly. The
            "all_transitions" strategy samples waiting times from an exponential
            distribution to figure out _all_ changes in the elapsed time and
            thus get the ending state. (It is unclear which method will be
            faster)
    """
    if strategy == "all_transitions":
        n = rate_matrix.shape[0]
        curr_state_index = starting_state
        current_t = 0  # We simulate the process starting from time 0.
        while True:
            # See when the next transition happens
            waiting_time = np.random.exponential(
                1.0 / -rate_matrix[curr_state_index, curr_state_index]
            )
            current_t += waiting_time
            if current_t >= elapsed_time:
                # We reached the end of the process
                return curr_state_index
            # Update the curr_state_index
            weights = list(
                rate_matrix[curr_state_index, :curr_state_index]
            ) + list(rate_matrix[curr_state_index, (curr_state_index + 1) :])
            assert len(weights) == n - 1
            new_state_index = random.choices(
                population=range(n - 1), weights=weights, k=1
            )[0]
            # Because new_state_index is in [0, n - 2], we must map it back to
            # [0, n - 1].
            if new_state_index >= curr_state_index:
                new_state_index += 1
            curr_state_index = new_state_index
    elif strategy == "chain_jump":
        raise NotImplementedError
    else:
        raise Exception(f"Unknown strategy: {strategy}")


def _map_func(args: Dict):
    """
    Version of simulate_msas run by an individual process.
    """
    tree_dir = args[0]
    site_rates_dir = args[1]
    contact_map_dir = args[2]
    families = args[3]
    amino_acids = args[4]
    pi_1_path = args[5]
    Q_1_path = args[6]
    pi_2_path = args[7]
    Q_2_path = args[8]
    strategy = args[9]
    output_msa_dir = args[10]
    random_seed = args[11]

    for family in families:
        st = time.time()
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        site_rates = read_site_rates(
            site_rates_path=os.path.join(site_rates_dir, family + ".txt")
        )
        contact_map = read_contact_map(
            contact_map_path=os.path.join(contact_map_dir, family + ".txt")
        )
        pi_1_df = read_probability_distribution(pi_1_path)
        Q_1_df = read_rate_matrix(Q_1_path)
        pi_2_df = read_probability_distribution(pi_2_path)
        Q_2_df = read_rate_matrix(Q_2_path)

        pairs_of_amino_acids = [
            aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
        ]
        # Validate states of rate matrices and root distribution
        if list(pi_1_df.index) != amino_acids:
            raise Exception(
                f"pi_1 index is:\n{pi_1_df.index}\nbut expected amino acids:"
                f"\n{amino_acids}"
            )
        if list(pi_2_df.index) != pairs_of_amino_acids:
            raise Exception(
                f"pi_2 index is:\n{pi_2_df.index}\nbut expected pairs of amino "
                f"acids:\n{pairs_of_amino_acids}"
            )
        if list(Q_1_df.index) != amino_acids:
            raise Exception(
                f"Q_1 index is:\n{Q_1_df.index}\n\nbut expected amino acids:"
                f"\n{amino_acids}"
            )
        if list(Q_1_df.columns) != amino_acids:
            raise Exception(
                f"Q_1 columns are:\n{Q_1_df.columns}\n\nbut expected amino "
                f"acids:\n{amino_acids}"
            )
        if list(Q_2_df.index) != pairs_of_amino_acids:
            raise Exception(
                f"Q_2 index is:\n{Q_2_df.index}\n\nbut expected pairs of amino "
                f"acids:\n{pairs_of_amino_acids}"
            )
        if list(Q_2_df.columns) != pairs_of_amino_acids:
            raise Exception(
                f"Q_1 columns are:\n{Q_2_df.columns}\n\nbut expected pairs of "
                f"amino acids:\n{pairs_of_amino_acids}"
            )
        pi_1 = pi_1_df.to_numpy().reshape(-1)
        pi_2 = pi_2_df.to_numpy().reshape(-1)
        Q_1 = Q_1_df.to_numpy()
        Q_2 = Q_2_df.to_numpy()

        num_sites = len(site_rates)

        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
        # Validate that each site is in contact with at most one other site
        contacting_sites = list(sum(contacting_pairs, ()))
        if len(set(contacting_sites)) != len(contacting_sites):
            raise Exception(
                f"Each site can only be in contact with one other site. "
                f"The contacting sites were: {contacting_pairs}"
            )
        independent_sites = [
            i for i in range(num_sites) if i not in contacting_sites
        ]

        n_independent_sites = len(independent_sites)
        n_contacting_pairs = len(contacting_pairs)

        # We work with *integer states* and then convert back to amino acids at
        # the end. The first n_independent_sites columns will evolve
        # independently, and the last n_independent_sites columns will
        # co-evolve.
        seed = (
            int(hashlib.md5(family.encode()).hexdigest()[:8], 16) + random_seed
        )
        random.seed(seed)
        np.random.seed(seed)
        msa_int = {}  # type: Dict[str, List[int]]

        # Sample root state
        root_states = []
        for i in range(n_independent_sites):
            root_states.append(sample(pi_1))
        for i in range(n_contacting_pairs):
            root_states.append(sample(pi_2))
        msa_int[tree.root()] = root_states

        # Depth first search from root
        root = tree.root()
        for node in tree.preorder_traversal():
            if node == root:
                continue
            node_states_int = []
            parent, branch_length = tree.parent(node)
            parent_states = msa_int[parent]
            for i in range(n_independent_sites):
                node_states_int.append(
                    sample_transition(
                        starting_state=parent_states[i],
                        rate_matrix=Q_1,
                        elapsed_time=branch_length
                        * site_rates[independent_sites[i]],
                        strategy=strategy,
                    )
                )
            for i in range(n_contacting_pairs):
                node_states_int.append(
                    sample_transition(
                        starting_state=parent_states[n_independent_sites + i],
                        rate_matrix=Q_2,
                        elapsed_time=branch_length,  # No site rate adjustment
                        strategy=strategy,
                    )
                )
            msa_int[node] = node_states_int

        # Now just map back the integer states to amino acids
        msa = {}
        for node in msa_int.keys():
            node_states_int = msa_int[node]
            node_states = ["" for i in range(num_sites)]
            for i in range(n_independent_sites):
                state_int = node_states_int[i]
                state_str = amino_acids[state_int]
                node_states[independent_sites[i]] = state_str
            for i in range(n_contacting_pairs):
                state_int = node_states_int[n_independent_sites + i]
                state_str = pairs_of_amino_acids[state_int]
                (site_1, site_2) = contacting_pairs[i]
                if site_2 >= len(node_states):
                    raise Exception(
                        f"Site {(site_1, site_2)} out of range: "
                        f"{len(node_states)}"
                    )
                node_states[site_1] = state_str[0]
                node_states[site_2] = state_str[1]
            msa[node] = "".join(node_states)
            if not all([state != "" for state in state_str]):
                raise Exception("Error mapping integer states to amino acids.")
        msa_path = os.path.join(output_msa_dir, family + ".txt")
        write_msa(
            msa,
            msa_path,
        )
        secure_parallel_output(output_msa_dir, family)
        et = time.time()
        open(os.path.join(output_msa_dir, family + ".profiling"), "w").write(
            f"Total time: {et - st}\n"
        )


@cached_parallel_computation(
    parallel_arg="families",
    exclude_args=[
        "num_processes",
        "cpp_command_line_prefix",
        "cpp_command_line_suffix",
    ],
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def simulate_msas(
    tree_dir: str,
    site_rates_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    pi_2_path: str,
    Q_2_path: str,
    strategy: str,
    random_seed: int,
    num_processes: Optional[int] = 1,
    use_cpp_implementation: bool = True,
    cpp_command_line_prefix: str = "",
    cpp_command_line_suffix: str = "0",
    output_msa_dir: Optional[str] = None,
) -> None:
    """
    Simulate multiple sequence alignments (MSAs).

    Given a contact map and models for the evolution of contacting sites and
    non-contacting sites, protein sequences are simulated and written out to
    output_msa_paths.

    Details:
    - For each position, it must be either in contact with exactly 1 other
        position, or not be in contact with any other position. The diagonal
        of the contact matrix is ignored.
    - If i is in contact with j, then j is in contact with i (i.e. the relation
        is symmetric, and so the contact map is symmetric).
    - The Q_2 matrix is sparse: only 2 * len(amino_acids) - 1 entries in each
        row are non-zero, since only one amino acid in a contacting pair
        can mutate at a time.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves. Rates for sites that co-evolve are ignored.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of (valid) amino acids.
        pi_1_path: Path to an array of length len(amino_acids). It indicates,
            for sites that evolve independently (i.e. that are not in contact
            with any other site), the probabilities for the root state.
        Q_1_path: Path to an array of size len(amino_acids) x len(amino_acids),
            the rate matrix describing the evolution of sites that evolve
            independently (i.e. that are not in contact with any other site).
        pi_2_path: Path to an array of length len(amino_acids) ** 2. It
            indicates, for sites that are in contact, the probabilities for
            their root state.
        Q_2_path: Path to an array of size (len(amino_acids) ** 2) x
            (len(amino_acids) ** 2), the rate matrix describing the evolution
            of sites that are in contact.
        strategy: Either 'all_transitions' or 'chain_jump'. The
            'all_transitions' strategy samples all state changes on the tree
            and does not require the matrix exponential, while the 'chain_jump'
            strategy does not sample all state changes on the tree but requires
            the matrix exponential.
        output_msa_dir: Directory where to write the multiple sequence
            alignments to in FASTA format.
        random_seed: Random seed for reproducibility. Using the same random
            seed and strategy leads to the exact same simulated data.
        num_processes: Number of processes used to parallelize computation.
        use_cpp_implementation: If to use efficient C++ implementation
            instead of Python.
        cpp_command_line_prefix: E.g. to run the C++ binary on slurm.
        cpp_command_line_suffix: For extra C++ args related to performance.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Going to simulate MSAs for {len(families)} families using "
        f"{num_processes} processes."
    )

    if not os.path.exists(output_msa_dir):
        os.makedirs(output_msa_dir)

    if use_cpp_implementation:
        # check if the binary exists
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cpp_path = os.path.join(dir_path, "simulate.cpp")
        bin_path = os.path.join(dir_path, "simulate")
        print(f"cpp_path = {cpp_path}")
        if not os.path.exists(bin_path):
            # load openmpi/openmp modules
            # Currently it should run on the interactive node
            command = f"mpicxx -std=c++11 -O3 -o {bin_path} {cpp_path}"
            os.system(command)
            if not os.path.exists(bin_path):
                raise Exception(
                    f"Couldn't compile simulate.cpp. Command: {command}"
                )
        with tempfile.NamedTemporaryFile("w") as families_file:
            families_path = families_file.name
            open(families_path, "w").write(" ".join(families))
            command = ""
            command += f" {cpp_command_line_prefix}"
            command += f" mpirun -np {num_processes}"
            command += f" {bin_path}"
            command += f" {tree_dir}"
            command += f" {site_rates_dir}"
            command += f" {contact_map_dir}"
            command += f" {len(families)}"
            command += f" {len(amino_acids)}"
            command += f" {pi_1_path}"
            command += f" {Q_1_path}"
            command += f" {pi_2_path}"
            command += f" {Q_2_path}"
            command += f" {strategy}"
            command += f" {output_msa_dir}"
            command += f" {random_seed}"
            command += f" {families_path}"
            command += " " + " ".join(amino_acids)
            command += " " + cpp_command_line_suffix
            # print(f"Going to run:\n{command}")
            os.system(command)
            return

    map_args = [
        [
            tree_dir,
            site_rates_dir,
            contact_map_dir,
            get_process_args(process_rank, num_processes, families),
            amino_acids,
            pi_1_path,
            Q_1_path,
            pi_2_path,
            Q_2_path,
            strategy,
            output_msa_dir,
            random_seed,
        ]
        for process_rank in range(num_processes)
    ]

    # Map step (distribute families among processes)
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
