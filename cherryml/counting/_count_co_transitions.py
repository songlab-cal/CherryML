import logging
import multiprocessing
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

from cherryml import caching
from cherryml.io import (
    read_contact_map,
    read_msa,
    read_tree,
    write_count_matrices,
)
from cherryml.utils import get_process_args, quantization_idx


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


def _map_func(args) -> List[Tuple[float, pd.DataFrame]]:
    """
    Version of count_co_transitions run by an individual process.

    Results from each process are later aggregated in the master process.
    """
    tree_dir = args[0]
    msa_dir = args[1]
    contact_map_dir = args[2]
    families = args[3]
    amino_acids = args[4]
    quantization_points = np.array(sorted(args[5]))
    edge_or_cherry = args[6]
    minimum_distance_for_nontrivial_contact = args[7]

    pairs_of_amino_acids = [
        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
    ]
    aa_pair_to_int = {
        aa_pair: i for (i, aa_pair) in enumerate(pairs_of_amino_acids)
    }

    num_amino_acids = len(amino_acids)
    count_matrices_numpy = np.zeros(
        shape=(
            len(quantization_points),
            num_amino_acids**2,
            num_amino_acids**2,
        )
    )
    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        msa = read_msa(msa_path=os.path.join(msa_dir, family + ".txt"))
        contact_map = read_contact_map(
            contact_map_path=os.path.join(contact_map_dir, family + ".txt")
        )
        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if abs(i - j) >= minimum_distance_for_nontrivial_contact and i < j
        ]
        if edge_or_cherry == "cherry++":
            total_pairs = []

            def dfs(node) -> Optional[Tuple[int, float]]:
                """
                Pair up leaves under me.

                Return a single unpaired leaf and its distance, it such exists.
                """
                if tree.is_leaf(node):
                    return (node, 0.0)
                unmatched_leaves_under = []
                distances_under = []
                for child, branch_length in tree.children(node):
                    maybe_unmatched_leaf, maybe_distance = dfs(child)
                    if maybe_unmatched_leaf is not None:
                        assert maybe_distance is not None
                        unmatched_leaves_under.append(maybe_unmatched_leaf)
                        distances_under.append(maybe_distance + branch_length)
                assert len(unmatched_leaves_under) == len(distances_under)
                index = 0

                while index + 1 <= len(unmatched_leaves_under) - 1:
                    total_pairs.append(1)
                    (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                        (unmatched_leaves_under[index], distances_under[index]),
                        (
                            unmatched_leaves_under[index + 1],
                            distances_under[index + 1],
                        ),
                    )
                    # NOTE: Copy-pasta from below ("cherry" case)...
                    leaf_seq_1, leaf_seq_2 = msa[leaf_1], msa[leaf_2]
                    branch_length_total = branch_length_1 + branch_length_2
                    q_idx = quantization_idx(
                        branch_length_total, quantization_points
                    )
                    if q_idx is not None:
                        for i, j in contacting_pairs:
                            # We accumulate the transitions in both directions
                            state_1 = leaf_seq_1[i] + leaf_seq_1[j]
                            state_2 = leaf_seq_2[i] + leaf_seq_2[j]
                            if (
                                leaf_seq_1[i] in amino_acids
                                and leaf_seq_1[j] in amino_acids
                                and leaf_seq_2[i] in amino_acids
                                and leaf_seq_2[j] in amino_acids
                            ):
                                for s, e in [
                                    (state_1, state_2),
                                    (state_1[::-1], state_2[::-1]),
                                    (state_2, state_1),
                                    (state_2[::-1], state_1[::-1]),
                                ]:
                                    s_idx = aa_pair_to_int[s]
                                    e_idx = aa_pair_to_int[e]
                                    count_matrices_numpy[
                                        q_idx, s_idx, e_idx
                                    ] += 0.25

                    index += 2
                if len(unmatched_leaves_under) % 2 == 0:
                    return (None, None)
                else:
                    return (unmatched_leaves_under[-1], distances_under[-1])

            dfs(tree.root())
            assert len(total_pairs) == int(len(tree.leaves()) / 2)
        else:
            for node in tree.nodes():
                if edge_or_cherry == "edge":
                    node_seq = msa[node]
                    # Extract all transitions on edges starting at 'node'
                    for child, branch_length in tree.children(node):
                        child_seq = msa[child]
                        q_idx = quantization_idx(
                            branch_length, quantization_points
                        )
                        if q_idx is not None:
                            for i, j in contacting_pairs:
                                start_state = node_seq[i] + node_seq[j]
                                end_state = child_seq[i] + child_seq[j]
                                if (
                                    node_seq[i] in amino_acids
                                    and node_seq[j] in amino_acids
                                    and child_seq[i] in amino_acids
                                    and child_seq[j] in amino_acids
                                ):
                                    for s, e in [
                                        (start_state, end_state),
                                        (start_state[::-1], end_state[::-1]),
                                    ]:
                                        s_idx = aa_pair_to_int[s]
                                        e_idx = aa_pair_to_int[e]
                                        count_matrices_numpy[
                                            q_idx, s_idx, e_idx
                                        ] += 0.5
                elif edge_or_cherry == "cherry":
                    children = tree.children(node)
                    if len(children) == 2 and all(
                        [tree.is_leaf(child) for (child, _) in children]
                    ):
                        (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                            children[0],
                            children[1],
                        )
                        leaf_seq_1, leaf_seq_2 = msa[leaf_1], msa[leaf_2]
                        branch_length_total = branch_length_1 + branch_length_2
                        q_idx = quantization_idx(
                            branch_length_total, quantization_points
                        )
                        if q_idx is not None:
                            for i, j in contacting_pairs:
                                # We accumulate the transitions in both directions
                                state_1 = leaf_seq_1[i] + leaf_seq_1[j]
                                state_2 = leaf_seq_2[i] + leaf_seq_2[j]
                                if (
                                    leaf_seq_1[i] in amino_acids
                                    and leaf_seq_1[j] in amino_acids
                                    and leaf_seq_2[i] in amino_acids
                                    and leaf_seq_2[j] in amino_acids
                                ):
                                    for s, e in [
                                        (state_1, state_2),
                                        (state_1[::-1], state_2[::-1]),
                                        (state_2, state_1),
                                        (state_2[::-1], state_1[::-1]),
                                    ]:
                                        s_idx = aa_pair_to_int[s]
                                        e_idx = aa_pair_to_int[e]
                                        count_matrices_numpy[
                                            q_idx, s_idx, e_idx
                                        ] += 0.25
    count_matrices = [
        [
            q,
            pd.DataFrame(
                count_matrices_numpy[q_idx, :, :],
                index=pairs_of_amino_acids,
                columns=pairs_of_amino_acids,
            ),
        ]
        for (q_idx, q) in enumerate(quantization_points)
    ]
    return count_matrices


@caching.cached_computation(
    exclude_args=[
        "num_processes",
        "use_cpp_implementation",
        "cpp_command_line_prefix",
        "cpp_command_line_suffix",
    ],
    output_dirs=["output_count_matrices_dir"],
    write_extra_log_files=True,
)
def count_co_transitions(
    tree_dir: str,
    msa_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    quantization_points: List[Union[str, float]],
    edge_or_cherry: str,
    minimum_distance_for_nontrivial_contact: int,
    output_count_matrices_dir: Optional[str] = None,
    num_processes: int = 1,
    use_cpp_implementation: bool = True,
    cpp_command_line_prefix: str = "",
    cpp_command_line_suffix: str = "",
) -> None:
    """
    Count the number of co-transitions.

    For a tree, an MSA, and a contact map, count the number of co-transitions
    between pairs of non-trivial contacting amino acids at cherries of the
    trees. This computation is aggregated over all the families.

    The computational complexity of this function is as follows. Let:
    - f be the number of families,
    - n be the (average) number of sequences in each family,
    - l be the (average) length of each protein,
    - b be the number of quantization points ('buckets'), and
    - s = len(amino_acids) be the number of amino acids ('states'),
    Then the computational complexity is: O(f * (n * l + b * s^4)).

    Details:
    - Branches whose lengths are smaller than the smallest quantization point,
        or larger than the larger quantization point, are ignored.
    - Only transitions involving valid amino acids are counted.
    - A contact between positions (i, j) is considered non-trivial if
        |i - j| >= minimum_distance_for_nontrivial_contact. Otherwise,
        the contact is considered 'trivial' and ignored when counting.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of (valid) amino acids.
        quantization_points: List of time values used to approximate branch
            lengths.
        edge_or_cherry: Whether to count transitions on edges (which are
            unidirectional), or on cherries (which are bi-directional).
        minimum_distance_for_nontrivial_contact: The minimum distance - in
            terms of indexing - required for an amino acid contact to be
            considered non-trivial.
        output_count_matrices_dir: Directory where to write the count matrices
            to.
        num_processes: Number of processes used to parallelize computation.
        use_cpp_implementation: If to use efficient C++ implementation
            instead of Python.
        cpp_command_line_prefix: E.g. to run the C++ binary on slurm.
        cpp_command_line_suffix: For extra C++ args related to performance.
    """
    if edge_or_cherry.startswith("cherry++__"):
        edge_or_cherry = "cherry++"
    start_time = time.time()

    if not os.path.exists(output_count_matrices_dir):
        os.makedirs(output_count_matrices_dir)
    quantization_points = [float(q) for q in quantization_points]

    if use_cpp_implementation:
        # check if the binary exists
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cpp_path = os.path.join(dir_path, "_count_co_transitions.cpp")
        bin_path = os.path.join(dir_path, "_count_co_transitions")
        # print(f"cpp_path = {cpp_path}")
        if not os.path.exists(bin_path):
            # load openmpi/openmp modules
            # Currently it should run on the interactive node
            command = f"mpicxx -std=c++11 -O3 -o {bin_path} {cpp_path}"
            os.system(command)
            if not os.path.exists(bin_path):
                raise Exception(
                    "Couldn't compile _count_co_transitions.cpp. "
                    f"Command: {command}"
                )
        with tempfile.NamedTemporaryFile("w") as families_file:
            families_path = families_file.name
            open(families_path, "w").write(" ".join(families))
            command = ""
            command += f" {cpp_command_line_prefix}"
            command += f" mpirun -np {num_processes}"
            command += f" {bin_path}"
            command += f" {tree_dir}"
            command += f" {msa_dir}"
            command += f" {contact_map_dir}"
            command += f" {len(families)}"
            command += f" {len(amino_acids)}"
            command += f" {len(quantization_points)}"
            command += f" {families_path}"
            command += " " + " ".join(amino_acids)
            command += " " + " ".join([str(p) for p in quantization_points])
            command += f" {edge_or_cherry}"
            command += f" {minimum_distance_for_nontrivial_contact}"
            command += f" {output_count_matrices_dir}"
            command += f" {cpp_command_line_suffix}"
            logger.info(
                f"Going to run C++ implementation on {len(families)} families "
                f"using {num_processes} ranks. Output dir: "
                f"{output_count_matrices_dir}"
            )
            os.system(command)

            # Remove auxiliary files
            for pid in range(num_processes):
                result_pid_path = os.path.join(
                    output_count_matrices_dir, f"result_{pid}.txt"
                )
                if os.path.exists(result_pid_path):
                    os.remove(result_pid_path)

            logger.info("Done!")
            with open(
                os.path.join(output_count_matrices_dir, "profiling.txt"), "w"
            ) as profiling_file:
                profiling_file.write(
                    f"Total time: {time.time() - start_time} seconds with "
                    f"{num_processes} processes.\n"
                )
            return

    map_args = [
        [
            tree_dir,
            msa_dir,
            contact_map_dir,
            get_process_args(process_rank, num_processes, families),
            amino_acids,
            quantization_points,
            edge_or_cherry,
            minimum_distance_for_nontrivial_contact,
        ]
        for process_rank in range(num_processes)
    ]

    logger.info(
        f"Going to run Python implementation on {len(families)} families"
    )

    # Map step (distribute families among processes)
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            count_matrices_per_process = list(
                tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args))
            )
    else:
        count_matrices_per_process = list(
            tqdm.tqdm(map(_map_func, map_args), total=len(map_args))
        )

    # Reduce step (aggregate count matrices from all processes)
    count_matrices = count_matrices_per_process[0]
    for process_rank in range(1, num_processes):
        for q_idx in range(len(quantization_points)):
            count_matrices[q_idx][1] += count_matrices_per_process[
                process_rank
            ][q_idx][1]

    write_count_matrices(
        count_matrices, os.path.join(output_count_matrices_dir, "result.txt")
    )
    with open(
        os.path.join(output_count_matrices_dir, "profiling.txt"), "w"
    ) as profiling_file:
        profiling_file.write(
            f"Total time: {time.time() - start_time} seconds with "
            f"{num_processes} processes.\n"
        )
