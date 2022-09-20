import contextlib
import os
from typing import List, Optional

import numpy as np

amino_acids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def get_amino_acids() -> List[str]:
    return amino_acids[:]


def quantization_idx(
    branch_length: float, quantization_points_sorted: np.array
) -> Optional[int]:
    if (
        branch_length < quantization_points_sorted[0]
        or branch_length > quantization_points_sorted[-1]
    ):
        return None
    smallest_upper_bound_idx = np.searchsorted(
        quantization_points_sorted, branch_length
    )
    if smallest_upper_bound_idx == 0:
        return 0
    else:
        left_value = quantization_points_sorted[smallest_upper_bound_idx - 1]
        right_value = quantization_points_sorted[smallest_upper_bound_idx]
        relative_error_left = branch_length / left_value - 1
        relative_error_right = right_value / branch_length - 1
        if relative_error_left < relative_error_right:
            return smallest_upper_bound_idx - 1
        else:
            return smallest_upper_bound_idx


def get_process_args(
    process_rank: int, num_processes: int, all_args: List
) -> List:
    process_args = [
        all_args[i]
        for i in range(len(all_args))
        if i % num_processes == process_rank
    ]
    return process_args


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def get_families(
    msa_dir: str,
) -> List[str]:
    """
    Get the list of protein families names.

    Args:
        msa_dir: Directory with the MSA files. There should be one file with
            name family.txt for each protein family.

    Returns:
        The list of protein family names in the provided directory.
    """
    families = sorted(list(os.listdir(msa_dir)))
    families = [x.split(".")[0] for x in families if x.endswith(".txt")]
    return families
