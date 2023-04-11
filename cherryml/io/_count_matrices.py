import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def read_count_matrices(
    count_matrices_path: str,
) -> List[Tuple[float, pd.DataFrame]]:
    res = []
    lines = open(count_matrices_path, "r").read().strip().split("\n")
    line_idx = 0
    num_matrices, s = lines[line_idx].strip().split(" ")
    if s != "matrices":
        raise Exception(
            f"In file {count_matrices_path}, expected line '[num_matrices] "
            f"matrices', but found: '{lines[line_idx]}'"
        )
    num_matrices = int(num_matrices)
    line_idx += 1
    num_states, s = lines[line_idx].strip().split(" ")
    if s != "states":
        raise Exception(
            f"In file {count_matrices_path}, expected line '[num_states] "
            f"states', but found: '{lines[line_idx]}'"
        )
    num_states = int(num_states)
    line_idx += 1
    for _ in range(num_matrices):
        q = float(lines[line_idx])
        line_idx += 1
        states = lines[line_idx].strip().split()
        if len(states) != num_states:
            raise Exception(
                f"Error reading count matrices file: {count_matrices_path}\n"
                f"Expected {num_states} states in line {line_idx}, but instead "
                f"found {len(states)} states: {states}"
            )
        line_idx += 1
        num_states = len(states)
        count_matrix_lines = []
        row_states = []
        for line in range(num_states):
            row_state = lines[line_idx].strip().split()[0]
            row_states.append(row_state)
            count_matrix_line = list(
                map(float, lines[line_idx].strip().split()[1:])
            )
            if len(count_matrix_line) != num_states:
                raise Exception(
                    f"Could not read count matrices. Line: {count_matrix_line}"
                )
            line_idx += 1
            count_matrix_lines.append(count_matrix_line)
        count_matrix_array = np.array(count_matrix_lines)
        count_matrix = pd.DataFrame(
            count_matrix_array,
            index=row_states,
            columns=states,
        )
        res.append((q, count_matrix))
    return res


def write_count_matrices(
    count_matrices: List[Tuple[float, pd.DataFrame]], count_matrices_path: str
) -> None:
    count_matrix_dir = os.path.dirname(count_matrices_path)
    if count_matrix_dir != "" and not os.path.exists(count_matrix_dir):
        os.makedirs(count_matrix_dir)
    num_matrices = len(count_matrices)
    num_states = len(count_matrices[0][1])
    with open(count_matrices_path, "w") as out_file:
        out_file.write(f"{num_matrices} matrices\n{num_states} states\n")
    for q, count_matrix in count_matrices:
        with open(count_matrices_path, "a") as out_file:
            out_file.write(f"{q}\n")
        count_matrix.to_csv(
            count_matrices_path, mode="a", index=True, header=True, sep="\t"
        )
