import os
from typing import List

import numpy as np
import pandas as pd


def write_probability_distribution(
    probability_distribution: np.array,
    states: List[str],
    probability_distribution_path: str,
) -> None:
    probability_distribution_dir = os.path.dirname(
        probability_distribution_path
    )
    if not os.path.exists(probability_distribution_dir):
        os.makedirs(probability_distribution_dir)
    if len(states) != probability_distribution.shape[0]:
        raise Exception(
            f"probability_distribution has shape "
            f"{probability_distribution.shape}, inconsistent with states: "
            f"{states}"
        )
    probability_distribution_df = pd.DataFrame(
        probability_distribution.reshape(-1),
        index=states,
        columns=["prob"],
    )
    probability_distribution_df.index.name = "state"
    probability_distribution_df.to_csv(
        probability_distribution_path,
        sep="\t",
        index=True,
    )


def write_rate_matrix(
    rate_matrix: np.array,
    states: List[str],
    rate_matrix_path: str,
) -> None:
    rate_matrix_dir = os.path.dirname(rate_matrix_path)
    if rate_matrix_dir != "" and not os.path.exists(rate_matrix_dir):
        os.makedirs(rate_matrix_dir)
    rate_matrix_df = pd.DataFrame(
        rate_matrix,
        index=states,
        columns=states,
    )
    rate_matrix_df.to_csv(
        rate_matrix_path,
        sep="\t",
        index=True,
    )


def read_rate_matrix(rate_matrix_path: str) -> pd.DataFrame:
    res = pd.read_csv(
        rate_matrix_path,
        delim_whitespace=True,
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    ).astype(float)
    return res


def read_mask_matrix(mask_matrix_path: str) -> pd.DataFrame:
    res = pd.read_csv(
        mask_matrix_path,
        delim_whitespace=True,
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    ).astype(int)
    return res


def read_probability_distribution(
    probability_distribution_path: str,
) -> pd.DataFrame:
    res = pd.read_csv(
        probability_distribution_path,
        delim_whitespace=True,
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    ).astype(float)
    if res.shape[1] != 1:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should be one-dimensional."
        )
    if abs(res.sum().sum() - 1.0) > 1e-6:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should add to 1.0, with a tolerance of 1e-6."
        )
    return res

def read_computed_cherries_from_file(file_path):
    cherries = []
    distances = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0

        while i < len(lines):
            x = lines[i].strip()
            y = lines[i + 1].strip()
            cherries.append((x, y))

            distance = float(lines[i + 2].strip())
            distances.append(distance)

            i += 3

    return cherries, distances