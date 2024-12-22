import numpy as np
import pandas as pd

from typing import List
from cherryml import markov_chain


def _condition_on_non_gap(conditional_probability_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: Assumes that the gap state is the last one in the alphabet!
    """
    if conditional_probability_matrix.columns[-1] != "-":
        raise ValueError(
            "It is assumed that the gap state is the last one! "
            "Last state was instead: "
            f"{conditional_probability_matrix.columns[-1]}"
        )

    data = conditional_probability_matrix.values.copy()
    row_sums = np.sum(data[:, :-1], axis=1, keepdims=True)
    data[:, :-1] /= row_sums
    data[:, -1] = 1.0

    res = pd.DataFrame(
        data,
        index=conditional_probability_matrix.index,
        columns=conditional_probability_matrix.columns
    )
    return res


def _test_condition_on_non_gap():
    mexp_df = pd.DataFrame(
        np.array(
            [
                [0.3, 0.2, 0.5],
                [0.4, 0.2, 0.4],
                [0.1, 0.1, 0.8]
            ]
        ),
        index=["A", "B", "-"],
        columns=["A", "B", "-"],
    )
    cpm_no_gaps = _condition_on_non_gap(
        mexp_df
    )
    pd.testing.assert_frame_equal(
        cpm_no_gaps,
        pd.DataFrame(
            np.array(
                [
                    [0.6, 0.4, 1.0],
                    [2/3, 1/3, 1.0],
                    [0.5, 0.5, 1.0]
                ]
            ),
            index=["A", "B", "-"],
            columns=["A", "B", "-"],
        )
    )
    assert(cpm_no_gaps.loc["A", "-"] == 1.0)


def matrix_exponential_reversible(
    rate_matrix: np.array,
    exponents: List[float],
) -> np.array:
    """
    Compute matrix exponential (batched).

    Args:
        rate_matrix: Rate matrix for which to compute the matrix exponential
        exponents: List of exponents.
    Returns:
        3D tensor where res[:, i, i] contains exp(rate_matrix * exponents[i])
    """
    return markov_chain.matrix_exponential_reversible(
        exponents=exponents,
        fact=markov_chain.FactorizedReversibleModel(rate_matrix),
        device="cpu",
    )


if __name__ == "__main__":
    _test_condition_on_non_gap()
