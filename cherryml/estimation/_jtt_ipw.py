import logging
import os
import sys
import time
from typing import Optional

import numpy as np

from cherryml import caching
from cherryml.io import read_count_matrices, read_mask_matrix, write_rate_matrix
from cherryml.markov_chain import normalized


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
    write_extra_log_files=True,
)
def jtt_ipw(
    count_matrices_path: str,
    mask_path: Optional[str],
    use_ipw: bool,
    output_rate_matrix_dir: str,
    normalize: bool = False,
    max_time: Optional[float] = None,
    pseudocounts: float = 1e-8,
    symmetrize_count_matrices: bool = True,
) -> None:
    """
    JTT-IPW estimator.

    Args:
        max_time: Only data from transitions with length <= max_time will be
            used to compute the estimator. The estimator works best on short
            transitions, which poses a bias-variance tradeoff.
    """
    start_time = time.time()

    logger = logging.getLogger(__name__)
    logger.info("Starting")

    # Open frequency matrices
    count_matrices = read_count_matrices(count_matrices_path)
    states = list(count_matrices[0][1].index)
    num_states = len(states)

    if mask_path is not None:
        mask_mat = read_mask_matrix(mask_path).to_numpy()
    else:
        mask_mat = np.ones(shape=(num_states, num_states))

    qtimes, cmats = zip(*count_matrices)
    del count_matrices
    qtimes = list(qtimes)
    cmats = list(cmats)
    if max_time is not None:
        valid_time_indices = [
            i for i in range(len(qtimes)) if qtimes[i] <= max_time
        ]
        qtimes = [qtimes[i] for i in valid_time_indices]
        cmats = [cmats[i] for i in valid_time_indices]
    cmats = [(cmat.to_numpy() + pseudocounts) for cmat in cmats]

    n_time_buckets = len(cmats)
    assert cmats[0].shape == (num_states, num_states)
    if symmetrize_count_matrices:
        # Coalesce transitions a->b and b->a together
        for i in range(n_time_buckets):
            cmats[i] = (cmats[i] + np.transpose(cmats[i])) / 2.0
    # Apply masking
    for i in range(n_time_buckets):
        cmats[i] = cmats[i] * mask_mat

    # Compute CTPs
    # Compute total frequency matrix (ignoring branch lengths)
    F = sum(cmats)
    # Zero the diagonal such that summing over rows will produce the number of
    # transitions from each state.
    F_off = F * (1.0 - np.eye(num_states))
    # Compute CTPs
    CTPs = F_off / (F_off.sum(axis=1)[:, None])

    # Compute mutabilities
    if use_ipw:
        M = np.zeros(shape=(num_states))
        for i in range(n_time_buckets):
            qtime = qtimes[i]
            cmat = cmats[i]
            cmat_off = cmat * (1.0 - np.eye(num_states))
            M += 1.0 / qtime * cmat_off.sum(axis=1)
        M /= F.sum(axis=1)
    else:
        M = 1.0 / np.median(qtimes) * F_off.sum(axis=1) / (F.sum(axis=1))

    # JTT-IPW estimator
    res = np.diag(M) @ CTPs
    np.fill_diagonal(res, -M)

    if normalize:
        res = normalized(res)

    write_rate_matrix(
        res, states, os.path.join(output_rate_matrix_dir, "result.txt")
    )

    logger.info("Done!")
    with open(
        os.path.join(output_rate_matrix_dir, "profiling.txt"), "w"
    ) as profiling_file:
        profiling_file.write(
            f"Total time: {time.time() - start_time} seconds\n"
        )
