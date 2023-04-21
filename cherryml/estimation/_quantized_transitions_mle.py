import logging
import os
import sys
import tempfile
import time
from typing import Optional

from threadpoolctl import threadpool_limits

from cherryml import caching
from cherryml.io import (
    read_count_matrices,
    read_mask_matrix,
    read_probability_distribution,
    read_rate_matrix,
)

from ._ratelearn import RateMatrixLearner


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
    exclude_args=["device", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"],
    write_extra_log_files=True,
)
def quantized_transitions_mle(
    count_matrices_path: str,
    initialization_path: Optional[str],
    mask_path: Optional[str],
    output_rate_matrix_dir: Optional[str],
    stationary_distribution_path: Optional[str] = None,
    rate_matrix_parameterization: str = "pande_reversible",
    device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 2000,
    do_adam: bool = True,
    loss_normalization: bool = True,
    OMP_NUM_THREADS: Optional[int] = 1,
    OPENBLAS_NUM_THREADS: Optional[int] = 1,
    return_best_iter: bool = True,
):
    start_time = time.time()

    logger = logging.getLogger(__name__)
    logger.info("Starting")

    assert device in ["cpu", "cuda"]
    count_matrices = read_count_matrices(count_matrices_path)
    states = list(count_matrices[0][1].index)
    with tempfile.NamedTemporaryFile("w") as mask2_file:
        # We need to convert the mask matrix to the ratelearn format.
        mask2_path = mask2_file.name
        # mask2_path = "mask2_path.txt"
        if mask_path is not None:
            mask = read_mask_matrix(mask_path)
            mask_array = mask.to_numpy()
            mask_str = ""
            for i in range(mask_array.shape[0]):
                for j in range(mask_array.shape[1]):
                    if j:
                        mask_str += " "
                    mask_str += f"{mask_array[i, j]}"
                mask_str += "\n"
            open(mask2_path, "w").write(mask_str)
        else:
            mask2_path = None

        if stationary_distribution_path is not None:
            stationnary_distribution = read_probability_distribution(
                stationary_distribution_path
            ).to_numpy()
        else:
            stationnary_distribution = None
        if initialization_path is not None:
            initialization = read_rate_matrix(initialization_path).to_numpy()
        else:
            initialization = None

        with threadpool_limits(limits=OPENBLAS_NUM_THREADS, user_api="blas"):
            with threadpool_limits(limits=OMP_NUM_THREADS, user_api="openmp"):
                rate_matrix_learner = RateMatrixLearner(
                    branches=[x[0] for x in count_matrices],
                    mats=[x[1].to_numpy() for x in count_matrices],
                    states=states,
                    output_dir=output_rate_matrix_dir,
                    stationnary_distribution=stationnary_distribution,
                    mask=mask2_path,
                    rate_matrix_parameterization=rate_matrix_parameterization,
                    device=device,
                    initialization=initialization,
                )
                rate_matrix_learner.train(
                    lr=learning_rate,
                    num_epochs=num_epochs,
                    do_adam=do_adam,
                    loss_normalization=loss_normalization,
                    return_best_iter=return_best_iter,
                )

    logger.info("Done!")
    with open(
        os.path.join(output_rate_matrix_dir, "profiling.txt"), "w"
    ) as profiling_file:
        profiling_file.write(
            f"Total time: {time.time() - start_time} seconds with "
            f"{OPENBLAS_NUM_THREADS} OPENBLAS_NUM_THREADS and {OMP_NUM_THREADS}"
            " OMP_NUM_THREADS\n"
        )
