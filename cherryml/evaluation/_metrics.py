from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

RateMatrixType = np.array
MaskMatrixType = np.array

from cherryml.global_vars import TITLES


def _masked_log_ratio(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> np.array:
    if y.shape != y_hat.shape:
        raise ValueError(
            "y and y_hat should have the same shape. Shapes are: "
            f"y.shape={y.shape}, y_hat.shape={y_hat.shape}"
        )
    num_states = y.shape[0]
    assert y.shape == (num_states, num_states)
    assert y_hat.shape == (num_states, num_states)
    off_diag_mask = 1 - np.eye(num_states, num_states)
    ratio = y / y_hat
    log_ratio = np.log(ratio)
    masked_log_ratio = log_ratio * off_diag_mask
    if mask_matrix is not None:
        for i in range(num_states):
            for j in range(num_states):
                if mask_matrix[i, j] == 0:
                    masked_log_ratio[i, j] = 0
    return masked_log_ratio


def l_infty_norm(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    masked_log_ratio = _masked_log_ratio(y, y_hat, mask_matrix)
    res = np.max(np.abs(masked_log_ratio))
    return res


def rmse(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    num_states = y.shape[0]
    masked_log_ratio = _masked_log_ratio(y, y_hat, mask_matrix)
    masked_log_ratio_squared = masked_log_ratio * masked_log_ratio
    if mask_matrix is not None:
        total_non_masked_entries = (
            mask_matrix.sum().sum() - num_states
        )  # Need to remove the diagonal
    else:
        total_non_masked_entries = num_states * (num_states - 1)
    res = np.sqrt(np.sum(masked_log_ratio_squared) / total_non_masked_entries)
    return res


def mre(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    """
    Max relative error.
    """
    return np.exp(l_infty_norm(y, y_hat, mask_matrix)) - 1


def relative_error(
    y: float,
    y_hat: float,
) -> float:
    assert y > 0
    assert y_hat > 0
    if y > y_hat:
        return y / y_hat - 1
    else:
        return y_hat / y - 1


def relative_errors(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> np.array:
    """
    Relative errors.
    """
    num_states = y.shape[0]
    if mask_matrix is None:
        mask_matrix = np.ones(
            shape=(num_states, num_states), dtype=int
        ) - np.eye(num_states, dtype=int)
    nonzero_indices = list(zip(*np.where(mask_matrix == 1)))
    nonzero_indices = [(i, j) for (i, j) in nonzero_indices if i != j]
    relative_errors = [
        relative_error(y[i, j], y_hat[i, j]) for (i, j) in nonzero_indices
    ]
    return relative_errors


def mean_relative_error(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    """
    Average relative error.
    """
    return np.mean(
        relative_errors(
            y=y,
            y_hat=y_hat,
            mask_matrix=mask_matrix,
        )
    )


def plot_rate_matrix_predictions(
    y_true: RateMatrixType,
    y_pred: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
    density_plot: bool = False,
    alpha: Optional[float] = 0.3,
) -> None:
    num_states = y_true.shape[0]
    if mask_matrix is None:
        mask_matrix = np.ones(shape=(num_states, num_states), dtype=int)
    nonzero_indices = list(zip(*np.where(mask_matrix == 1)))

    ys_true = [
        np.log(y_true[i, j]) / np.log(10)
        for (i, j) in nonzero_indices
        if i != j
    ]

    ys_pred = [
        np.log(y_pred[i, j]) / np.log(10)
        for (i, j) in nonzero_indices
        if i != j
    ]

    if density_plot:
        sns.jointplot(x=ys_true, y=ys_pred, kind="hex", color="#4CB391")
    else:
        sns.scatterplot(x=ys_true, y=ys_pred, alpha=alpha)
        # plt.scatter(ys_true, ys_pred, alpha=alpha)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if TITLES:
        plt.title("True vs predicted rate matrix entries")
    plt.xlabel(
        "True entry $Q_{" + f"{mask_matrix.shape[0]}" + "}[i, j]$", fontsize=18
    )
    plt.ylabel(
        "Predicted entry $\hat{Q}_{" + f"{mask_matrix.shape[0]}" + "}[i, j]$",
        fontsize=18,
    )  # noqa
    plt.axis("scaled")

    min_y_data = min(ys_true + ys_pred)
    min_y = -7

    ticks = [np.log(10**i) / np.log(10) for i in range(min_y, 1)]
    tickslabels = [f"$10^{{{i}}}$" for i in range(min_y, 1)]
    plt.xticks(ticks, tickslabels)
    plt.yticks(ticks, tickslabels)

    plt.plot([min_y, 0], [min_y, 0], color="r")

    spearman = scipy.stats.spearmanr(ys_true, ys_pred).correlation
    print(f"spearman: {spearman}")
    pearson = scipy.stats.pearsonr(ys_true, ys_pred)[0]
    print(f"pearson: {pearson}")


def plot_rate_matrices_against_each_other(
    y_true: RateMatrixType,
    y_pred: RateMatrixType,
    y_true_name: str,
    y_pred_name: str,
    mask_matrix: Optional[MaskMatrixType] = None,
    density_plot: bool = False,
    min_y: int = -7,
    alpha: float = 0.3,
) -> None:
    """
    Plot "true" vs "predicted" rate matrix. These need not be true vs estimated,
    they can be anything.
    """
    num_states = y_true.shape[0]
    if mask_matrix is None:
        mask_matrix = np.ones(shape=(num_states, num_states), dtype=int)
    nonzero_indices = list(zip(*np.where(mask_matrix == 1)))

    ys_true = [
        np.log(y_true[i, j]) / np.log(10)
        for (i, j) in nonzero_indices
        if i != j
    ]

    ys_pred = [
        np.log(y_pred[i, j]) / np.log(10)
        for (i, j) in nonzero_indices
        if i != j
    ]

    spearman = scipy.stats.spearmanr(ys_true, ys_pred).correlation
    print(f"{y_true_name} vs {y_pred_name} spearman: {spearman}")
    pearson = scipy.stats.pearsonr(ys_true, ys_pred)[0]
    print(f"{y_true_name} vs {y_pred_name} pearson: {pearson}")

    if density_plot:
        sns.jointplot(x=ys_true, y=ys_pred, kind="hex", color="#4CB391")
    else:
        sns.scatterplot(x=ys_true, y=ys_pred, alpha=alpha)
        # plt.scatter(ys_true, ys_pred, alpha=alpha)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if TITLES:
        plt.title("True vs predicted rate matrix entries")
        plt.xlabel(
            "Entry "
            + y_true_name
            + "$_{"
            + f"{mask_matrix.shape[0]}"
            + "}[i, j]$",
            fontsize=18,
        )
        plt.ylabel(
            "Entry "
            + y_pred_name
            + "$_{"
            + f"{mask_matrix.shape[0]}"
            + "}[i, j]$",
            fontsize=18,
        )  # noqa
    plt.axis("scaled")

    min_y_data = min(ys_true + ys_pred)

    ticks = [np.log(10**i) / np.log(10) for i in range(min_y, 1)]
    tickslabels = [f"$10^{{{i}}}$" for i in range(min_y, 1)]
    plt.xticks(ticks, tickslabels)
    plt.yticks(ticks, tickslabels)

    plt.plot([min_y, 0], [min_y, 0], color="r")
