import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

import cherryml.utils


def compute_stationary_distribution(rate_matrix: np.array) -> np.array:
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationary_dist = eigvecs[:, index]
    stationary_dist = stationary_dist / sum(stationary_dist)
    return stationary_dist


def matrix_exponential_pytorch(
    exponents: List[float],
    Q: np.array,
    device: str,
) -> np.array:
    """
    Matrix exponential with Pytorch.
    """
    batch_size = len(exponents)
    num_states = Q.shape[0]
    TQ = np.zeros(shape=(batch_size, num_states, num_states))
    for i in range(batch_size):
        TQ[i, :, :] = exponents[i] * Q
    if device == "cuda":
        # Use striding since tensor might be too large to fit in GPU.
        # Also, striding by 64 does not seem to degrade runtime more than
        # 10% wrt no striding at all.
        res = np.zeros(shape=TQ.shape)
        stride = 64
        for i in range(0, TQ.shape[0], stride):
            res[i : (i + stride), :, :] = (
                torch.matrix_exp(
                    torch.tensor(TQ[i : (i + stride), :, :], device="cuda")
                )
                .cpu()
                .numpy()
            )
        return res
    elif device == "cpu":
        return torch.matrix_exp(torch.tensor(TQ, device="cpu")).numpy()
    else:
        raise Exception(f"Unknown device: {device}")


class FactorizedReversibleModel:
    """
    Factorized Reversible Model.

    Factorizes a rate matrix Q corresponding to a reversible model into matrices
    P2, U, D, U_t, P1 in such a way that the matrix exponential can be computed
    as follows, for any t >= 0:
        exp(t * Q) = P2 @ U @ np.diag(np.exp(length * D)) @ U_t @ P1
    """

    def __init__(self, Q: np.array) -> None:
        eigvals, eigvecs = np.linalg.eig(Q.transpose())
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        eigvals = np.abs(eigvals)
        index = np.argmin(eigvals)
        stationary_dist = eigvecs[:, index]
        stationary_dist = stationary_dist / sum(stationary_dist)

        P1 = np.diag(np.sqrt(stationary_dist))
        P2 = np.diag(np.sqrt(1 / stationary_dist))

        S = P1 @ Q @ P2

        D, U = np.linalg.eigh(S)

        self.P2 = P2
        self.U = U
        self.D = D
        self.U_t = U.transpose()
        self.P1 = P1

    def get_factorization(self):
        return (self.P2, self.U, self.D, self.U_t, self.P1)


def matrix_exponential_reversible(
    exponents: List[float],
    fact: FactorizedReversibleModel,
    device: str,
) -> np.array:
    """
    Matrix exponential for a reversible model.

    Given the factorization `fact` of a rate matrix `Q` and a list of exponents
    `t`, compute a 3D tensor `res` such that:
        `res[i, :, :] = exp(t[i] Q)`
    If the SVD is provided, it will be used; otherwise, the Taylor expansion
    will be used.
    """
    (P2, U, D, U_t, P1) = fact.get_factorization()
    num_states = len(D)
    batch_size = len(exponents)

    expTD = np.zeros(
        shape=(
            batch_size,
            num_states,
            num_states,
        )
    )
    for i in range(batch_size):
        expTD[i, :, :] = np.diag(np.exp(exponents[i] * fact.D))

    if device == "cuda":
        # Use striding since tensor might be too large to fit in GPU.
        stride = 64

        expTQ = np.zeros(shape=(batch_size, num_states, num_states))

        P2_gpu = torch.tensor(P2[None, :, :], device="cuda")
        U_gpu = torch.tensor(U[None, :, :], device="cuda")
        U_t_gpu = torch.tensor(U.transpose()[None, :, :], device="cuda")
        P1_gpu = torch.tensor(P1[None, :, :], device="cuda")

        for i in range(0, expTQ.shape[0], stride):
            expTD_gpu = torch.tensor(
                expTD[i : (i + stride), :, :], device="cuda"
            )
            expTQ[i : (i + stride), :, :] = (
                torch.matmul(
                    torch.matmul(P2_gpu, U_gpu),
                    torch.matmul(
                        expTD_gpu,
                        torch.matmul(
                            U_t_gpu,
                            P1_gpu,
                        ),
                    ),
                )
                .cpu()
                .numpy()
            )
    elif device == "cpu":
        expTQ = (P2[None, :, :] @ U[None, :, :]) @ (
            expTD @ (U_t[None, :, :] @ P1[None, :, :])
        )
    else:
        raise Exception(f"Unknown device: {device}")
    return expTQ


def matrix_exponential(
    exponents: np.array,
    Q: Optional[np.array],
    fact: Optional[FactorizedReversibleModel],
    reversible: bool,
    device: bool,
):
    if reversible:
        return matrix_exponential_reversible(exponents, fact, device)
    else:
        return matrix_exponential_pytorch(exponents, Q, device)


def wag_matrix() -> pd.DataFrame():
    wag = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_rate_matrices/_WAG.txt",
        ),
        index_col=0,
        sep="\t",
    )
    pi = compute_stationary_distribution(wag)
    wag_rate = np.dot(-np.diag(wag), pi)
    res = wag / wag_rate
    assert res.shape == (20, 20)
    return res


def wag_stationary_distribution() -> pd.DataFrame():
    wag = wag_matrix()
    pi = compute_stationary_distribution(wag)
    res = pd.DataFrame(pi, index=wag.index)
    return res


def equ_matrix() -> pd.DataFrame():
    equ = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_rate_matrices/_Equ.txt",
        ),
        index_col=0,
        sep="\t",
    )
    return equ


def equ_stationary_distribution() -> pd.DataFrame():
    pi = [1.0 / 20.0] * 20
    res = pd.DataFrame(pi, index=cherryml.utils.amino_acids)
    return res


def _composite_index(i: int, j: int, num_states: int):
    return i * num_states + j


def chain_product(rate_matrix_1: np.array, rate_matrix_2: np.array) -> np.array:
    assert rate_matrix_1.shape == rate_matrix_2.shape
    num_states = rate_matrix_1.shape[0]
    product_matrix = np.zeros((num_states**2, num_states**2))
    for i in range(num_states):
        for j in range(num_states):
            for k in range(num_states):
                product_matrix[
                    _composite_index(i, k, num_states),
                    _composite_index(i, j, num_states),
                ] = rate_matrix_2[k, j]
                product_matrix[
                    _composite_index(k, j, num_states),
                    _composite_index(i, j, num_states),
                ] = rate_matrix_1[k, i]
    for i in range(num_states):
        for j in range(num_states):
            product_matrix[
                _composite_index(i, j, num_states),
                _composite_index(i, j, num_states),
            ] = (
                rate_matrix_1[i, i] + rate_matrix_2[j, j]
            )
    return product_matrix


def compute_mutation_rate(rate_matrix: np.array) -> float:
    pi = compute_stationary_distribution(rate_matrix)
    mutation_rate = pi @ -np.diag(rate_matrix)
    return mutation_rate


def normalized(rate_matrix: np.array) -> np.array:
    mutation_rate = compute_mutation_rate(rate_matrix)
    res = rate_matrix / mutation_rate
    return res


def get_equ_path():
    return "data/rate_matrices/equ.txt"


def get_jtt_path():
    return "data/rate_matrices/jtt.txt"


def get_wag_path():
    return "data/rate_matrices/wag.txt"


def get_wag_stationary_path():
    return "data/rate_matrices/wag_stationary.txt"


def get_lg_path():
    return "data/rate_matrices/lg.txt"


def get_lg_stationary_path():
    return "data/rate_matrices/lg_stationary.txt"


def get_lg_x_lg_path():
    return "data/rate_matrices/lg_x_lg.txt"


def get_equ_x_equ_path():
    return "data/rate_matrices/equ_x_equ.txt"


def get_lg_x_lg_stationary_path():
    return "data/rate_matrices/lg_x_lg_stationary.txt"


def get_aa_coevolution_mask_path():
    return "data/mask_matrices/aa_coevolution_mask.txt"


def get_coevolution_matrix_path():
    """
    Path to 400 x 400 coevolutionary model from CherryML paper.
    """
    return "data/rate_matrices/coevolution/coevolution.txt"


def get_coevolution_matrix_stationary_path():
    """
    Path to stationary distribution of 400 x 400 coevolutionary model from
    CherryML paper.
    """
    return "data/rate_matrices/coevolution/coevolution_stationary.txt"
