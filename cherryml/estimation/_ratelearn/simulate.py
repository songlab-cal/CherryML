import numpy as np
import pandas as pd
import torch
import torch.distributions as db
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def generate_data(
    Q_true: torch.Tensor,
    m: int,
    batch_size: int = 1000,
    distribution: str = "exp",
    lower_unif: float = None,
    upper_unif: float = None,
    quantile: float = 10,
    alpha: float = 0.05,
    exact_matrix_exp: bool = True,
    device="cuda",
    pi=None,
):
    """Generate triplet observations associated to a given rate matrix

    Parameters
    ----------
    Q_true : Rate matrix
    m : Number of datapoints
    batch_size : int, Batch-size used to generate data, optional

    Returns
    -------
    torch.tensor
        Of shape (number of observations, 3)
        The three dimensions respectively correspond to
            1. the start state
            2. the ending state
            3. the branch length
    """
    Q_np = Q_true.cpu().numpy()
    d, u = np.linalg.eigh(Q_np)

    d = torch.tensor(d, device=device)
    u = torch.tensor(u, device=device)
    Q_true = torch.tensor(Q_true, device=device)
    assert Q_true.ndim == 2
    num_states = Q_true.shape[-1]
    print(
        f"Generating {m} synthetic datapoints of the form "
        "(starting_state, ending_state, branch_length)"
    )
    # branch_lengths = 0.05 * torch.rand(m)
    if distribution == "exp":
        rate = -np.log(alpha) / quantile
        print("rate", rate)
        branch_lengths = db.Exponential(rate).sample((m,))
    elif distribution == "unif":
        branch_lengths = db.Uniform(low=lower_unif, high=upper_unif).sample(
            (m,)
        )
    elif distribution == "logunif":
        branch_lengths = (
            db.Uniform(low=lower_unif, high=upper_unif).sample((m,)).exp()
        )
    elif distribution == "constant":
        branch_lengths = quantile * torch.ones(m).long()
    else:
        raise ValueError
    assert branch_lengths.ndim == 1
    starting_state = torch.randint(0, num_states, (m,))
    if pi is not None:
        starting_state = db.Categorical(probs=pi).sample((m,))
    _data = TensorDataset(starting_state, branch_lengths)
    dl = DataLoader(_data, batch_size=batch_size, shuffle=False)
    ending_state = []
    for tensors in tqdm(dl):
        starting_state_, branch_length_ = tensors
        starting_state_ = starting_state_.to(device)
        branch_length_ = branch_length_.to(device)
        if exact_matrix_exp:
            diag_term = (branch_length_[:, None] * d).exp()
            diag_term = torch.diag_embed(diag_term)
            # print(u.shape, diag_term.shape)
            transition_probs_from_starting_state = u @ (diag_term @ u.T)
            transition_probs_from_starting_state2 = torch.matrix_exp(
                branch_length_[:, None, None] * Q_true
            )
            err = (
                (
                    transition_probs_from_starting_state2
                    - transition_probs_from_starting_state
                )
                .abs()
                .max(1)
                .values.max(1)
                .values
            )
            id_branch_max = err.argmax()
            id_branch_min = err.argmin()
            print("max", branch_length_[id_branch_max], err.max())
            print("min", branch_length_[id_branch_min], err.min())

        else:
            transition_probs_from_starting_state = torch.matrix_exp(
                branch_length_[:, None, None] * Q_true
            )
        transition_probs_from_starting_state = (
            transition_probs_from_starting_state[
                torch.arange(len(starting_state_)), starting_state_
            ]
        )
        transition_probs_from_starting_state = nn.ReLU()(
            transition_probs_from_starting_state
        )
        ending_state_ = db.Categorical(
            transition_probs_from_starting_state
        ).sample()
        ending_state.append(ending_state_.cpu())
    ending_state = torch.cat(ending_state)
    tdata = torch.cat(
        [
            starting_state[:, None],
            ending_state[:, None],
            branch_lengths[:, None],
        ],
        1,
    )
    return tdata


def convert_triplet_to_quantized(
    tdata: torch.Tensor, num_states: int, q: int = 100
):
    """Converts an observational dataset to a quantized representation

    Parameters
    ----------
    tdata : torch.Tensor
        Shape (n observations, 3)
    num_states : int
        Number of states in total
    q : int, optional
        Number of desired quantiles, by default 100

    Returns
    -------
    TensorDataset
        (Quantized time, frequency matrix)
    """
    # First step: Attribute pseudotime to each observation
    df = pd.Series(tdata[:, 2].numpy()).to_frame("time")
    df.loc[:, "bins"] = pd.qcut(df["time"], q=q)
    gped = df.groupby("bins")["time"]
    df.loc[:, "qmin"] = gped.transform(lambda x: np.quantile(x, q=0.25))
    df.loc[:, "qmax"] = gped.transform(lambda x: np.quantile(x, q=0.75))
    df.loc[:, "qtime"] = gped.transform("mean")
    df.loc[:, "idx1"] = tdata[:, 0].long().numpy()
    df.loc[:, "idx2"] = tdata[:, 1].long().numpy()

    def get_cmatrix(my_df: pd.DataFrame):
        """Construct frequency matrix"""
        idx_vals = (
            my_df.groupby(["idx1", "idx2"]).size().to_frame("val").reset_index()
        )
        idx1 = torch.tensor(idx_vals.idx1.values)
        idx2 = torch.tensor(idx_vals.idx2.values)
        idx = torch.cat([idx1[None], idx2[None]], 0)
        vals = torch.tensor(idx_vals.val)
        cmat = torch.sparse_coo_tensor(
            idx, vals, (num_states, num_states)
        ).to_dense()[None]
        return cmat

    # Second step: Compute frequency matrices
    qtime_to_mats = df.groupby(["qtime", "qmin", "qmax"]).apply(get_cmatrix)
    qtime_to_mats = qtime_to_mats.to_frame("cmat").reset_index()

    # Last step: return modified dataset
    qtimes = torch.tensor(qtime_to_mats.qtime.values)
    cmats = torch.tensor(np.concatenate(qtime_to_mats.cmat.values))
    print(qtimes.shape, cmats.shape)
    return TensorDataset(qtimes, cmats)
