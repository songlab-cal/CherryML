from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn as nn


def solve_stationery_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist


def _are_almost_equal(a: np.array, b: np.array, decimal: int = 7) -> bool:
    """
    Wrapper around np.testing that returns a boolean
    """
    try:
        np.testing.assert_almost_equal(a, b, decimal=decimal)
        return True
    except AssertionError:
        return False


class RateMatrix(nn.Module):
    def __init__(
        self,
        num_states,
        mode,
        mask: torch.tensor,
        pi=None,
        pi_requires_grad=False,
        initialization: Optional[np.array] = None,
    ):
        super().__init__()
        if pi is not None:
            assert pi.ndim == 1
            pi_logits = torch.log(pi)
            self._pi = nn.parameter.Parameter(
                pi_logits.clone(), requires_grad=pi_requires_grad
            )
        self.num_states = num_states
        self.mode = mode
        nparams_half = int(0.5 * num_states * (num_states - 1))
        self.upper_diag = nn.parameter.Parameter(
            0.01 * torch.randn(nparams_half, requires_grad=True)
        )
        if mode in ["default", "stationary", "pande"]:
            self.lower_diag = nn.parameter.Parameter(
                0.01 * torch.randn(nparams_half, requires_grad=True)
            )
        self.activation = nn.Softplus()
        self.mask = mask

        if initialization is not None and mode == "pande_reversible":
            # Initialize upper_diag and pi.
            # Need to decompose the initialization into pande_reversible
            # components: 1/sqrt(pi) * S * sqrt(pi)
            # Need to be careful about inverting the activation functions.
            pi = solve_stationery_dist(initialization)
            if np.any(np.abs(pi) < 1e-8):
                raise ValueError(
                    "Stationary distribution of initialization is degenerate."
                )
            if np.any(
                np.abs(mask.numpy() * initialization - initialization) > 1e-8
            ):
                raise ValueError("initialization not compatible with mask")
            pi_inv_mat = np.diag(1.0 / np.sqrt(pi))
            pi_mat = np.diag(np.sqrt(pi))
            assert pi_inv_mat.shape == (num_states, num_states)
            assert pi_mat.shape == (num_states, num_states)
            S = pi_mat @ initialization @ pi_inv_mat
            if not _are_almost_equal(S, np.transpose(S), decimal=4):
                warnings.warn('S and its transpose are not almost equal up to 4 decimal places.')
            vals = [
                np.log(np.exp(S[i, j]) - 1)
                for i in range(num_states)
                for j in range(i + 1, num_states)
            ]
            self._pi.data.copy_(torch.tensor(np.log(pi)))
            self.upper_diag.data.copy_(torch.tensor(vals))
            np.testing.assert_almost_equal(
                self().detach().numpy(), initialization, decimal=3
            )
        elif initialization:
            raise ValueError(
                f"Parameter initialization not implemented for mode {mode}"
            )

    # @property
    # def pi(self):
    #     return nn.Softmax()(self._pi)

    # @property
    # def pi_mat(self):
    #     return torch.diag(self.pi)

    def forward(self):
        device = self.upper_diag.device
        if self.mode == "default":
            mat = torch.zeros(
                self.num_states,
                self.num_states,
                device=device,
            )
            triu_indices = torch.triu_indices(
                row=self.num_states,
                col=self.num_states,
                offset=1,
                device=device,
            )
            mat[triu_indices[0], triu_indices[1]] = self.activation(
                self.upper_diag
            )
            tril_indices = torch.tril_indices(
                row=self.num_states,
                col=self.num_states,
                offset=-1,
                device=device,
            )
            mat[tril_indices[0], tril_indices[1]] = self.activation(
                self.lower_diag
            )
            mat = mat * self.mask
            mat = mat - torch.diag(mat.sum(1))

        if self.mode in ["stationary_reversible", "stationary"]:
            rmat_off = torch.zeros(
                self.num_states, self.num_states, device=device
            )
            triu_indices = torch.triu_indices(
                row=self.num_states,
                col=self.num_states,
                offset=1,
                device=device,
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(
                self.upper_diag
            )
            if self.mode == "stationary_reversible":
                rmat_off = rmat_off + rmat_off.T
            elif self.mode == "stationary":
                tril_indices = torch.tril_indices(
                    row=self.num_states,
                    col=self.num_states,
                    offset=-1,
                    device=device,
                )
                rmat_off[tril_indices[0], tril_indices[1]] = self.activation(
                    self.lower_diag
                )
            rmat_off = rmat_off * self.mask
            pi = nn.Softmax()(self._pi)
            pi_mat = torch.diag(pi)
            rmat_diag = -(rmat_off @ pi) / pi
            rmat = rmat_off + torch.diag(rmat_diag)

            mat = rmat @ pi_mat

        if self.mode == "pande_reversible":
            self.mask = self.mask.to(device=device)
            rmat_off = torch.zeros(
                self.num_states, self.num_states, device=device
            )
            triu_indices = torch.triu_indices(
                row=self.num_states,
                col=self.num_states,
                offset=1,
                device=device,
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(
                self.upper_diag
            )
            rmat_off = rmat_off + rmat_off.T
            rmat_off = rmat_off * self.mask

            pi = nn.Softmax(dim=-1)(self._pi)
            pi_mat = torch.diag(pi.sqrt())
            pi_inv_mat = torch.diag(pi.sqrt() ** (-1))
            mat = (pi_inv_mat @ rmat_off) @ pi_mat
            mat -= torch.diag(mat.sum(1))

        if self.mode == "pande":
            rmat_off = torch.zeros(
                self.num_states, self.num_states, device=device
            )
            triu_indices = torch.triu_indices(
                row=self.num_states,
                col=self.num_states,
                offset=1,
                device=device,
            )
            rmat_off[triu_indices[0], triu_indices[1]] = self.activation(
                self.upper_diag
            )
            tril_indices = torch.tril_indices(
                row=self.num_states,
                col=self.num_states,
                offset=-1,
                device=device,
            )
            rmat_off[tril_indices[0], tril_indices[1]] = self.activation(
                self.lower_diag
            )
            rmat_off = rmat_off * self.mask

            pi = nn.Softmax(-1)(self._pi)
            pi_mat = torch.diag(pi.sqrt())
            pi_inv_mat = torch.diag(pi.sqrt() ** (-1))
            mat = (pi_inv_mat @ rmat_off) @ pi_mat
            mat -= torch.diag(mat.sum(1))
            # mat += torch.eye(self.num_states, device=mat.device) *
        return mat
