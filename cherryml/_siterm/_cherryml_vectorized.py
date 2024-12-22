import sys
import numpy as np
import time
import torch
from typing import Dict, List, Optional
import logging
import warnings
from threadpoolctl import threadpool_limits
from scipy.linalg import expm


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def _are_almost_equal(a: np.array, b: np.array, decimal: int = 7) -> bool:
    """
    Wrapper around np.testing that returns a boolean
    """
    try:
        np.testing.assert_almost_equal(a, b, decimal=decimal)
        return True
    except AssertionError:
        return False


def solve_stationary_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationary_dist = eigvecs[:, index]
    stationary_dist = stationary_dist / sum(stationary_dist)
    return stationary_dist


def _normalize_rate_matrices(rate_matrices: np.array) -> np.array:
    """
    Normalize rate matrices so that the diagonal elements average to -1.

    Args:
        rate_matrices: 3D array of shape (num_sites, num_states, num_states),
                       where each rate_matrices[i, :, :] is a square matrix.

    Returns:
        normalized_rate_matrices: 3D array of the same shape as rate_matrices,
                                  with normalized diagonal elements.
    """
    # Compute the average of diagonal elements for each matrix
    diagonal_averages = np.mean(np.diagonal(rate_matrices, axis1=1, axis2=2), axis=1)

    # Compute the scaling factor to make the diagonal average -1
    scaling_factors = -1 / diagonal_averages  # Shape: (num_sites,)

    # Scale each matrix by the corresponding factor
    normalized_rate_matrices = rate_matrices * scaling_factors[:, np.newaxis, np.newaxis]

    return normalized_rate_matrices


def solve_stationary_dist_fast(rate_matrices: np.array, device: str) -> np.array:
    """
    Compute the stationary distribution of the given rate matrices using power iteration.

    Args:
        rate_matrices: 3D array of shape num_sites x num_states x num_states. Here
            rate_matrices[i, :, :] is a time-reversible rate matrix.

    Returns:
        stationary_distributions: 2D array of shape num_sites x num_states
    """
    st = time.time()
    rate_matrices = _normalize_rate_matrices(rate_matrices)  # For numerical stability since power iteration will fail if global rate of a rate matrix is arbitrarily small.
    # print(f"rate_matrices = {rate_matrices}")
    num_sites, num_states, _ = rate_matrices.shape
    # Convert to PyTorch tensor for CPU/GPU-accelerated matrix operations
    rate_matrices = torch.tensor(rate_matrices, dtype=torch.float32, device=device)
    # Compute matrix exponentials in a vectorized manner
    exp_matrices = torch.matrix_exp(rate_matrices).detach().cpu().numpy()  # Shape: (num_sites, num_states, num_states)
    # print(f"time exp_matrices: {time.time() - st}")

    st = time.time()
    # Power iteration to converge to stationary distribution
    for _ in range(100):  # Run for a fixed number of iterations
        exp_matrices = exp_matrices @ exp_matrices
        # Normalize each row to sum to 1
        row_sums = exp_matrices.sum(axis=2, keepdims=True)  # Compute row sums
        exp_matrices /= row_sums  # Normalize rows
    stationary_distributions = exp_matrices[:, 0, :]
    stationary_distributions /= stationary_distributions.sum(axis=1, keepdims=True)
    # print(f"time power iteration: {time.time() - st}")

    return stationary_distributions


def quantized_transitions_mle_vectorized_over_sites(
    counts: np.array,
    times: List[float],
    num_epochs: int,
    initialization: Optional[np.array] = None,
    num_cores: int = 1,
    device: str = "cpu",  # New argument to toggle between CPU and GPU
) -> Dict:
    """
    Estimate site-specific rate matrices given their count matrices.

    Letting:
        L: Number of sites.
        B: Number of time buckets (a.k.a. "quantization points")
        N: Number of states.

    This function taked in a `counts` array of shape L x B x N x N and a
    `times` array of shape L x B and estimates site specific rate matrices.

    The loss that this method optimizes is:

    argmin_{Q of size L x N x N where Q[l, :, :] is reversible} 
        -sum_{l=1}^L
            1/sum(counts[l, :, :, :])
                * sum_{b = 1}^B
                    <counts[l, b, :, :], log(mexp(times[l, b] * Q[l, :, :]))>

    For each site, the rate matrix which optimizes the loss is used. (I.e.
    the last iterate is NOT used.)

    Args:
        counts: A numpy array of shape L x B x N x N.
            counts[l, b, :, :] contains the count matrix for site l at time
            index b.
        times: A numpy array of shape L x B. times[l, b] contains the b-th time
            bucket for site l.
        num_epochs: Number of epochs to train for.
        initialization: Of shape L x N x N, the initialization to use.
        num_cores: Constrain the number of used cores (but numpy/torch) to this
            number.
        device: What device to use. Enables running on GPU. NOTE: I have not
            tested this code on GPU so it might not give the intended speedup.

    Returns:
        A dictionary with the following entries:
        "res": Numpy array of size L x N x N with the learnt site specific rate
            matrices.
        "loss_per_epoch": Numpy array of size num_epochs containing the loss
            (i.e. negative composite likelihood) per epoch.
        "loss_per_epoch_per_site": Numpy array of size num_epochs x L
            containing the loss (i.e. negative composite likelihood)
            per epoch and per site.
        "time_...": The time taken by this specific step.
    """
    profiling_res = {}
    st = time.time()
    logger = logging.getLogger(__name__)

    # Ensure the specified device is valid
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"device=cuda requested but device not available."
        )
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    class ReversibleRateMatrices(torch.nn.Module):

        def __init__(self, L, N, initialization):
            super().__init__()
    
            # Initialize theta and Theta
            self.theta = torch.nn.Parameter(0.01 * torch.randn(L, N, device=device))  # L sets of theta
            self.Theta = torch.nn.Parameter(0.01 * torch.randn(L, N, N, device=device))  # L sets of Theta
    
            # Mask to keep only upper triangular part of Theta (excluding diagonal)
            upper_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
            self.register_buffer('upper_mask', upper_mask)  # Buffer to avoid training it
    
            # Initialize parameters based on the given "initialization" array
            if initialization is not None:
                self.initialize_parameters(initialization)

        def initialize_parameters(self, initialization):
            """
            Initialize theta and Theta based on the given rate matrices.

            This function is essentially self-tested, meaning that if the
            assertions in this function pass, it has done the right thing.
            """
            L, N, _ = initialization.shape

            # Step 1: Compute stationary distributions for all rate matrices
            st = time.time()
            pi_all = solve_stationary_dist_fast(rate_matrices=initialization, device="cpu")  # Stationary distributions
            # print(f"time solve_stationary_dist_fast: {time.time() - st}")

            st = time.time()

            # Step 2: Validate stationary distributions
            if not (np.allclose(pi_all.sum(axis=1), 1, atol=1e-3) and np.all(pi_all > 1e-8)):
                raise ValueError("At least one stationary distribution is degenerate.")

            # Step 3: Compute theta = log(pi)
            theta_all = np.log(pi_all)  # Shape: (L, N)

            # Step 4: Compute D_pi, D_inv_pi, and S for all matrices
            sqrt_pi_all = np.sqrt(pi_all)[:, :, None]  # Shape: (L, N, 1)
            inv_sqrt_pi_all = 1.0 / sqrt_pi_all  # Shape: (L, N, 1)
            D_pi = sqrt_pi_all * np.eye(N)[None, :, :]  # Shape: (L, N, N)
            D_inv_pi = inv_sqrt_pi_all * np.eye(N)[None, :, :]  # Shape: (L, N, N)

            S_all = D_pi @ initialization @ D_inv_pi  # Batch computation of S matrices, shape: (L, N, N)

            # Step 5: Validate symmetry of S matrices
            symmetric_diff = np.abs(S_all - S_all.transpose(0, 2, 1))  # Check S - S.T
            if not np.allclose(symmetric_diff, 0, atol=1e-4):
                warnings.warn('At least one S matrix is not symmetric up to 4 decimal places.')

            # Step 6: Compute Theta matrices
            upper_indices = np.triu_indices(N, k=1)  # Indices of the upper triangle
            Theta_all = np.zeros_like(S_all)  # Initialize Theta matrices
            Theta_all[:, upper_indices[0], upper_indices[1]] = np.log(np.exp(S_all[:, upper_indices[0], upper_indices[1]]) - 1)
            Theta_all = (Theta_all + Theta_all.transpose(0, 2, 1)) / 2.0  # Symmetrize Theta

            # Convert to PyTorch tensors and set the parameters
            self.theta = torch.nn.Parameter(torch.tensor(theta_all, dtype=torch.float64, device=device))
            self.Theta = torch.nn.Parameter(torch.tensor(Theta_all, dtype=torch.float64, device=device))

            # Check that we recover the initialization
            np.testing.assert_almost_equal(
                self().detach().cpu().numpy(), initialization, decimal=3
            )
            # print(f"time invert parameterization: {time.time() - st}")

        def forward(self):
            # Parameterize pi for all datasets using softmax along N dimension
            pi = torch.nn.functional.softmax(self.theta, dim=1)  # Shape: L x N
            # print(f"self.theta = {self.theta.detach().numpy()}")
    
            # Parameterize S for all datasets
            symmetric_Theta = self.Theta + self.Theta.transpose(1, 2)  # Symmetrize Theta
            S = torch.nn.functional.softplus(symmetric_Theta) * self.upper_mask  # Zero diagonal
            S = S + S.transpose(1, 2)  # Make S symmetric (L x N x N)
    
            # Compute Q off-diagonal for all datasets
            D_inv_pi = torch.diag_embed(1.0 / pi.sqrt())  # L x N x N (diag(1/pi.sqrt()))
            D_pi = torch.diag_embed(pi.sqrt())  # L x N x N (diag(pi.sqrt()))
            Q_off_diag = D_inv_pi @ S @ D_pi  # Matrix multiplication (L x N x N)
    
            # Set diagonal entries to ensure rows sum to 0 for all datasets
            Q = Q_off_diag.clone()
            row_sums = Q_off_diag.sum(dim=2)  # L x N
            Q -= torch.diag_embed(row_sums)  # Subtract row sums from the diagonal
    
            return Q
    
    def loss_fn(Q, counts, times):
        """
        Compute the normalized loss function for multiple rate matrices with matrix-specific time intervals.
        Args:
            Q: Tensor of shape (L, N, N) containing the rate matrices.
            counts: Tensor of shape (L, B, N, N) containing count matrices.
            times: Tensor of shape (L, B) containing time intervals.
        Returns:
            Loss value as a scalar tensor.
        """
        L, B, N, _ = counts.shape
    
        # Expand times to match dimensions (L, B, 1, 1) for broadcasting
        times_expanded = times.view(L, B, 1, 1)
    
        # Compute matrix exponential for each rate matrix and each batch
        mexp_result = torch.matrix_exp(times_expanded * Q.unsqueeze(1))  # Shape: (L, B, N, N)
    
        # Compute the element-wise logarithm
        log_mexp_result = torch.log(mexp_result)  # Shape: (L, B, N, N)
    
        # Compute the inner product for each matrix and batch
        inner_product = (counts * log_mexp_result).sum(dim=(2, 3))  # Shape: (L, B)
    
        # Sum over batches and normalize by total counts for each dataset
        total_counts = counts.sum(dim=(1, 2, 3))  # Total counts per rate matrix (L,)
        per_site_loss = (-inner_product.sum(dim=1) / total_counts)
        normalized_loss = per_site_loss.sum()  # Scalar loss
    
        return per_site_loss, normalized_loss
    
    with threadpool_limits(limits=num_cores, user_api="blas"),\
        threadpool_limits(limits=num_cores, user_api="openmp"):
            torch.set_num_threads(num_cores)  # Ensure PyTorch respects the limit

            profiling_res["time_preamble"] = time.time() - st
            st = time.time()
            counts = torch.tensor(counts, device=device)
            times = torch.tensor(times, device=device)
            profiling_res["time_send_counts_to_gpu"] = time.time() - st
            st = time.time()

            # Set the random seed for reproducibility
            def set_seed(seed=42):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            # Example usage
            set_seed(42)  # Set seed for deterministic behavior
            
            L = counts.shape[0]  # Number of datasets / rate matrices
            N = counts.shape[-1]  # Size of each rate matrix
            B = counts.shape[1]  # Number of time points / count matrices per dataset

            logger.info(f"Going to estimate site rate matrices for L={L} sites, over N={N} states. Number of time buckets: {B}.")

            # Initialize model
            model = ReversibleRateMatrices(L, N, initialization).to(device)
            
            # Use Adam optimizer with a learning rate of 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            
            profiling_res["time_initialize_model"] = time.time() - st

            # Optimization loop for num_epochs epochs
            st = time.time()
            loss_per_epoch_per_site = np.zeros(shape=(num_epochs, L))
            loss_per_epoch = np.zeros(shape=(num_epochs,))

            # Initialize tensors on the GPU for tracking the best losses and corresponding Q matrices
            loss_best = torch.full((L,), float('inf'), device=device)  # Tensor initialized with infinity
            # Qs_best = torch.zeros((L, N, N), device=device)  # Placeholder for the best Q matrices
            Qs_best = model().detach()


            profiling_res["time_initialize_tensors"] = time.time() - st

            time_zero_grad = 0.0
            time_get_Q = 0.0
            time_compute_loss = 0.0
            time_cpu_loss_analysis = 0.0
            time_backwards = 0.0
            time_optimizer_step = 0.0

            for epoch in range(num_epochs):
                st = time.time()
                optimizer.zero_grad()  # Reset gradients
                time_zero_grad += time.time() - st

                st = time.time()
                Q = model()  # Forward pass
                time_get_Q += time.time() - st

                st = time.time()
                per_site_loss, loss = loss_fn(Q, counts, times)  # Compute loss
                time_compute_loss += time.time() - st

                st = time.time()
                # Update the best loss and corresponding Q matrix per site directly on GPU
                is_better_loss = per_site_loss < loss_best  # Boolean tensor indicating where loss improved
                loss_best = torch.where(is_better_loss, per_site_loss, loss_best)  # Update best loss
                Qs_best = torch.where(
                    is_better_loss.view(-1, 1, 1),  # Expand dimensions for broadcasting
                    Q,  # Update to current Q if loss improved
                    Qs_best  # Keep the previous Q otherwise
                )
                loss_per_epoch_per_site[epoch, :] = per_site_loss.detach().cpu().numpy()
                loss_per_epoch[epoch] = loss.detach().cpu().numpy()
                time_cpu_loss_analysis += time.time() - st

                st = time.time()
                loss.backward()  # Backpropagation
                time_backwards += time.time() - st

                st = time.time()
                optimizer.step()  # Adam optimizer step
                time_optimizer_step += time.time() - st

            profiling_res["time_zero_grad"] = time_zero_grad
            profiling_res["time_get_Q"] = time_get_Q
            profiling_res["time_compute_loss"] = time_compute_loss
            profiling_res["time_cpu_loss_analysis"] = time_cpu_loss_analysis
            profiling_res["time_backwards"] = time_backwards
            profiling_res["time_optimizer_step"] = time_optimizer_step

            # print(f"_cherryml_vectorized per_site_loss: {per_site_loss}")
            logger.info(f"Optimization complete. Time: {time.time() - st}")

            res_dict = {
                "res": Qs_best.detach().cpu().numpy(),  # Only at the end do we get the tensor back to CPU.
                "loss_per_epoch": loss_per_epoch,
                "loss_per_epoch_per_site": loss_per_epoch_per_site,
            }
            res_dict = {**res_dict, **profiling_res}

            return res_dict
