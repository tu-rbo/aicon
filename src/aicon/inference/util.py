from typing import Union

import numpy as np
import torch


def chi2_df_3_cdf(x):
    return 2.0 * (-0.5 * np.pi**0.5 * (1.0 - torch.erf(torch.sqrt(x) / 2**0.5) ) - torch.exp(- x / 2.0) * torch.sqrt(x) / 2**0.5 + np.pi**0.5 / 2.0) / np.pi**0.5


def chi2_df_2_cdf(x):
    return 1.0 - torch.exp( x / 2.0)


def compute_mahalanobis_dist(x, y, Sigma, is_batch=True):
    if is_batch:
        return torch.sqrt(torch.einsum("bi,bij,bj->b", y-x, torch.inverse(Sigma), y-x))
    else:
        raise NotImplementedError()


def approximate_mixture_of_gaussians_by_gaussian(mu, Sigma, p, is_batch=False):
    if is_batch:
        n_batch, N, Dim = mu.size()
        mu_hat = torch.einsum("bnd,bn->bd", mu, p)
        Sigma_hat = torch.zeros(n_batch, Dim, Dim)

        for j in range(n_batch):
            for i in range(N):
                Sigma_hat[j] += p[j, i] * (Sigma[j, i] + (mu[j, i] - mu_hat[j]).view(-1,1).mm((mu[j, i] - mu_hat[j]).view(1, -1)))
    else:
        N, Dim = mu.size()
        mu_hat = mu.t().mv(p)
        Sigma_hat = torch.zeros(Dim, Dim)

        for i in range(N):
            Sigma_hat += p[i] * (Sigma[i] + (mu[i] - mu_hat).view(-1,1).mm((mu[i] - mu_hat).view(1, -1)))
    return mu_hat, Sigma_hat


def compute_likelihood_gaussians(z, h, R):
    n_models = len(h)
    p = torch.zeros(n_models)
    for i in range(n_models):
        nu = z - h[i](None)
        Q = R[i](None)
        p[i] = torch.det(Q) ** -.5 * torch.exp(-.5 * nu.dot(torch.inverse(Q).mv(nu)))
    return p / torch.sum(p)

def make_safe_for_inversion(matrix, is_batch=True):
    N = matrix.shape[-1]
    i, j = torch.triu_indices(N, N)
    new_matrix = matrix.clone()
    if is_batch:
        new_matrix[:, i, j] =  torch.transpose(matrix, dim0=-2, dim1=-1)[:, i, j]
        new_matrix = new_matrix + torch.eye(N, dtype=matrix.dtype, device=matrix.device).unsqueeze(0).repeat(new_matrix.shape[0], 1, 1) * 0.00000000001
    else:
        new_matrix.T[i, j] = matrix[i, j]
        new_matrix = new_matrix + torch.eye(N, dtype=matrix.dtype, device=matrix.device) * 0.00000000001

    return new_matrix


def gradient_preserving_clipping(t : torch.Tensor, minimum : Union[None, float] = None, maximum : Union[None, float] = None):
    delta = torch.zeros_like(t)
    delta[t < minimum] = minimum - t[t < minimum]
    delta[t > maximum] = maximum - t[t > maximum]
    return t + delta.detach()
