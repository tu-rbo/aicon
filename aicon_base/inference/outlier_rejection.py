"""
Outlier Rejection for Gaussian Beliefs

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Tuple

import torch


def reject_outlier(
    mu_new: torch.Tensor,
    Sigma_new: torch.Tensor,
    mu_old: torch.Tensor,
    Sigma_old: torch.Tensor,
    innovation: torch.Tensor,
    Sigma_inv_innovation: torch.Tensor,
    threshold: float,
    preserve_gradient: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mahalanobis = torch.sqrt(torch.einsum("i,ij,j->", innovation, Sigma_inv_innovation, innovation))
    if mahalanobis < threshold:
        return mu_new, Sigma_new

    if preserve_gradient:
        delta_mu = mu_old - mu_new
        delta_sigma = Sigma_old - Sigma_new
        return mu_new + delta_mu.detach(), Sigma_new + delta_sigma.detach()
    return mu_old, Sigma_old
