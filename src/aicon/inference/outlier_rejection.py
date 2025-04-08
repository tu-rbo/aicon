from typing import Tuple

import torch

from aicon.inference.util import make_safe_for_inversion


def reject_outlier(mu_new: torch.Tensor, Sigma_new: torch.Tensor, mu_old: torch.Tensor, Sigma_old: torch.Tensor,
                   Sigma_innovation: torch.Tensor, innovation: torch.Tensor, threshold: float,
                   preserve_gradient: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    mahalanobis = torch.einsum("i,ij,j->", innovation, make_safe_for_inversion(Sigma_innovation, is_batch=False),
                               innovation)
    if mahalanobis < threshold:
        return mu_new, Sigma_new
   
    if preserve_gradient:
        delta_mu = mu_old - mu_new
        delta_sigma = Sigma_old - Sigma_new
        return mu_new + delta_mu.detach(), Sigma_new + delta_sigma.detach()
    return mu_old, Sigma_old
