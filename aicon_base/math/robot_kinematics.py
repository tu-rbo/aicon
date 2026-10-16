"""
Robot Kinematics Functions

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch


def dh_params_to_htransform_batch(alpha, a, d, theta):
    cos_alpha = torch.cos(alpha).unsqueeze(-1)
    sin_alpha = torch.sin(alpha).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    a_unsqueezed = a.unsqueeze(-1)
    d_unsqueezed = d.unsqueeze(-1)
    first_row = torch.cat([cos_theta, -sin_theta, torch.zeros_like(a_unsqueezed), a_unsqueezed], dim=-1)
    second_row = torch.cat(
        [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d_unsqueezed * sin_alpha], dim=-1
    )
    third_row = torch.cat([sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d_unsqueezed * cos_alpha], dim=-1)
    fourth_row = torch.cat(
        [
            torch.zeros_like(a_unsqueezed),
            torch.zeros_like(a_unsqueezed),
            torch.zeros_like(a_unsqueezed),
            torch.ones_like(a_unsqueezed),
        ],
        dim=-1,
    )
    transform = torch.cat(
        [first_row.unsqueeze(-2), second_row.unsqueeze(-2), third_row.unsqueeze(-2), fourth_row.unsqueeze(-2)], dim=-2
    )
    return transform
