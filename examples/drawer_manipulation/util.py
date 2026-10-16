"""
Utility Functions for Drawer Manipulation with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from aicon_base.math.se3_and_SE3 import homogeneous_transform_inverse, exponential_map_se3
from aicon_real_robots.camera_visibility import visible_likelihood_given_relative_pos


def get_sine_of_angles(relative_pos):
    sin_ax = relative_pos[0] / torch.norm(relative_pos)
    sin_ay = relative_pos[1] / torch.norm(relative_pos)
    sin_az = relative_pos[2] / torch.norm(relative_pos)
    only_angles_sin = torch.concatenate([sin_ax.unsqueeze(0), sin_ay.unsqueeze(0), sin_az.unsqueeze(0)], dim=0)
    return only_angles_sin


def likelihood_func_visible(pose_ee, position_drawer, H_ee_to_cam, steepness: float = 15.0):
    relative_pos = torch.einsum(
        "ki,ij,j->k",
        homogeneous_transform_inverse(H_ee_to_cam),
        homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
        torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]),
    )[:3]
    likelihood = visible_likelihood_given_relative_pos(relative_pos, steepness)
    return likelihood


def likelihood_dist_func(pose_ee, position_drawer, H_ee_to_cam):
    relative_pos = torch.einsum(
        "ki,ij,j->k",
        homogeneous_transform_inverse(H_ee_to_cam),
        homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
        torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]),
    )[:3]
    likelihood = torch.exp(-torch.pow(torch.norm(relative_pos) - 0.5, 2) / 2 / 0.1)
    return likelihood
