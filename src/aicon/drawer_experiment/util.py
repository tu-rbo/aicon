"""
Utility functions for the drawer experiment.
This module provides helper functions for processing sensor data and calculating likelihoods
in the drawer manipulation environment.
"""

import torch

from aicon.inference.util import gradient_preserving_clipping
from aicon.math.util_3d import homogeneous_transform_inverse, exponential_map_se3


def get_sine_of_angles(relative_pos: torch.Tensor) -> torch.Tensor:
    """
    Calculate the sine of angles from a relative position vector.
    
    Args:
        relative_pos: 3D position vector [x, y, z]
    
    Returns:
        torch.Tensor: Vector of sine values for each angle [sin(ax), sin(ay), sin(az)]
    """
    sin_ax = relative_pos[0] / torch.norm(relative_pos)
    sin_ay = relative_pos[1] / torch.norm(relative_pos)
    sin_az = relative_pos[2] / torch.norm(relative_pos)
    only_angles_sin = torch.concatenate([sin_ax.unsqueeze(0), sin_ay.unsqueeze(0), sin_az.unsqueeze(0)],
                                        dim=0)
    return only_angles_sin


def likelihood_func_visible(pose_ee: torch.Tensor, position_drawer: torch.Tensor,
                          H_ee_to_cam: torch.Tensor, steepness: float = 15.0) -> torch.Tensor:
    """
    Calculate the likelihood that the drawer is visible in the camera frame.
    This function considers the field of view constraints of the camera.
    
    Args:
        pose_ee: End-effector pose in exponential map coordinates
        position_drawer: Drawer position in world coordinates
        H_ee_to_cam: Homogeneous transform from end-effector to camera frame
        steepness: Steepness parameter for the error function (default: 15.0)
    
    Returns:
        torch.Tensor: Likelihood value between 0 and 1
    """
    relative_pos = torch.einsum("ki,ij,j->k",
                                homogeneous_transform_inverse(H_ee_to_cam),
                                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype,
                                                                       device=position_drawer.device)]))[:3]
    # determine angle from principal camera axis to current relative vector
    if relative_pos[2] <= 0:
        angle_x = torch.pi - torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.pi - torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    else:
        angle_x = torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    # now apply error function depending on how far in/out of FOV it is
    # 0.6 = 69.4 / 180 * PI / 2.0 with 69.4° FOV in x
    # 0.37 = 42.5 / 180 * PI / 2.0 with 42.5° FOV in y
    # clipping 0.1 above to not go into 0 area of the error function
    angle_x = gradient_preserving_clipping(angle_x, 0.0, 0.8)
    likelihood_x = 0.5 - 0.5 * torch.erf((angle_x - 0.6) * steepness)
    angle_y = gradient_preserving_clipping(angle_y, 0.0, 0.57)
    likelihood_y = 0.5 - 0.5 * torch.erf((angle_y - 0.37) * steepness)
    likelihood = likelihood_x * likelihood_y
    return likelihood


def likelihood_dist_func(pose_ee: torch.Tensor, position_drawer: torch.Tensor,
                        H_ee_to_cam: torch.Tensor) -> torch.Tensor:
    """
    Calculate the likelihood based on the distance between the end-effector and drawer.
    
    Args:
        pose_ee: End-effector pose in exponential map coordinates
        position_drawer: Drawer position in world coordinates
        H_ee_to_cam: Homogeneous transform from end-effector to camera frame
    
    Returns:
        torch.Tensor: Likelihood value between 0 and 1
    """
    relative_pos = torch.einsum("ki,ij,j->k",
                                homogeneous_transform_inverse(H_ee_to_cam),
                                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype,
                                                                       device=position_drawer.device)]))[:3]
    likelihood = torch.exp(-torch.pow(torch.norm(relative_pos) - 0.5, 2) / 2 / 0.1)
    return likelihood