"""
Geometry Helpers for the Simulated Drawer Examples

Poses of the end-effector are [x, y, z, rx, ry, rz] with the orientation as a rotation vector (axis * angle).

Authors: Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from aicon_base.inference.util import gradient_preserving_clipping
from aicon_base.math.se3_and_SE3 import homogeneous_transform_inverse

# end-effector -> camera transform of the simulated wrist camera
H_EE_TO_CAM = [[1.0, 0.0, 0.0, -0.08], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def ee_to_camera(dtype, device) -> torch.Tensor:
    return torch.tensor(H_EE_TO_CAM, dtype=dtype, device=device)


def pose_vec_to_homogeneous(pose_ee: torch.Tensor) -> torch.Tensor:
    """4x4 homogeneous transform of a pose [x, y, z, rx, ry, rz] (Rodrigues' formula)."""
    t = pose_ee[:3]
    rotvec = pose_ee[3:]
    theta = torch.norm(rotvec)
    k = rotvec / torch.clamp(theta, min=torch.tensor(1e-10, dtype=pose_ee.dtype, device=pose_ee.device))
    zero = torch.zeros((), dtype=pose_ee.dtype, device=pose_ee.device)
    K = torch.stack(
        [torch.stack([zero, -k[2], k[1]]), torch.stack([k[2], zero, -k[0]]), torch.stack([-k[1], k[0], zero])], dim=0
    )
    R = (
        torch.eye(3, dtype=pose_ee.dtype, device=pose_ee.device)
        + torch.sin(theta) * K
        + (1.0 - torch.cos(theta)) * torch.mm(K, K)
    )
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose_ee.dtype, device=pose_ee.device)
    return torch.cat([torch.cat([R, t.unsqueeze(1)], dim=1), bottom], dim=0)


def point_in_camera_frame(pose_ee: torch.Tensor, point: torch.Tensor, H_ee_to_cam: torch.Tensor) -> torch.Tensor:
    """A world point expressed in the camera frame of an end-effector pose."""
    return torch.einsum(
        "ki,ij,j->k",
        homogeneous_transform_inverse(H_ee_to_cam),
        homogeneous_transform_inverse(pose_vec_to_homogeneous(pose_ee)),
        torch.cat([point, torch.ones(1, dtype=point.dtype, device=point.device)]),
    )[:3]


def bearing(relative_pos: torch.Tensor) -> torch.Tensor:
    """Unit direction to a point (what a camera without depth measures)."""
    return relative_pos / torch.norm(relative_pos)


def likelihood_func_visible(
    pose_ee: torch.Tensor, position_drawer: torch.Tensor, H_ee_to_cam: torch.Tensor, steepness: float = 15.0
) -> torch.Tensor:
    """Likelihood that the drawer is inside the camera's field of view (69.4 x 42.5 degrees), smoothed by an error
    function of the angle from the optical axis."""
    relative_pos = point_in_camera_frame(pose_ee, position_drawer, H_ee_to_cam)
    if relative_pos[2] <= 0:  # behind the camera
        angle_x = torch.pi - torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.pi - torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    else:
        angle_x = torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    # half field of view 0.37 rad (42.5 deg) in x and 0.6 rad (69.4 deg) in y; clipped 0.2 rad outside it so the
    # error function stays away from its flat zero region
    angle_x = gradient_preserving_clipping(angle_x, 0.0, 0.57)
    likelihood_x = 0.5 - 0.5 * torch.erf((angle_x - 0.37) * steepness)
    angle_y = gradient_preserving_clipping(angle_y, 0.0, 0.8)
    likelihood_y = 0.5 - 0.5 * torch.erf((angle_y - 0.6) * steepness)
    return likelihood_x * likelihood_y


def likelihood_dist_func(
    pose_ee: torch.Tensor, position_drawer: torch.Tensor, H_ee_to_cam: torch.Tensor
) -> torch.Tensor:
    """Likelihood of a good view by distance: highest at 0.5 m from the camera."""
    relative_pos = point_in_camera_frame(pose_ee, position_drawer, H_ee_to_cam)
    return torch.exp(-torch.pow(torch.norm(relative_pos) - 0.5, 2) / 2 / 0.1)
