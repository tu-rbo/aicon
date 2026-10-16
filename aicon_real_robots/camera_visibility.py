"""
Camera Visibility Model for AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from aicon_base.inference.util import gradient_preserving_clipping


def visible_likelihood_given_relative_pos(
    relative_pos,
    steepness,
    steepness_depth=10,
    depth_midpoint=0.25,
    fov_max_x=0.6,
    fov_max_y=0.37,
    fov_adder=0.05,
    combine: str = "product",
):
    """Likelihood that a point at `relative_pos` (camera frame, metres) is in view of the camera.

    Combines a soft field-of-view test per image axis (half-angles `fov_max_x`, `fov_max_y` in rad) with a soft
    minimum-depth test around `depth_midpoint`. `combine` merges the two axes: "product" multiplies them, "min"
    takes the worse one, which keeps the gradient back into view at full strength when both axes are out of view.
    """
    # angles between the camera's principal axis and the point
    mask = relative_pos[..., 2] <= 0
    dist_likelihood_behind = torch.ones_like(relative_pos[..., 2])
    if depth_midpoint is not None:
        dist_likelihood = torch.erf(steepness_depth * (relative_pos[..., 2] - depth_midpoint)) * 0.5 + 0.5
        dist_likelihood[mask] = dist_likelihood_behind[mask]
        angle_x = torch.abs(torch.atan2(relative_pos[..., 0], relative_pos[..., 2]))
        angle_y = torch.abs(torch.atan2(relative_pos[..., 1], relative_pos[..., 2]))
    else:
        dist_likelihood = dist_likelihood_behind
        angle_x = torch.abs(torch.atan2(relative_pos[..., 0], relative_pos[..., 2].detach()))
        angle_y = torch.abs(torch.atan2(relative_pos[..., 1], relative_pos[..., 2].detach()))

    # Defaults match a 69.4 x 42.5 deg field of view. Clipping slightly above it keeps the erf out of its
    # flat tail so the gradient never vanishes.
    angle_x = gradient_preserving_clipping(angle_x, 0.0, fov_max_x + fov_adder)
    likelihood_x = 0.5 - 0.5 * torch.erf((angle_x - fov_max_x) * steepness)
    angle_y = gradient_preserving_clipping(angle_y, 0.0, fov_max_y + fov_adder)
    likelihood_y = 0.5 - 0.5 * torch.erf((angle_y - fov_max_y) * steepness)
    if combine == "product":
        likelihood_xy = likelihood_x * likelihood_y
    elif combine == "min":
        likelihood_xy = torch.minimum(likelihood_x, likelihood_y)
    else:
        raise ValueError("unknown combine rule %r, expected 'product' or 'min'" % combine)
    # point trackers lose points that get too close to the camera
    likelihood = likelihood_xy * dist_likelihood
    return likelihood
