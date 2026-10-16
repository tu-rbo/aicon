"""
Angle Utilities for AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import math

import torch


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle into (-pi, pi]. Same arithmetic as ROS' ``angles.normalize_angle``, but elementwise."""
    a = angle % (2.0 * math.pi)
    return torch.where(a > math.pi, a - 2.0 * math.pi, a)


def shortest_angular_distance(from_angle: torch.Tensor, to_angle: torch.Tensor) -> torch.Tensor:
    """Signed shortest rotation from ``from_angle`` to ``to_angle``, in (-pi, pi]."""
    return normalize_angle(to_angle - from_angle)
