"""
Obstacle Map and Start/Target Configuration of the 2D Navigation Example

Coordinate convention: a position p is a continuous 2D point; the cell it occupies is (round(p[0]), round(p[1])), i.e.
integer coordinates are cell centres. Grid value 1 = free, 0 = obstacle.

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import numpy as np

OBS_IMG_SIZE = 30

# where the agent starts, where the target really is, and the initial estimate of the target position
START = (5.0, 5.0)
TARGET = (25.0, 24.0)
TARGET_PRIOR = (23.0, 25.0)


def make_obs():
    """The obstacle map: two walls across the room, each with a gap, inside the outer walls."""
    img = np.ones((OBS_IMG_SIZE, OBS_IMG_SIZE))
    img[2:25, 10] = 0
    img[8:12, 10] = 1
    img[5:28, 18] = 0
    img[18:22, 18] = 1
    # outer walls
    img[:, :2] = 0
    img[:2, :] = 0
    img[:, 28:] = 0
    img[28:, :] = 0
    return img


def make_obs_torch():
    """The obstacle map as a tensor of the default dtype."""
    import torch

    return torch.from_numpy(make_obs()).to(torch.get_default_dtype())
