"""
Parameters for Real-Robot Object Pushing with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import math

NUM_KEYPOINTS: int = 8


UNCERTAINTY_SCALE: float = 30.0


REAL_GOAL_ID: int = 0
REAL_GOAL_BASE: (float, float, float) = (0.5, 0.0, -3.141592653589793)

REAL_GOAL_VARIANTS: dict = {  # goal ID -> (dx [m], dy [m], dtheta [rad]) added to REAL_GOAL_BASE
    0: (0.00, 0.00, 0.0),
    1: (0.2, 0.00, 0.0),
    2: (-0.2, 0.00, 0.0),
    3: (0.00, 0.2, 0.0),
    4: (0.00, -0.2, 0.0),
    5: (-0.15, 0.15, 0.0),
    6: (-0.15, -0.15, 0.0),
    7: (0.00, 0.00, 1.5707963267948966),
    8: (0.00, 0.00, 3.141592653589793),
    9: (0.00, 0.00, 4.71238898038469),
    10: (-0.05, -0.12, 4.71238898038469),
    100: (-0.05, +0.12, 1.5707963267948966),
}


def real_goal_pose(goal_id: int = None) -> (float, float, float):
    """Goal pose for `goal_id` (default REAL_GOAL_ID), with theta wrapped into [-pi, pi)."""
    if goal_id is None:
        goal_id = REAL_GOAL_ID
    if goal_id not in REAL_GOAL_VARIANTS:
        raise ValueError(f"REAL_GOAL_ID {goal_id} is not one of {sorted(REAL_GOAL_VARIANTS)}")
    dx, dy, dth = REAL_GOAL_VARIANTS[goal_id]
    x, y, th = REAL_GOAL_BASE
    th = (th + dth + math.pi) % (2 * math.pi) - math.pi
    return x + dx, y + dy, th


# Throttled, compressed image streams recorded for offline analysis. Kept outside /camera/, which the bag recorder
# excludes.
RECORD_DEPTH_TOPIC: str = "/aicon/recorded/depth_throttled"
RECORD_COLOR_TOPIC: str = "/aicon/recorded/color_throttled"
