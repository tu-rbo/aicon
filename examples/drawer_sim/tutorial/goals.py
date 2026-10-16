"""
Goals for the Simulated Drawer Examples

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.components import Goal


class DrawerOpenViaJointGoal(Goal):
    """Bring the estimated joint value of the drawer to `open_value` [m] (negative is open)."""

    def __init__(
        self,
        is_active: bool,
        open_value: float,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.open_value = open_value
        super().__init__(
            name="ReduceJointStateDifference", is_active=is_active, dtype=dtype, device=device, mockbuild=mockbuild
        )

    def define_goal_cost_function(self):
        open_value = torch.tensor(self.open_value, dtype=self.dtype, device=self.device)

        def goal_func(kinematic_joint):
            cost = 10.0 * torch.abs(kinematic_joint[2:3] - open_value)
            return cost, cost

        return goal_func
