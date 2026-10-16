"""
Goals for the 2D Navigation Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.components import Goal


class TargetGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name="TargetGoal",
            is_active=is_active,
            mockbuild=mockbuild,
            fulfilled_value=None,
            device=device,
            dtype=dtype,
        )

    def define_goal_cost_function(self):

        def goal_func(target_pos, target_uncertainty, target_dist):
            cost = target_dist + torch.sum(torch.trace(target_uncertainty)).unsqueeze(0)
            return (cost,), (cost,)

        return goal_func


class ObstacleGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scale=2.0,
    ):
        super().__init__(
            name="ObstacleGoal",
            is_active=is_active,
            mockbuild=mockbuild,
            fulfilled_value=None,
            device=device,
            dtype=dtype,
        )
        self.scale = scale

    def define_goal_cost_function(self):

        def goal_func(current_dist2obs):
            cost = 1.0 / (torch.pow(current_dist2obs, 2) + 0.000000001) * self.scale
            return (cost,), (cost,)

        return goal_func
