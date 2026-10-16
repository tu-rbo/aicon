"""
Goals for the PushT Example

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
from aicon_base.math.angles import shortest_angular_distance
from .env import env_instance


class Push2GoalSim(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="Push2Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):
        goal_pose = torch.tensor(env_instance.info["goal_pose"], dtype=self.dtype, device=self.device)

        def goal_func(object_position):
            cost = torch.sum(
                torch.pow(torch.abs(goal_pose[:2] - object_position[:2]), 2), keepdim=True, dim=0
            ) + 200 * torch.pow(shortest_angular_distance(object_position[2:3], goal_pose[2:3]), 2)
            return cost, cost

        return goal_func
