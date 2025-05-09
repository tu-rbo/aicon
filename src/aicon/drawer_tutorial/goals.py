"""
Goals for the drawer experiment.
This module defines various goal conditions for the drawer manipulation environment,
including opening drawers, maintaining workspace constraints, and managing uncertainty.
"""

from typing import Union

import torch

from aicon.base_classes.goals import Goal
from aicon.math.util_3d import exponential_map_se3



class DrawerOpenViaJointGoal(Goal):
    """
    Goal to open a drawer to a specific joint value.
    The goal is considered fulfilled when the drawer joint reaches the target value.
    """

    def __init__(self, is_active: bool, open_value: float = -0.5,
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the DrawerOpenViaJointGoal.
        
        Args:
            is_active: Whether the goal is active
            open_value: Target joint value for the drawer (defaults to 0.5)
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        self.open_value = torch.tensor(open_value, dtype=dtype, device=device)
        super().__init__(name="ReduceJointStateDifference", is_active=is_active, mockbuild=False)

    def define_goal_cost_function(self):
        """
        Define the cost function for opening the drawer via joint control.
        
        Returns:
            function: Cost function that takes kinematic_joint as input
        """
        open_value = self.open_value.clone()

        def goal_func(kinematic_joint):
            """
            Calculate the cost based on the difference from target joint value.
            
            Args:
                kinematic_joint: Current joint state
            
            Returns:
                tuple: (cost, cost) where cost is the scaled difference from target
            """
            dist = 10.0 * torch.sum(torch.abs(kinematic_joint[2:3] - open_value))
            cost = dist
            return cost, cost
        return goal_func


