"""
Goals for the drawer experiment.
This module defines various goal conditions for the drawer manipulation environment,
including opening drawers, maintaining workspace constraints, and managing uncertainty.
"""

from typing import Union

import torch

from aicon.base_classes.goals import Goal
from aicon.math.util_3d import exponential_map_se3


class DrawerOpenGoal(Goal):
    """
    Goal to open a drawer to a specific position.
    The goal is considered fulfilled when the drawer reaches the target position.
    """

    def __init__(self, is_active: bool, point: Union[torch.tensor, None] = None,
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the DrawerOpenGoal.
        
        Args:
            is_active: Whether the goal is active
            point: Target position for the drawer (defaults to [0.6, 0.435, 0.47])
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        if point is None:
            self.point = torch.tensor([0.6, 0.435, 0.47], dtype=dtype, device=device)
        else:
            self.point = point
        super().__init__(name="ReduceDistToPoint", is_active=is_active, mockbuild=False)

    def define_goal_cost_function(self):
        """
        Define the cost function for opening the drawer.
        
        Returns:
            function: Cost function that takes position_drawer and uncertainty_drawer as input
        """
        goal_point = self.point.clone()

        def goal_func(position_drawer, uncertainty_drawer):
            """
            Calculate the cost based on the distance to the target position.
            
            Args:
                position_drawer: Current position of the drawer
                uncertainty_drawer: Uncertainty in drawer position
            
            Returns:
                tuple: (cost, cost) where cost is the distance to the target position
            """
            dist = torch.sum(torch.abs(goal_point - position_drawer))
            cost = dist
            return cost, cost
        return goal_func


class WorkspaceGoal(Goal):
    """
    Goal to maintain the end-effector within workspace constraints.
    The goal penalizes positions below certain thresholds in the y and z dimensions.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the WorkspaceGoal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="WorkspaceGoal", is_active=is_active, mockbuild=False,
                        device=device, dtype=dtype)

    def define_goal_cost_function(self):
        """
        Define the cost function for workspace constraints.
        
        Returns:
            function: Cost function that takes pose_ee as input
        """
        def goal_func(pose_ee):
            """
            Calculate the cost based on workspace violations.
            
            Args:
                pose_ee: End-effector pose
            
            Returns:
                tuple: (cost, cost) where cost penalizes positions below thresholds
            """
            H = exponential_map_se3(pose_ee)
            t = H[:3, 3]
            cost = torch.zeros(1, device=self.device, dtype=self.dtype)
            if t[2] < 0.2:
                cost = cost - 10 * (t[2] - 0.2)
            if t[1] < 0.2:
                cost = cost - 10 * (t[1] - 0.2)
            return cost, cost
        return goal_func


class DrawerOpenViaJointGoal(Goal):
    """
    Goal to open a drawer to a specific joint value.
    The goal is considered fulfilled when the drawer joint reaches the target value.
    """

    def __init__(self, is_active: bool, open_value: float = 0.5,
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


class DrawerUncertaintyGoal(Goal):
    """
    Goal to reduce uncertainty in the drawer position.
    The goal is considered fulfilled when the uncertainty is minimized.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the DrawerUncertaintyGoal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="MakeDrawerPositionCertain", is_active=is_active,
                        mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        """
        Define the cost function for reducing drawer position uncertainty.
        
        Returns:
            function: Cost function that takes uncertainty_drawer as input
        """
        def goal_func(uncertainty_drawer):
            """
            Calculate the cost based on the trace of the uncertainty matrix.
            
            Args:
                uncertainty_drawer: Uncertainty matrix for drawer position
            
            Returns:
                tuple: (cost, cost) where cost is the trace of the uncertainty matrix
            """
            cost = torch.sum(torch.trace(uncertainty_drawer))
            return cost, cost
        return goal_func


class JointUncertaintyGoal(Goal):
    """
    Goal to reduce uncertainty in the joint state.
    The goal is considered fulfilled when the joint uncertainty is minimized.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the JointUncertaintyGoal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="MakeJointCertain", is_active=is_active,
                        mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        """
        Define the cost function for reducing joint state uncertainty.
        
        Returns:
            function: Cost function that takes uncertainty_joint as input
        """
        def goal_func(uncertainty_joint):
            """
            Calculate the cost based on the trace of the joint uncertainty matrix.
            
            Args:
                uncertainty_joint: Uncertainty matrix for joint state
            
            Returns:
                tuple: (cost, cost) where cost is the trace of the uncertainty matrix
            """
            cost = torch.sum(torch.trace(uncertainty_joint))
            return cost, cost
        return goal_func