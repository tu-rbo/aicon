"""
Goals for Drawer Manipulation with AICON

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
from aicon_base.math.se3_and_SE3 import exponential_map_se3


class DrawerOpenGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        point: Union[torch.tensor, None] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        if point is None:
            self.point = torch.tensor([0.6, 0.435, 0.47], dtype=dtype, device=device)
        else:
            self.point = point
        super().__init__(name="ReduceDistToPoint", is_active=is_active, mockbuild=False)

    def define_goal_cost_function(self):
        goal_point = self.point.clone()

        def goal_func(position_drawer, uncertainty_drawer):
            dist = torch.sum(torch.abs(goal_point - position_drawer)).unsqueeze(0)
            cost = dist
            return (cost,), (cost,)

        return goal_func


class WorkspaceGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="WorkspaceGoal", is_active=is_active, mockbuild=False, device=device, dtype=dtype)

    def define_goal_cost_function(self):

        def goal_func(pose_ee):
            H = exponential_map_se3(pose_ee)
            t = H[:3, 3]
            cost = torch.zeros((1,), device=self.device, dtype=self.dtype)
            if t[2] < 0.2:
                cost = cost - 10 * (t[2] - 0.2)
            if t[1] < 0.2:
                cost = cost - 10 * (t[1] - 0.2)
            return (cost,), (cost,)

        return goal_func


class DrawerOpenViaJointGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        open_value: float = 0.5,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.open_value = torch.tensor(open_value, dtype=dtype, device=device)
        super().__init__(name="ReduceJointStateDifference", is_active=is_active, mockbuild=False)

    def define_goal_cost_function(self):
        open_value = self.open_value.clone()

        def goal_func(kinematic_joint):
            dist = torch.sum(torch.abs(kinematic_joint[2:3] - open_value)).unsqueeze(0)
            cost = 1e13 * dist
            return (cost,), (cost,)

        return goal_func


class DrawerUncertaintyGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name="MakeDrawerPositionCertain", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype
        )

    def define_goal_cost_function(self):

        def goal_func(uncertainty_drawer):
            cost = torch.sum(torch.trace(uncertainty_drawer)).unsqueeze(0)
            print(cost)
            return (cost,), (cost,)

        return goal_func


class JointUncertaintyGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="MakeJointCertain", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):

        def goal_func(uncertainty_joint):
            cost = torch.sum(torch.trace(uncertainty_joint)).unsqueeze(0)
            print(cost)
            return (cost,), (cost,)

        return goal_func


class EEPoseGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="ReachEEGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        goal_pose = torch.tensor(
            [
                0.59054894097980292766,
                -0.29536731939005100855,
                0.73576056788895194405,
                -2.22657611704285152143,
                0.13139455261296245636,
                -2.17245522225555109941,
            ]
        )

        def goal_func(pose_ee):
            cost = torch.sum(torch.abs(exponential_map_se3(goal_pose) - exponential_map_se3(pose_ee))).unsqueeze(0)
            print("current cost:", cost)
            return (cost,), (cost,)

        return goal_func


class ObstacleAvoidanceGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name="ObstacleAvoidanceGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype
        )

    def define_goal_cost_function(self):

        def goal_func(likelihood_obstacle_collision):
            cost = 0.01 * torch.mean(likelihood_obstacle_collision).unsqueeze(0)
            return (cost,), (cost,)

        return goal_func
