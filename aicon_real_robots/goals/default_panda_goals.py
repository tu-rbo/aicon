from typing import Union

import torch

from aicon_base.components import Goal


class JointLimitOnStateGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="JointLimitGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        joint_limits_low = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=self.dtype, device=self.device
        )
        joint_limits_high = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=self.dtype, device=self.device
        )
        interval = 0.2

        def goal_func(robot_joint_states):
            joint_limits_low_cost = joint_limits_low - robot_joint_states
            joint_limits_high_cost = robot_joint_states - joint_limits_high
            in_interval_low = joint_limits_low_cost > -interval
            in_interval_high = joint_limits_high_cost > -interval
            cost = 1e15 * torch.sum(
                joint_limits_low_cost * in_interval_low.detach() + joint_limits_high_cost * in_interval_high.detach()
            ).unsqueeze(0)
            return (cost,), (cost,)

        return goal_func


class JointLimitOnLikelihoodGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="JointLimitGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        def goal_func(likelihood_joint_at_limit):
            cost = -1e6 * torch.sum(likelihood_joint_at_limit).unsqueeze(0)
            return (cost,), (cost,)

        return goal_func


class PostureGoal(Goal):

    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="PostureGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):
        def goal_func(joint_range_utilization):
            cost = 1e3 * torch.sum(joint_range_utilization).unsqueeze(0)
            return (cost,), (cost,)

        return goal_func


class SelfCollisionGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="SelfCollisionGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):

        def goal_func(likelihood_self_collision):
            cost = 1e6 * likelihood_self_collision
            return (cost,), (cost,)

        return goal_func


class SingularityGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="SingularityGoal", is_active=is_active, mockbuild=mockbuild, device=device, dtype=dtype)

    def define_goal_cost_function(self):

        def goal_func(likelihood_ee_singularity):
            cost = 1e5 * likelihood_ee_singularity
            return (cost,), (cost,)

        return goal_func
