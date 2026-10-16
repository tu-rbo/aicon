"""
Goals for the Blocks World Domain

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


class StackAonBGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):

        super().__init__(name="StackAonB", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):

        def goal_func(likelihood_below):
            cost = 1.0 - likelihood_below[0, 4].unsqueeze(0)
            return cost, cost

        return goal_func


class StackAonBonCGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):

        super().__init__(name="StackAonBonC", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):

        def goal_func(likelihood_below):
            cost = 2.0 - likelihood_below[0, 4] - likelihood_below[6, 0].unsqueeze(0)
            return cost, cost

        return goal_func


class SmartStackAonBonC(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):

        super().__init__(name="SmartStackAonBonC", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):

        def goal_func(likelihood_below):
            if likelihood_below[0, 4] < 0.5:
                cost = 1.0 - likelihood_below[0, 4].unsqueeze(0)
            else:
                cost = 1.0 - likelihood_below[6, 0].unsqueeze(0)
            return cost, cost

        return goal_func


class UnstackAonBGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):

        super().__init__(name="UnstackAonB", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):

        def goal_func(likelihood_below):
            cost = likelihood_below[1, 0].unsqueeze(0)
            return cost, cost

        return goal_func


class UnstackAllGoal(Goal):
    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):

        super().__init__(name="UnstackAll", is_active=is_active, mockbuild=False, fulfilled_value=0.95)

    def define_goal_cost_function(self):

        def goal_func(likelihood_below):
            cost = torch.sum(likelihood_below).unsqueeze(0)
            return cost, cost

        return goal_func
