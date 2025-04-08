"""
Goals for the blocks world experiment.
This module defines various goal conditions for the blocks world environment,
including stacking and unstacking goals with different strategies.
"""

from typing import Union

import torch

from aicon.base_classes.goals import Goal


class StackAonBGoal(Goal):
    """
    Goal to stack block A on top of block B.
    The goal is considered fulfilled when the likelihood of block A being on block B
    exceeds the fulfilled_value threshold.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the StackAonB goal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="StackAonB", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):
        """
        Define the cost function for the StackAonB goal.
        
        Returns:
            function: Cost function that takes likelihood_below as input
        """
        def goal_func(likelihood_below):
            """
            Calculate the cost based on the likelihood of block A being on block B.
            
            Args:
                likelihood_below: Matrix of likelihoods for block relationships
            
            Returns:
                tuple: (cost, cost) where cost is 1.0 - likelihood_below[0, 4]
            """
            cost = 1.0 - likelihood_below[0, 4]
            return cost, cost
        return goal_func


class StackAonBonCGoal(Goal):
    """
    Goal to stack block A on block B, and block B on block C.
    The goal is considered fulfilled when both stacking relationships are likely.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the StackAonBonC goal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="StackAonBonC", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):
        """
        Define the cost function for the StackAonBonC goal.
        
        Returns:
            function: Cost function that takes likelihood_below as input
        """
        def goal_func(likelihood_below):
            """
            Calculate the cost based on the likelihood of both stacking relationships.
            
            Args:
                likelihood_below: Matrix of likelihoods for block relationships
            
            Returns:
                tuple: (cost, cost) where cost is 2.0 - likelihood_below[0, 4] - likelihood_below[6, 0]
            """
            cost = 2.0 - likelihood_below[0, 4] - likelihood_below[6, 0]
            return cost, cost
        return goal_func


class SmartStackAonBonC(Goal):
    """
    Goal to stack block A on block B, and block B on block C, with a smart strategy.
    The goal focuses on one stacking relationship at a time based on current progress.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the SmartStackAonBonC goal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="SmartStackAonBonC", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):
        """
        Define the cost function for the SmartStackAonBonC goal.
        
        Returns:
            function: Cost function that takes likelihood_below as input
        """
        def goal_func(likelihood_below):
            """
            Calculate the cost based on the current progress of stacking relationships.
            
            Args:
                likelihood_below: Matrix of likelihoods for block relationships
            
            Returns:
                tuple: (cost, cost) where cost depends on which stacking relationship needs attention
            """
            if likelihood_below[0, 4] < 0.5:
                cost = 1.0 - likelihood_below[0, 4]
            else:
                cost = 1.0 - likelihood_below[6, 0]
            return cost, cost
        return goal_func


class UnstackAonBGoal(Goal):
    """
    Goal to unstack block A from block B.
    The goal is considered fulfilled when the likelihood of block A being on block B
    is below the fulfilled_value threshold.
    """

    def __init__(self, is_active: bool, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the UnstackAonB goal.
        
        Args:
            is_active: Whether the goal is active
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name="UnstackAonB", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

    def define_goal_cost_function(self):
        """
        Define the cost function for the UnstackAonB goal.
        
        Returns:
            function: Cost function that takes likelihood_below as input
        """
        def goal_func(likelihood_below):
            """
            Calculate the cost based on the likelihood of block A being on block B.
            
            Args:
                likelihood_below: Matrix of likelihoods for block relationships
            
            Returns:
                tuple: (cost, cost) where cost is likelihood_below[1,0]
            """
            cost = likelihood_below[1,0]
            return cost, cost
        return goal_func