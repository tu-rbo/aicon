import inspect
from abc import abstractmethod
from typing import Callable, Iterable, Type, Union

import torch
from tensordict import TensorDict

from aicon.base_classes.util import collect_args, gather_partial_derivatives, collect_shapes


class Goal:
    """
    Base class for goals in the AICON framework.
    
    A Goal represents an objective that components should strive to achieve.
    Goals can be used to guide the behavior of components through cost functions
    that are differentiable and can be optimized.
    
    Attributes:
        name (str): Unique identifier for the goal
        is_active (bool): Whether the goal is currently active
        g_func (Callable): Goal cost function
        fulfilled_value (Union[float, None]): Value at which the goal is considered fulfilled
    """
    
    name: str
    is_active: bool
    g_func: Callable
    fulfilled_value : Union[float, None]

    def __init__(self, name: str, is_active:bool=True, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, fulfilled_value : Union[float, None] = None):
        """
        Initialize a new goal.
        
        Args:
            name: Unique identifier for the goal
            is_active: Whether the goal is initially active
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
            fulfilled_value: Optional value at which the goal is considered fulfilled
        """
        self.device = device
        self.dtype = dtype
        self.name = name
        self.is_active = is_active
        self.fulfilled_value = fulfilled_value
        if not mockbuild:
            self.g_func = self.define_goal_cost_function()

    @abstractmethod
    def define_goal_cost_function(self):
        """
        Define the goal's cost function.
        
        This method should be implemented by subclasses to define how the goal's
        cost is computed from the current state.
        
        Returns:
            Callable: The goal cost function
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass


def evaluate_goals(goals: Iterable[Type[Goal]], output_dict):
    """
    Evaluate multiple goals and compute their derivatives.
    
    This function computes the cost values and partial derivatives for a collection
    of goals based on the current output dictionary.
    
    Args:
        goals: Iterable of Goal objects to evaluate
        output_dict: Dictionary of current outputs/quantities
        
    Returns:
        Tuple containing:
        - TensorDict of partial derivatives for each goal
        - TensorDict of current cost values for each goal
    """
    partial_derivatives = TensorDict({}, batch_size=())
    goal_values = TensorDict({}, batch_size=())
    for g in goals:
        if g.is_active:
            args, diff_argnums, input_names_diff = collect_args(output_dict, inspect.getfullargspec(g.g_func).args, None, [g.name])
            partial_jacs, current_cost_value = torch.func.jacfwd(g.g_func, argnums=diff_argnums,
                                                                                    has_aux=True, randomness="same")(*args)
            in_out_shapes = collect_shapes(output_dict, {g.name: current_cost_value})
            partial_derivatives = partial_derivatives.update(gather_partial_derivatives(partial_jacs, input_names_diff, [g.name], in_out_shapes))
            goal_values[g.name] = current_cost_value
    return partial_derivatives, goal_values
