# -*- coding: utf-8 -*-
"""
Base for Goals in AICON

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import inspect
from abc import abstractmethod
from typing import Callable, Union, Iterable, Type

import torch
from tensordict import TensorDict

from aicon_base.components.util import collect_args, collect_shapes, gather_partial_derivatives


class Goal:
    """A cost on quantities of the component it is attached to. Implement `define_goal_cost_function`, returning a
    function of quantity names that returns ((cost,), (cost,)). The goal is deactivated once its cost falls below
    `fulfilled_value` (None or -inf: never)."""

    name: str
    is_active: bool
    g_func: Callable
    fulfilled_value: Union[float, None]

    def __init__(
        self,
        name: str,
        is_active: bool = True,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        fulfilled_value: Union[float, None] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.name = name
        self.is_active = is_active
        self.fulfilled_value = fulfilled_value
        if not mockbuild:
            self.g_func = self.define_goal_cost_function()

    @abstractmethod
    def define_goal_cost_function(self):
        pass


def evaluate_goals(goals: Iterable[Type[Goal]], output_dict):
    partial_derivatives = TensorDict({}, batch_size=())
    goal_values = TensorDict({}, batch_size=())
    for g in goals:
        if g.is_active:
            args, diff_argnums, input_names_diff = collect_args(
                output_dict, inspect.getfullargspec(g.g_func).args, None, [g.name]
            )
            partial_jacs, current_cost_value_tuple = torch.func.jacfwd(
                g.g_func, argnums=diff_argnums, has_aux=True, randomness="same"
            )(*args)
            in_out_shapes = collect_shapes(output_dict, {g.name: current_cost_value_tuple[0]})
            partial_derivatives = partial_derivatives.update(
                gather_partial_derivatives(partial_jacs, input_names_diff, [g.name], in_out_shapes)
            )
            goal_values[g.name] = current_cost_value_tuple[0]
    return partial_derivatives, goal_values
