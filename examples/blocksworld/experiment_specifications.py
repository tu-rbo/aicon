"""
Experiment Specification for the Blocks World Domain

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from .actions import BlockPuttingAction
from .connections import build_connections
from .estimators import ClearEstimator, BelowEstimator
from .goal import SmartStackAonBonC, StackAonBGoal, StackAonBonCGoal, UnstackAonBGoal, UnstackAllGoal
from .sensors import BlocksBelowSensor


def get_building_functions_basic_blocks_world(init_setup: str = "3towers", goal: str = "StackAonB"):
    connection_builders = build_connections()
    match goal:
        case "StackAonB":
            goal_class = StackAonBGoal
        case "StackAonBonC":
            goal_class = StackAonBonCGoal
        case "SmartStackAonBonC":
            goal_class = SmartStackAonBonC
        case "UnstackAonB":
            goal_class = UnstackAonBGoal
        case "UnstackAll":
            goal_class = UnstackAllGoal
        case _:
            raise ValueError(f"unknown goal {goal}")
    diff_depth = 12
    component_builders = {
        "BlocksBelowSensor": lambda mockbuild: BlocksBelowSensor(
            "BlocksBelowSensor",
            connections={k: connection_builders[k] for k in ("BelowLikelihoodDummySensing",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            setup=init_setup,
        ),
        "BelowEstimator": lambda mockbuild: BelowEstimator(
            "BelowEstimator",
            connections={
                k: connection_builders[k] for k in ("BelowLikelihoodDummySensing", "ClearLikelihood", "BelowLikelihood")
            },
            goals={
                goal: lambda d, t, mockbuild=False: goal_class(is_active=True, dtype=t, device=d, mockbuild=mockbuild)
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            prevent_loops_in_differentiation=False,
            max_length_differentiation_trace=diff_depth,
        ),
        "ClearEstimator": lambda mockbuild: ClearEstimator(
            "ClearEstimator",
            connections={k: connection_builders[k] for k in ("ClearLikelihood", "BelowLikelihood")},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            prevent_loops_in_differentiation=False,
            max_length_differentiation_trace=diff_depth,
        ),
        "BlockPuttingAction": lambda mockbuild: BlockPuttingAction(
            "BlockPuttingAction",
            connections={k: connection_builders[k] for k in ("BelowLikelihood", "BelowLikelihoodDummySensing")},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
    }
    frame_rates = {
        "BlocksBelowSensor": 10,
        "BelowEstimator": 10,
        "ClearEstimator": 10,
        "BlockPuttingAction": 10,
    }
    return component_builders, connection_builders, frame_rates
