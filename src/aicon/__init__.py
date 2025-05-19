"""
Active Interconnect (AICON)

A framework for differentiable online multimodal interactive perception in robotics.
"""

__version__ = "1.0.0"
__author__ = "Vito Mengers"
__email__ = "v.mengers@tu-berlin.de"

# Import main components
from aicon.base_classes.components import (
    Component,
    SensorComponent,
    EstimationComponent,
    ActionComponent,
)
from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.derivatives import DerivativeDict
from aicon.base_classes.goals import Goal

from aicon import sensors, middleware, math, inference, drawer_experiment, blocksworld_experiment, actions

# Import examples
from aicon.blocksworld_experiment import run_experiment_sync

__all__ = [
    "Component",
    "SensorComponent",
    "EstimationComponent",
    "ActionComponent",
    "ActiveInterconnection",
    "DerivativeDict",
    "Goal",
    "run_experiment_sync",
    "run_experiment",
    "sensors",
    "middleware",
    "math",
    "inference",
    "drawer_experiment",
    "blocksworld_experiment",
    "actions"
]
