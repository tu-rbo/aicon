"""
Blocks World with AICON: solve stacking tasks without a planner

Usage (from the repository root):
    python -m examples.blocksworld.run_experiment                        # all default setup/goal combinations
    python -m examples.blocksworld.run_experiment --setup 3towers --goal StackAonBonC

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import random

import numpy as np
import torch

from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from .experiment_specifications import get_building_functions_basic_blocks_world

DEFAULT_SETUPS = ["3towers", "3towersAlt", "2towers", "2towersAlt", "0towers"]
DEFAULT_GOALS = ["UnstackAll", "SmartStackAonBonC", "UnstackAonB", "StackAonB", "StackAonBonC"]


def run_experiment(init_setup: str, goal: str, max_ticks: int = 200) -> int:
    """Run until every goal is fulfilled; returns the number of ticks used, or -1 if it did not finish."""
    component_builders, _, _ = get_building_functions_basic_blocks_world(init_setup=init_setup, goal=goal)
    components = build_components(component_builders)
    for i in range(max_ticks):
        if all(not g.is_active for c in components.values() for g in c.goals.values()):
            return i
        components = run_component_sequence(components, torch.tensor(i * 0.01))
        action = components["BlockPuttingAction"].internal_action
        if action is not None:
            components["BlocksBelowSensor"].take_action(action)
    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", nargs="+", default=DEFAULT_SETUPS, help="initial block configurations")
    parser.add_argument("--goal", nargs="+", default=DEFAULT_GOALS, help="goals to pursue")
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for init_setup in args.setup:
        for goal in args.goal:
            ticks = run_experiment(init_setup, goal, max_ticks=args.max_ticks)
            result = f"solved after {ticks} ticks" if ticks >= 0 else f"not solved within {args.max_ticks} ticks"
            print(f"{init_setup} / {goal}: {result}")


if __name__ == "__main__":
    main()
