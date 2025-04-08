"""
Synchronous experiment runner for the blocks world domain.
This module provides functionality to run blocks world experiments in a synchronous manner,
without ROS middleware, for testing and development purposes.
"""

import random
from typing import Callable

import numpy as np
import torch

from aicon.blocksworld_experiment.experiment_specifications import get_building_functions_basic_blocks_world
from aicon.middleware.python_sequential import build_components, run_component_sequence


def start_all_components(functions_constructor: Callable):
    """
    Initialize all components for the blocks world experiment.
    
    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates
    
    Returns:
        Dictionary of initialized components
    """
    component_builders, connection_builders, frame_rates = functions_constructor()
    return build_components(component_builders)


def run_experiment(functions_constructor: Callable):
    """
    Run a blocks world experiment synchronously.
    
    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates
    """
    components = start_all_components(functions_constructor=functions_constructor)

    def determine_goals_fulfilled(cs):
        """
        Check if all active goals in the components are fulfilled.
        
        Args:
            cs: Dictionary of components
            
        Returns:
            True if all goals are fulfilled, False otherwise
        """
        for _, comp in cs.items():
            for _, goal in comp.goals.items():
                if goal.is_active:
                    return False
        return True

    # Run the experiment for up to 200 timesteps or until goals are fulfilled
    for i in range(200):
        if determine_goals_fulfilled(components):
            break
        components = run_component_sequence(components, torch.tensor(i * 0.01))
        action = components["BlockPuttingAction"].internal_action
        if action is not None:
            components["BlocksBelowSensor"].take_action(action)


if __name__ == "__main__":
    # Set up deterministic behavior for reproducibility
    torch.set_default_dtype(torch.float64)
    torch._dynamo.config.capture_func_transforms = True
    torch.use_deterministic_algorithms(True)
    torch.set_printoptions(profile="full", precision=20)

    # Set random seeds for reproducibility
    seed = 111
    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Run a simple test experiment
    NUM_BLOCKS = 10
    init_setup_list = ["3towers"]
    goal_list = ["StackAonBonC"]

    for init_setup in init_setup_list:
        for goal in goal_list:
            setup_func = lambda: get_building_functions_basic_blocks_world(init_setup=init_setup,
                                                                           goal=goal)
            print("Start exp " + init_setup + "_" + goal)
            run_experiment(functions_constructor=setup_func)