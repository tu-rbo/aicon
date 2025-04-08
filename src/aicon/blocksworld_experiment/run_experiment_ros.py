"""
ROS-based experiment runner for the blocks world domain.
This module provides functionality to run blocks world experiments using ROS middleware,
including ROS bag recording for data collection.
"""

import random
import time
from typing import Callable

import numpy as np
import torch

from aicon.blocksworld_experiment.experiment_specifications import get_building_functions_basic_blocks_world
from aicon.middleware.ros_wrapping import build_components_with_ros_middleware
from aicon.middleware.util_ros import ROSBagRecorder


def start_all_components(functions_constructor: Callable, passive_mode: bool = False):
    """
    Initialize all components for the blocks world experiment using ROS middleware.
    
    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates
        passive_mode: If True, run in passive mode without active control
    
    Returns:
        List of subprocesses running the components
    """
    component_builders, connection_builders, frame_rates = functions_constructor()
    return build_components_with_ros_middleware(component_builders, passive_mode=passive_mode,
                                         connection_building_functions=connection_builders,
                                         frame_rates=frame_rates)


def run_experiment(functions_constructor: Callable, passive_mode: bool = False, rosbag_name: str = ""):
    """
    Run a blocks world experiment with ROS bag recording.
    
    Args:
        functions_constructor: Function that returns component builders, connection builders, and frame rates
        passive_mode: If True, run in passive mode without active control
        rosbag_name: Name prefix for the ROS bag file
    
    Returns:
        Tuple of (subprocesses, rosbag_recorder)
    """
    rosbag_recorder = ROSBagRecorder(topic_name_space="/direl_quantities",
                                     prefix="/home/vito/bagfiles/blocksworld_problems_collection/"+rosbag_name,
                                     topics=["/blocksworld_action",])
    time.sleep(5)
    subprocesses = start_all_components(functions_constructor=functions_constructor, passive_mode=passive_mode)
    return subprocesses, rosbag_recorder


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

    # Run experiments with different initial setups and goals
    NUM_BLOCKS = 30
    init_setup_list = ["3towers"]
    goal_list = ["StackAonB"]

    for init_setup in init_setup_list:
        for goal in goal_list:
            setup_func = lambda: get_building_functions_basic_blocks_world(init_setup=init_setup,
                                                                           goal=goal)
            print("Start exp " + init_setup + "_" + goal)
            subprocesses, rosbag_recorder = run_experiment(functions_constructor=setup_func, passive_mode=False,
                                                           rosbag_name=init_setup + "_" + goal + "_"+str(NUM_BLOCKS))
            time.sleep(1200)  # Run for 20 minutes
            rosbag_recorder.close()
            for p in subprocesses:
                p.terminate()
            time.sleep(10)  # Wait between experiments