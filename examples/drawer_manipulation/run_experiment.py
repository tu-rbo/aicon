"""
Drawer Manipulation with AICON: estimators and controllers (robot PC)

Start `run_sensors_only` on the vision PC first. Usage (from the repository root, ROS 2 sourced):
    python -m examples.drawer_manipulation.run_experiment

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

from aicon_ros_python_wrapping.ros_wrapping import build_components_with_ros_middleware, wait_for_ros_wrapped_components
from .experiment_specifications import get_basic_building_functions_joint_level


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=111, help="seed of torch, random and numpy")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    component_builders, connection_builders, frame_rates = get_basic_building_functions_joint_level()
    build_components_with_ros_middleware(
        component_builders,
        passive_mode=False,
        connection_building_functions=connection_builders,
        frame_rates=frame_rates,
        viz_kwargs={"grad_treshold": {"hand_synergy_activation": 0.01}},
    )
    try:
        wait_for_ros_wrapped_components()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
