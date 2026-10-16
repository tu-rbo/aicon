"""
Drawer Manipulation with AICON: camera, force-torque and joint sensors (vision / robot control PC)

Usage (from the repository root, ROS 2 sourced, AICON_TAPIR_CHECKPOINT and AICON_OS2D_ROOT set):
    python -m examples.drawer_manipulation.run_sensors_only

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
from .experiment_specifications import get_sensor_only_building_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=111, help="seed of torch, random and numpy")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    component_builders, connection_builders, frame_rates = get_sensor_only_building_functions()
    build_components_with_ros_middleware(
        component_builders,
        passive_mode=True,
        connection_building_functions=connection_builders,
        frame_rates=frame_rates,
        viz_kwargs={"grad_treshold": {"hand_synergy_activation": 0.01}},
    )
    wait_for_ros_wrapped_components()


if __name__ == "__main__":
    main()
