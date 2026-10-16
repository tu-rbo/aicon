"""
Real-Robot Object Pushing with AICON: sensors and visualization (vision / robot control PC)

Usage (from the repository root, ROS 2 sourced):
    python -m examples.object_pushing_real.run_sensors_only

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse

import torch

from aicon_ros_python_wrapping.ros_wrapping import build_components_with_ros_middleware, wait_for_ros_wrapped_components
from .experiment_specifications import get_sensor_only_building_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")
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
