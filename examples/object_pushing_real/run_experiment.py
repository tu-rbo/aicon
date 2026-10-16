"""
Real-Robot Object Pushing with AICON: estimators and controller (robot PC)

Start `run_sensors_only` on the vision PC first. Usage (from the repository root, ROS 2 sourced):
    python -m examples.object_pushing_real.run_experiment --bag-dir ~/bagfiles

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import os
import re

import torch

from aicon_ros_python_wrapping.ros_wrapping import build_components_with_ros_middleware, wait_for_ros_wrapped_components
from aicon_ros_python_wrapping.util_ros import ROSBagRecorder
from .experiment_specifications import get_building_functions
from .params import RECORD_COLOR_TOPIC, RECORD_DEPTH_TOPIC

BAG_EXCLUDE_TOPICS = [
    # raw image and depth streams are too large to record; everything else (incl. camera infos) is small
    "/camera/.*/(image_raw|image_rect_raw|metadata)",
    # the robot PC publishes an older franka_msgs/FrankaState definition than the one installed here, for which
    # rosbag2 cannot load a type support library
    "/franka_robot_state_broadcaster/robot_state",
]
BAG_EXCLUDE_REGEX = "|".join(BAG_EXCLUDE_TOPICS)
# Throttled, compressed image streams published by ObjectPushingSensor. They cannot be reconstructed after a run,
# so the recorder refuses to start if the exclude pattern would drop them.
BAG_REQUIRED_TOPICS = [RECORD_DEPTH_TOPIC, RECORD_COLOR_TOPIC]


def check_required_topics_recordable():
    pattern = re.compile(BAG_EXCLUDE_REGEX)
    excluded = [t for t in BAG_REQUIRED_TOPICS if pattern.search(t)]
    if excluded:
        raise RuntimeError("BAG_EXCLUDE_REGEX would drop required topic(s) %s from the recording." % excluded)


def run_experiment(bag_dir=None, name_suffix=""):
    rosbag_recorder = None
    if bag_dir is not None:
        check_required_topics_recordable()
        rosbag_recorder = ROSBagRecorder(
            record_all=True,
            exclude_regex=BAG_EXCLUDE_REGEX,
            prefix=os.path.join(bag_dir, "pushing_experiment" + name_suffix),
        )
    component_builders, connection_builders, frame_rates = get_building_functions()
    all_ros_subprocesses = build_components_with_ros_middleware(
        component_builders,
        passive_mode=False,
        connection_building_functions=connection_builders,
        frame_rates=frame_rates,
        num_threads=1,
        viz_kwargs={"grad_treshold": {"hand_synergy_activation": 0.01}},
    )
    try:
        wait_for_ros_wrapped_components()
    except KeyboardInterrupt:
        pass
    finally:
        if rosbag_recorder is not None:
            rosbag_recorder.close()
    return all_ros_subprocesses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag-dir", default=None, help="record a rosbag into this directory")
    parser.add_argument("--name-suffix", default="", help="appended to the bag name")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision("medium")
    run_experiment(bag_dir=os.path.expanduser(args.bag_dir) if args.bag_dir else None, name_suffix=args.name_suffix)


if __name__ == "__main__":
    main()
