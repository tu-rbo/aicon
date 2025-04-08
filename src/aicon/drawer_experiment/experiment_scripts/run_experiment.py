import random
from typing import Callable

import numpy as np
import torch

from aicon.drawer_experiment.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.ros_wrapping import build_components_with_ros_middleware, wait_for_ros_wrapped_components
from aicon.middleware.util_ros import ROSBagRecorder


def start_all_components(functions_constructor : Callable, passive_mode: bool = False):
    component_builders, connection_builders, frame_rates = functions_constructor()
    del component_builders["Visualizer"]    # do not start viz to because it impacts framerate of other components
    all_ros_subprocesses = build_components_with_ros_middleware(component_builders, passive_mode=passive_mode,
                                         connection_building_functions=connection_builders,
                                         frame_rates=frame_rates,
                                         viz_kwargs={"grad_treshold": {"hand_synergy_activation": 0.01}})
    return all_ros_subprocesses


def run_experiment(functions_constructor : Callable, passive_mode: bool = False, name_suffix : str = "", no_wait : bool = False):
    rosbag_recorder = ROSBagRecorder(topic_name_space="/direl_quantities",
                                     prefix="/home/vito/bagfiles/drawer_experiment"+name_suffix,
                                     topics=["/outer/camera/color/image_raw/compressed",
                                             "/outer/camera/color/camera_info",
                                             "/camera/color/image_raw/compressed", "/camera/color/camera_info",
                                             "/tf", "/tf_static"])
    all_ros_subprocesses = start_all_components(functions_constructor=functions_constructor, passive_mode=passive_mode)
    all_ros_subprocesses.append(rosbag_recorder)
    if not no_wait:
        try:
            wait_for_ros_wrapped_components()
        except KeyboardInterrupt:
            rosbag_recorder.close()
    return all_ros_subprocesses


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch._dynamo.config.capture_func_transforms = True
    torch.use_deterministic_algorithms(True)
    torch.set_printoptions(profile="full", precision=20)

    seed = 111
    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    run_experiment(functions_constructor=get_building_functions_basic_drawer_motion, passive_mode=False,
                   name_suffix="")
