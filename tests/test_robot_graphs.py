"""Build the real-robot graphs in mock mode (no hardware); needs a sourced ROS 2 installation with aicon_ros built."""

import importlib

import pytest
import torch

rclpy = pytest.importorskip("rclpy")
# installed by the colcon build; the source folder aicon_ros/ alone would make aicon_ros.msg importable
pytest.importorskip("aicon_ros_python_wrapping")
pytest.importorskip("aicon_ros.msg")


pytestmark = pytest.mark.usefixtures("ros_context")


def usable_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# (module, function, number of GPUs the graph places components on)
@pytest.mark.parametrize(
    "module, function, gpus",
    [
        ("examples.object_pushing_real.experiment_specifications", "get_building_functions", 0),
        ("examples.object_pushing_real.experiment_specifications", "get_sensor_only_building_functions", 1),
        ("examples.drawer_manipulation.experiment_specifications", "get_basic_building_functions_joint_level", 2),
        ("examples.drawer_manipulation.experiment_specifications", "get_sensor_only_building_functions", 2),
    ],
)
def test_robot_graph_builds(module, function, gpus):
    if usable_gpus() < gpus:
        pytest.skip(f"needs {gpus} usable CUDA device(s)")
    component_builders, connection_builders, frame_rates = getattr(importlib.import_module(module), function)()
    for name, build in component_builders.items():
        component = build(True)
        for connection_name in getattr(component, "connections", {}):
            assert connection_name in connection_builders, (name, connection_name)
