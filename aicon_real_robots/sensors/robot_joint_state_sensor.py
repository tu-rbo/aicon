"""
AICON Sensors for Robot Joints

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import traceback
from typing import Dict, Union, Tuple, Type

import torch
import rclpy
from sensor_msgs.msg import JointState
import threading

from aicon_base.components import SensorComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.timestamps import as_timestamp
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace, spin_node_in_thread

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile_joint_states = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10
)


class RobotJointSensor(SensorComponent):
    state_dim: int = 7
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"robot_joint_states_meas": (7,)}

    def obtain_measurements(self) -> bool:
        if self._internal_timestep <= 0:
            return False
        self.quantities["robot_joint_states_meas"] = torch.clone(self._internal_joint_states)
        self.timestamp = self._internal_timestep
        return True

    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name=name, connections=connections, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.frame_id = "unknown"

        self.node = rclpy.create_node(name + "_sensor_subscriber", namespace=default_ros_namespace)
        self.subscriber = self.node.create_subscription(
            JointState, ros_topic_base + "/joint_states", self.callback, qos_profile_joint_states
        )
        self._internal_timestep = as_timestamp(-1.0)
        self._internal_joint_states = torch.tensor(-1, dtype=self.dtype, device=self.device)

        spin_thread = threading.Thread(target=spin_node_in_thread, args=(self.node,), daemon=True)
        spin_thread.start()

    def callback(self, msg):
        try:
            self._internal_joint_states = torch.tensor(
                msg.position[:7]
            )  # assuming panda joints come first (big assumption, careful!)
            self._internal_timestep = as_timestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self.frame_id = msg.header.frame_id
        except Exception:
            print("Exception during Joint State Sensor Callback:")
            traceback.print_exc()
