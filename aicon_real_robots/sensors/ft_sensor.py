"""
AICON Sensors for Force-Torque

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union, Dict, Tuple, Type

import rclpy
import torch
from geometry_msgs.msg import WrenchStamped
import threading

from aicon_base.components import SensorComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.timestamps import as_timestamp
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace, default_qos_settings, spin_node_in_thread


class ForceTorqueSensor(SensorComponent):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        quantity_name: str,
        ros_topic: str,
        bias: torch.Tensor,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.quantity_name = quantity_name
        super().__init__(name=name, connections=connections, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return
        self.bias = bias
        self.node = rclpy.create_node(name + "_sensor_subscriber", namespace=default_ros_namespace)
        self.subscriber = self.node.create_subscription(WrenchStamped, ros_topic, self._callback, default_qos_settings)

        self._internal_timestep = as_timestamp(-1.0)
        self._internal_last_image = torch.tensor(-1, dtype=self.dtype, device=self.device)

        spin_thread = threading.Thread(target=spin_node_in_thread, args=(self.node,), daemon=True)
        spin_thread.start()

    def obtain_measurements(self) -> bool:
        if self._internal_timestep < 0:
            return False
        self.timestamp = self._internal_timestep
        self.quantities[self.quantity_name] = self._internal_last_ft - self.bias
        return True

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None) -> None:
        self.quantities[self.quantity_name] = torch.empty(6, dtype=self.dtype, device=self.device)
        self.local_quantity_names_and_shapes = {self.quantity_name: (6,)}

    def _callback(self, msg):
        self._internal_last_ft = torch.tensor(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self._internal_timestep = as_timestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
