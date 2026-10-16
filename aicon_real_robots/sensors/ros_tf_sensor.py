"""
AICON Sensors for Arbitrary Frames in tf2

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Union, Tuple, Type

import tf2_ros
import torch
import rclpy

from aicon_base.components import SensorComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.math.se3_and_SE3 import log_map_se3
from aicon_ros_python_wrapping.util_ros import get_tf_frame, time_ros_to_torch
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace


class ROSTFSensor(SensorComponent):
    def obtain_measurements(self) -> bool:
        try:
            transform, transform_time = get_tf_frame(
                buffer=self._tfBuffer,
                node=self.node,
                source_frame=self.source_frame,
                target_frame=self.target_frame,
                timeout=0.5,
            )
        except AssertionError:
            return False
        transform_time = time_ros_to_torch(transform_time)
        if self.timestamp >= transform_time > 1.0:
            return False
        self.timestamp = transform_time
        self.quantities[self.quantity_name] = log_map_se3(transform)
        return True

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None):
        self.quantities[self.quantity_name] = torch.empty(6, dtype=self.dtype, device=self.device)
        self.local_quantity_names_and_shapes = {self.quantity_name: (6,)}

    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        source_frame: str,
        target_frame: str,
        quantity_name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.quantity_name = quantity_name
        super().__init__(name=name, connections=connections, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.node = rclpy.create_node(name + "_sensor_subscriber", namespace=default_ros_namespace)

        self._tfBuffer = tf2_ros.Buffer()
        self._tflistener = tf2_ros.TransformListener(self._tfBuffer, self.node)

        self.source_frame = source_frame
        self.target_frame = target_frame
        self.frame_id = self.source_frame
