"""
AICON Sensors for RGB(D) Cameras

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Union, Tuple, Type

import rclpy
import torch
import threading
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from aicon_base.components import SensorComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.timestamps import as_timestamp
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace, spin_node_in_thread

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile_images_realsense = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=10
)

qos_profile_info_realsense = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10
)


def preprocess_img_depth(
    depth_img,
    depth_divisor,
    scaling_factor,
    dtype: torch.dtype = torch.get_default_dtype(),
    device: torch.device = torch.get_default_device(),
):
    # Process Depth
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
    if scaling_factor != 1.0:
        img = cv2.resize(img, dsize=(int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))
    img_torch = torch.tensor(img.astype(float) / depth_divisor, dtype=dtype, device=device)
    # prevent weird depth artefacts
    img_torch[torch.isnan(img_torch)] = 0.0
    img_torch[torch.isinf(img_torch)] = 0.0
    img_torch[img_torch > 2] = 0.0
    img_torch[img_torch < 0.1] = 0.0
    return img_torch


def preprocess_img_color(
    color_img,
    dtype: torch.dtype = torch.get_default_dtype(),
    device: torch.device = torch.get_default_device(),
    scaling_factor: float = 1.0,
):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(color_img, desired_encoding="passthrough")
    if scaling_factor != 1.0:
        img = cv2.resize(img, dsize=(int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))
    return torch.tensor(img.copy(), dtype=dtype, device=device)


def get_camtopic_name(camera_type: str, output_type: str = "rgb", camera_name: str = "camera"):
    match camera_type, output_type:
        case "realsense", "rgb":
            return "/camera/" + camera_name + "/color/"
        case "realsense", "depth":
            return "/camera/" + camera_name + "/depth/"
        case "realsense", "depth_aligned":
            return "/camera/" + camera_name + "/aligned_depth_to_color/"
        case "asus", "rgb":
            return "/camera/rgb/"
        case "asus", "depth_aligned":
            return "/camera/depth_registered/"
        case _:
            raise LookupError("No known camera topics for (" + camera_type + ", " + output_type, ")")


class ROSCameraRGBSensor(SensorComponent):
    def obtain_measurements(self) -> bool:
        if self._internal_timestep <= 0:
            return False
        self.quantities["rgb_image_" + self.name] = torch.clone(self._internal_last_image)
        self.quantities["cam_D_" + self.name] = torch.clone(self._internal_D)
        self.quantities["cam_K_" + self.name] = torch.clone(self._internal_K)
        self.quantities["cam_R_" + self.name] = torch.clone(self._internal_R)
        self.quantities["cam_P_" + self.name] = torch.clone(self._internal_P)
        self.timestamp = self._internal_timestep
        return True

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None):
        self.quantities["rgb_image_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_D_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_K_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_R_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_P_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.local_quantity_names_and_shapes = {
            "rgb_image_" + self.name: (-1,),
            "cam_D_" + self.name: (-1,),
            "cam_K_" + self.name: (-1,),
            "cam_R_" + self.name: (-1,),
            "cam_P_" + self.name: (-1,),
        }

    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__(name=name, connections=connections, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.frame_id = "unknown"
        self.scaling_factor = scaling_factor

        self.node = rclpy.create_node(name + "_sensor_subscriber", namespace=default_ros_namespace)
        self.image_sub = self.node.create_subscription(
            Image, ros_topic_base + "image_raw", self.callback_img, qos_profile=qos_profile_images_realsense
        )
        self.info_sub = self.node.create_subscription(
            CameraInfo, ros_topic_base + "camera_info", self.callback_info, qos_profile=qos_profile_info_realsense
        )
        self._internal_timestep = as_timestamp(-1.0)
        self._internal_last_image = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_D = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_K = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_R = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_P = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_first_info_received = False

        spin_thread = threading.Thread(target=spin_node_in_thread, args=(self.node,), daemon=True)
        spin_thread.start()

    def callback_img(self, msg):
        with self.lock:
            self._internal_last_image = preprocess_img_color(
                msg, dtype=self.dtype, device=self.device, scaling_factor=self.scaling_factor
            )
            self._internal_timestep = as_timestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self.frame_id = msg.header.frame_id

    def callback_info(self, msg):
        with self.lock:
            self._internal_D = torch.tensor(msg.d, dtype=self.dtype, device=self.device).reshape(5)
            self._internal_K = torch.tensor(msg.k, dtype=self.dtype, device=self.device).reshape(3, 3)
            self._internal_R = torch.tensor(msg.r, dtype=self.dtype, device=self.device).reshape(3, 3)
            self._internal_P = torch.tensor(msg.p, dtype=self.dtype, device=self.device).reshape(3, 4)
            self._internal_first_info_received = True


class ROSCameraRGBDSensor(ROSCameraRGBSensor):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base_color: str,
        ros_topic_base_depth: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__(
            name=name,
            connections=connections,
            ros_topic_base=ros_topic_base_color,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            scaling_factor=scaling_factor,
        )

        if mockbuild:
            return

        self._internal_last_depth_image = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_timestep_depth = as_timestamp(-1.0)
        self.depth_image_sub = self.node.create_subscription(
            Image, ros_topic_base_depth + "image_raw", self.callback_depth_img, qos_profile=qos_profile_images_realsense
        )

    def callback_depth_img(self, msg):
        with self.lock:
            img = preprocess_img_depth(
                msg, dtype=self.dtype, device=self.device, scaling_factor=self.scaling_factor, depth_divisor=1000.0
            )  # mm to m
            self._internal_last_depth_image = img
            self._internal_timestep_depth = as_timestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self.frame_id = msg.header.frame_id

    def obtain_measurements(self) -> bool:
        if not (super().obtain_measurements() and self._internal_timestep_depth > 0):
            return False
        self.quantities["depth_image_" + self.name] = torch.clone(self._internal_last_depth_image)
        self.timestamp = torch.minimum(self._internal_timestep, self._internal_timestep_depth)
        return True
