"""
Sensors for the drawer experiment.
This module defines sensor components for the drawer manipulation environment,
including camera-based position sensing.
"""

from typing import Dict, Union

import torch

from aicon.base_classes.connections import ActiveInterconnection
from aicon.drawer_experiment.util import get_sine_of_angles
from aicon.sensors.camera_sensors import ROSCameraRGBSensor


def center_threshold_rgb_img(img: torch.Tensor, threshold_lower: torch.Tensor,
                           threshold_upper: torch.Tensor) -> torch.Tensor:
    """
    Find the center of mass of pixels within a given threshold range in an RGB image.
    
    Args:
        img: RGB image tensor
        threshold_lower: Lower threshold for each channel
        threshold_upper: Upper threshold for each channel
    
    Returns:
        torch.Tensor: Center of mass coordinates [x, y] or [0, 0] if no pixels match
    """
    mask = torch.all(torch.logical_and(img >= threshold_lower, img <= threshold_upper), dim=-1)
    # import cv2 as cv
    # import numpy as np
    # cv.imshow("mask", mask.cpu().numpy().astype(np.uint8) * 255)
    # cv.waitKey(1)
    grid_x, grid_y = torch.meshgrid([torch.arange(img.shape[0]), torch.arange(img.shape[1])], indexing="ij")
    grid = torch.concatenate([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)
    ids = grid[mask]
    if torch.sum(mask) == 0:
        return torch.zeros(2) * torch.nan
    com = torch.mean(ids.to(dtype=img.dtype, device=img.device), dim=0)
    # com = torch.concatenate([torch.median(ids[:, 0]).unsqueeze(0), torch.median(ids[:, 1]).unsqueeze(0)]).type(torch.get_default_dtype())
    return com


class CameraDrawerPosSensor(ROSCameraRGBSensor):
    """
    Camera sensor for detecting drawer position.
    This sensor processes RGB images to estimate the relative position of the drawer
    in the camera frame.
    """

    state_dim = 3

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 ros_topic_base: str, lower_cam_threshold: torch.Tensor,
                 upper_cam_threshold: torch.Tensor, object_name: str = "drawer",
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the camera drawer position sensor.
        
        Args:
            name: Name of the sensor
            connections: Dictionary of active interconnections
            ros_topic_base: Base topic name for ROS communication
            lower_cam_threshold: Lower threshold for color segmentation
            upper_cam_threshold: Upper threshold for color segmentation
            object_name: Name of the object being tracked
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        self.lower_cam_threshold = lower_cam_threshold
        self.upper_cam_threshold = upper_cam_threshold
        self.object_name = object_name
        super().__init__(name, connections, ros_topic_base, dtype, device, mockbuild=mockbuild)

    def obtain_measurements(self) -> bool:
        """
        Process the latest camera image to obtain drawer position measurements.
        
        Returns:
            bool: True if measurements were successfully obtained, False otherwise
        """
        if self._internal_timestep <= 0:
            return False
        P_inv = torch.inverse(self._internal_P[:3, :3])
        loc = torch.flip(
            center_threshold_rgb_img(self._internal_last_image, self.lower_cam_threshold, self.upper_cam_threshold), dims=[0])
        rel_pos = torch.einsum("ij,j->i", P_inv,
                              torch.concat([loc, torch.ones(1, dtype=self.dtype, device=self.device)]))[:3]
        self.timestamp = torch.clone(self._internal_timestep)
        self.quantities["relative_position_in_CF_" + self.object_name] = get_sine_of_angles(rel_pos)
        return True

    def initial_definitions(self):
        """
        Initialize the sensor's quantities.
        Sets up the relative position quantity with the correct dimensions.
        """
        self.quantities["relative_position_in_CF_" + self.object_name] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)