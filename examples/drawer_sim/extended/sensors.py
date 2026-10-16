"""
Simulated Sensors for the Extended Drawer Example

Ground truth from the simulation with Gaussian noise; `noise_scale` multiplies all noise levels (0: exact).

Authors: James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Callable, Tuple, Union

import numpy as np
import torch

from ..tutorial.sensors import SimulationSensor
from ..util import bearing, ee_to_camera, point_in_camera_frame

POSITION_NOISE = 0.001  # [m] end-effector position and handle position
ORIENTATION_NOISE = 0.00175  # [rad] end-effector orientation
BEARING_NOISE = 0.0033  # per component of the unit bearing
FORCE_NOISE = 0.05  # [N]


class NoisySimulationSensor(SimulationSensor):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        env,
        noise_scale: float = 1.0,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.noise_scale = noise_scale
        super().__init__(name, connections, env, dtype=dtype, device=device, mockbuild=mockbuild)

    def noise(self, std) -> torch.Tensor:
        std = torch.as_tensor(std, dtype=self.dtype, device=self.device)
        return torch.randn(std.shape, dtype=self.dtype, device=self.device) * std * self.noise_scale


class EEPoseSensor(NoisySimulationSensor):
    """End-effector pose [x, y, z, rx, ry, rz]."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"ee_pos_meas": (6,)}

    def obtain_measurements(self) -> bool:
        pose = self.measurement(self.env.env.get_ee_pose())
        if self.noise_scale > 0:
            pose = pose + self.noise([POSITION_NOISE] * 3 + [ORIENTATION_NOISE] * 3)
        self.quantities["ee_pos_meas"] = pose
        return True


class BearingSensor(NoisySimulationSensor):
    """Bearing (unit direction) to the drawer handle in the wrist camera's frame."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"relative_position_in_CF_drawer": (3,)}

    def obtain_measurements(self) -> bool:
        handle = self.measurement(self.env.env.get_drawer_handle_pos())
        ee_pose = self.measurement(self.env.env.get_ee_pose())
        measured = bearing(point_in_camera_frame(ee_pose, handle, ee_to_camera(self.dtype, self.device)))
        if self.noise_scale > 0:
            measured = measured + self.noise([BEARING_NOISE] * 3)
        self.quantities["relative_position_in_CF_drawer"] = measured
        return True


class DrawerPoseSensor(NoisySimulationSensor):
    """Position of the drawer handle."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"drawer_pos_meas": (3,)}

    def obtain_measurements(self) -> bool:
        position = self.measurement(self.env.env.get_drawer_handle_pos())
        if self.noise_scale > 0:
            position = position + self.noise([POSITION_NOISE] * 3)
        self.quantities["drawer_pos_meas"] = position
        return True


class EEForceSensor(NoisySimulationSensor):
    """Magnitude of the force at the end-effector."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"ee_force_mag_meas": (1,)}

    def obtain_measurements(self) -> bool:
        force = self.env.env.get_ee_force_magnitude()
        if self.noise_scale > 0:
            force = force + np.random.randn() * FORCE_NOISE * self.noise_scale
        self.quantities["ee_force_mag_meas"] = self.measurement([force])
        return True
