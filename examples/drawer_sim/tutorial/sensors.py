"""
Simulated Sensors for the Drawer Tutorial

Each sensor reads a ground-truth quantity from the simulation.

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Callable, Tuple, Union

import torch

from aicon_base.components import SensorComponent


class SimulationSensor(SensorComponent):
    """A sensor that reads from the simulation environment `env`."""

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        env,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.env = env
        super().__init__(name, connections, dtype=dtype, device=device, mockbuild=mockbuild)

    def measurement(self, values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(-1).clone()


class DrawerPoseSensor(SimulationSensor):
    """Position of the drawer handle."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"drawer_pos_meas": (3,)}

    def obtain_measurements(self) -> bool:
        self.quantities["drawer_pos_meas"] = self.measurement(self.env.env.get_drawer_handle_pos())
        return True


class EEPoseSensor(SimulationSensor):
    """Position of the end-effector."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"ee_pos_meas": (3,)}

    def obtain_measurements(self) -> bool:
        self.quantities["ee_pos_meas"] = self.measurement(self.env.env.get_ee_pos())
        return True


class EEForceSensor(SimulationSensor):
    """Magnitude of the force at the end-effector."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"ee_force_mag_meas": (1,)}

    def obtain_measurements(self) -> bool:
        self.quantities["ee_force_mag_meas"] = self.measurement([self.env.env.get_ee_force_magnitude()])
        return True
