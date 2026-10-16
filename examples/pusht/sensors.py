"""
Simulated Sensors for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import SensorComponent
from .params import NUM_KEYPOINTS
from .env import env_instance


class KeypointSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"keypoint_locations": (NUM_KEYPOINTS, 2)}

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        wrong_order_points = torch.tensor(
            env_instance.observation["environment_state"].reshape(NUM_KEYPOINTS, 2),
            dtype=self.dtype,
            device=self.device,
        )
        self.quantities["keypoint_locations"] = torch.index_select(
            wrong_order_points, dim=0, index=torch.tensor([0, 1, 2, 5, 6, 7, 4, 3])
        )
        return True


class COMSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"object_COM": (2,)}

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        self.quantities["object_COM"] = torch.mean(
            torch.tensor(
                env_instance.observation["environment_state"].reshape(NUM_KEYPOINTS, 2),
                dtype=self.dtype,
                device=self.device,
            ),
            dim=0,
        )
        return True


class PusherSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pusher_point_meas": (2,)}

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        self.quantities["pusher_point_meas"] = env_instance.observation["agent_pos"]
        return True
