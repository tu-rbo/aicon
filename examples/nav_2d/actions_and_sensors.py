"""
Simulated Agent, Actions and Sensors for the 2D Navigation Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"


import math
from typing import Tuple, Dict, Callable, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .constants import obs_img_size
from aicon_base.components import ActionComponent, SensorComponent

location_agent = torch.tensor([0.0, 10.0], dtype=torch.float)
location_target = torch.tensor([20.0, 10.0], dtype=torch.float)
# Walls of the simulated world: cell (i, j) is the square [i - 0.5, i + 0.5] x [j - 0.5, j + 0.5], a wall where the
# obstacle image is 0 (see scenarios.py). The agent is a point that stops at the first wall it touches.
walls = None
collided = False


def set_initial_locations(agent, target):
    global location_agent, location_target
    location_agent = agent
    location_target = target


def set_obstacle_map(obstacle_image):
    """Walls of the simulated world (obstacle image with 0 = wall); also clears an earlier collision."""
    global walls, collided
    walls = np.asarray(obstacle_image) == 0
    collided = False


def first_wall_contact(start, end):
    """Fraction t in [0, 1] of the way from `start` to `end` at which the straight path first touches a wall cell, or
    None if it touches none. Cells outside the map count as walls."""
    (x0, y0), (x1, y1) = (float(v) for v in start), (float(v) for v in end)
    contact = None
    # candidate cells: those around the segment's bounding box
    for i in range(math.floor(min(x0, x1) + 0.5) - 1, math.floor(max(x0, x1) + 0.5) + 2):
        for j in range(math.floor(min(y0, y1) + 0.5) - 1, math.floor(max(y0, y1) + 0.5) + 2):
            inside_map = 0 <= i < walls.shape[0] and 0 <= j < walls.shape[1]
            if inside_map and not walls[i, j]:
                continue
            # slab test of the segment against the cell square
            t_enter, t_exit = 0.0, 1.0
            for p0, d, c in ((x0, x1 - x0, i), (y0, y1 - y0, j)):
                lo, hi = c - 0.5, c + 0.5
                if d == 0.0:
                    if not lo <= p0 <= hi:
                        t_enter, t_exit = 1.0, 0.0
                    continue
                t_lo, t_hi = sorted(((lo - p0) / d, (hi - p0) / d))
                t_enter, t_exit = max(t_enter, t_lo), min(t_exit, t_hi)
            if t_enter <= t_exit and (contact is None or t_enter < contact):
                contact = t_enter
    return contact


class NavActionComp(ActionComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "action": (2,),
    }

    def _start(self):
        pass

    def _stop(self):
        pass

    def send_action_values(self) -> None:
        action = self.quantities["action"]
        global location_agent, collided
        if collided:
            return
        new_location = location_agent + action
        contact = first_wall_contact(location_agent, new_location) if walls is not None else None
        if contact is not None:
            new_location = location_agent + contact * (new_location - location_agent)
            collided = True
        location_agent = new_location

    def determine_new_actions(self):
        gain = 10
        last_action = self.quantities["action"]
        steepest_grad = self.get_steepest_plus_projected("action", time_threshold=self.timestamp - 0.5)
        if torch.norm(steepest_grad) == 0:
            return
        new_action = last_action - gain * steepest_grad / torch.norm(steepest_grad)
        new_action = new_action / torch.norm(new_action) * 0.2
        self.quantities["action"] = new_action

    def initialize_quantities(self) -> bool:
        self.quantities["action"] = torch.zeros(2, dtype=self.dtype, device=self.device)
        return True

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        goals: Union[None, Dict[str, Callable]] = None,
        forward_mode: bool = False,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__(
            name,
            connections,
            goals=goals,
            forward_mode=forward_mode,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        )


class NavAgentSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"agent_meas": (2,)}

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        self.quantities["agent_meas"] = location_agent
        return True


class NavTargetSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"target_meas": (1,)}

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        agent2target = location_target - location_agent
        # bearing-only measurement: unit vector from the agent towards the target
        self.quantities["target_meas"] = torch.cat(
            [agent2target[0:1] / torch.norm(agent2target), agent2target[1:2] / torch.norm(agent2target)]
        )
        return True


class ObstacleImageSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"obstacle_image": (obs_img_size, obs_img_size)}
    fixed_obs_img = None

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        dist_trans = distance_transform_edt(self.fixed_obs_img)
        dist_trans[self.fixed_obs_img == 0] = 0
        self.quantities["obstacle_image"] = dist_trans
        return True

    def set_obstacle_img(self, img):
        self.fixed_obs_img = img
