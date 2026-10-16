"""
Sensor Components for the Blocks World Domain, incl. basic Simulator for the Domain

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Union, Type

import torch

from aicon_base.components import SensorComponent
from aicon_base.connections import ActiveInterconnection
from .global_params import NUM_BLOCKS


class BlocksBelowSensor(SensorComponent):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        setup: str = "3towers",
    ):
        self.quantity_name = "below_state_sensed"
        self.cls_quantity_names_and_shapes = {self.quantity_name: (NUM_BLOCKS, NUM_BLOCKS)}
        super().__init__(name, connections, dtype=dtype, device=device, mockbuild=mockbuild)
        self.initial_state = torch.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=dtype, device=device)
        match setup:
            case "3towers":
                self.initial_state[1, 0] = 1.0
                self.initial_state[2, 1] = 1.0
                self.initial_state[2, 0] = 1.0
                self.initial_state[3, 4] = 1.0
                self.initial_state[7, 6] = 1.0
                self.initial_state[8, 6] = 1.0
                self.initial_state[8, 7] = 1.0
                self.initial_state[9, 6] = 1.0
                self.initial_state[9, 7] = 1.0
                self.initial_state[9, 8] = 1.0
            case "3towersAlt":
                self.initial_state[1, 0] = 1.0
                self.initial_state[2, 1] = 1.0
                self.initial_state[2, 0] = 1.0
                self.initial_state[4, 3] = 1.0
                self.initial_state[7, 6] = 1.0
                self.initial_state[8, 6] = 1.0
                self.initial_state[8, 7] = 1.0
                self.initial_state[6, 9] = 1.0
                self.initial_state[7, 9] = 1.0
                self.initial_state[8, 9] = 1.0
            case "2towers":
                self.initial_state[1, 0] = 1.0
                self.initial_state[2, 1] = 1.0
                self.initial_state[2, 0] = 1.0
                self.initial_state[4, 2] = 1.0
                self.initial_state[4, 1] = 1.0
                self.initial_state[4, 0] = 1.0
                self.initial_state[5, 4] = 1.0
                self.initial_state[5, 2] = 1.0
                self.initial_state[5, 1] = 1.0
                self.initial_state[5, 0] = 1.0
                self.initial_state[7, 6] = 1.0
                self.initial_state[8, 6] = 1.0
                self.initial_state[8, 7] = 1.0
                self.initial_state[9, 6] = 1.0
                self.initial_state[9, 7] = 1.0
                self.initial_state[9, 8] = 1.0
            case "0towers":
                pass
            case "5towers":
                self.initial_state[1, 0] = 1.0
                self.initial_state[3, 2] = 1.0
                self.initial_state[5, 4] = 1.0
                self.initial_state[7, 6] = 1.0
                self.initial_state[9, 8] = 1.0
            case "2towersAlt":
                self.initial_state[1, 6] = 1.0
                self.initial_state[2, 1] = 1.0
                self.initial_state[2, 6] = 1.0
                self.initial_state[4, 2] = 1.0
                self.initial_state[4, 1] = 1.0
                self.initial_state[4, 6] = 1.0
                self.initial_state[5, 4] = 1.0
                self.initial_state[5, 2] = 1.0
                self.initial_state[5, 1] = 1.0
                self.initial_state[5, 6] = 1.0
                self.initial_state[7, 0] = 1.0
                self.initial_state[8, 0] = 1.0
                self.initial_state[8, 7] = 1.0
                self.initial_state[9, 0] = 1.0
                self.initial_state[9, 7] = 1.0
                self.initial_state[9, 8] = 1.0
            case _:
                raise ValueError(f"unknown setup {setup}")
        self.quantities[self.quantity_name] = self.initial_state
        self.current_sim_state = self.initial_state
        self.sim_timestamp = self.timestamp
        self.taken_valid_actions = 0
        if mockbuild:
            return

    def obtain_measurements(self) -> bool:
        self.timestamp = self.timestamp.clone()
        self.quantities[self.quantity_name] = self.current_sim_state
        return True

    def take_action(self, action_value):
        if torch.max(action_value) == 1:
            # we have an action we would like to do
            idx = torch.unravel_index(torch.argmax(torch.abs(action_value)), action_value.shape)
            which_field = idx[1:3]
            clear = 1 - torch.clip(torch.norm(self.current_sim_state, dim=0), 0, 1)
            if idx[0] == 0:
                # put this block on top (of another block)
                if self.current_sim_state[which_field] == 0 and which_field[0] != which_field[1]:
                    if clear[which_field[0]] == 1 and clear[which_field[1]] == 1:
                        self.taken_valid_actions += 1
                        self.current_sim_state[which_field] = 1
                        others_below = self.current_sim_state[which_field[1], :]
                        for i in range(NUM_BLOCKS):
                            if others_below[i] == 1:
                                self.current_sim_state[which_field[0], i] = 1
            elif idx[0] == 1:
                # remove it
                if self.current_sim_state[which_field] == 1:
                    if clear[which_field[0]]:
                        self.taken_valid_actions += 1
                        self.current_sim_state[which_field] = 0
                        others_below = self.current_sim_state[which_field[0], :]
                        for i in range(NUM_BLOCKS):
                            if others_below[i] == 1:
                                self.current_sim_state[which_field[0], i] = 0
