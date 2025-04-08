from typing import Dict, Union

import torch

from aicon.base_classes.components import SensorComponent
from aicon.base_classes.connections import ActiveInterconnection
from aicon.blocksworld_experiment.global_params import NUM_BLOCKS


class BlocksBelowSensor(SensorComponent):
    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, setup: str = "3towers", ROS_MODE : bool = False):
        self.quantity_name = "below_state_sensed"
        super().__init__(name, connections, dtype, device, mockbuild)
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
            case "0towers":
                pass
        self.quantities[self.quantity_name] = self.initial_state
        self.current_sim_state = self.initial_state
        self.sim_timestamp = self.timestamp
        self.taken_valid_actions = 0
        if mockbuild:
            return

        if ROS_MODE:
            from aicon.blocksworld_experiment.old_ros_action_communication import register_ros_action_callback
            register_ros_action_callback(self.take_action, dtype=self.dtype, device=self.device)

    def obtain_measurements(self) -> bool:
        self.timestamp = torch.tensor(self.timestamp, dtype=self.dtype, device=self.device)
        self.quantities[self.quantity_name] = self.current_sim_state
        return True

    def take_action(self, action_value):
        if torch.max(action_value) == 1:
            # we have an action we would like to do
            idx = torch.unravel_index(torch.argmax(torch.abs(action_value)),
                                      action_value.shape)
            which_field = idx[1:3]
            clear = 1 - torch.clip(torch.norm(self.current_sim_state, dim=0), 0, 1)
            if idx[0] == 0:
                # put this block on top
                if self.current_sim_state[which_field] == 0:
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
            print("Now taken actions " + str(self.taken_valid_actions))
            print(self.current_sim_state)

    def initial_definitions(self) -> None:
        self.quantities[self.quantity_name] = torch.empty((NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device)