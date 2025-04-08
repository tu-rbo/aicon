import random
from typing import Dict, Callable, Union

import torch

from aicon.base_classes.components import ActionComponent
from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.util import collect_derivatives
from aicon.blocksworld_experiment.global_params import NUM_BLOCKS


class BlockPuttingAction(ActionComponent):

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals : Union[None,Dict[str, Callable]] = None,
                 dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device(),
                 mockbuild : bool = False, send_action_func :  Union[None,Dict[str, Callable]] = None):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.internal_action = None
        self.send_actions = False
        self.wait_for_action_counter = 0
        self.tried_actions = dict()
        self._send_action_func = send_action_func

    def _start(self):
        self.send_actions = (self._send_action_func is not None)

    def _stop(self):
        self.send_actions = False

    def send_action_values(self) -> None:
        if self.internal_action is not None and self.send_actions:
            self._send_action_func(self.internal_action)

    def determine_new_actions(self):
        try:
            _, relevant_backward_derivatives = collect_derivatives(self.connections)
            my_derivatives = relevant_backward_derivatives["action_blocks"]
        except KeyError:
            print("No Derivatives for Actions yet!")
            return
        if self.wait_for_action_counter > 5:
            oldest_stamp_for_each_grad = torch.cat(
                [torch.min(torch.cat(stamps)).unsqueeze(0) for stamps in my_derivatives.timestamps])
            age_of_newest_grad = torch.tensor(self.timestamp, dtype=self.dtype, device=self.device) - torch.max(oldest_stamp_for_each_grad)
            if age_of_newest_grad > 0.2:
                return
            gradient_steepness = my_derivatives.derivatives_tensor
            with self.connections["BelowLikelihoodDummySensing"].lock:
                current_state = self.connections["BelowLikelihoodDummySensing"].connected_quantities["below_state_sensed"]
                already_tried_mask = torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=torch.bool, device=self.device)
                for k, v in self.tried_actions.items():
                    if torch.equal(k, current_state):
                        already_tried_mask = v
                clear = 1 - torch.clip(torch.norm(current_state, dim=0), 0, 1)
            gradient_steepness[:, :, already_tried_mask] = 0.0
            choosen_gradient_idx = None
            while torch.sum(torch.abs(gradient_steepness)) > 0:
                # print("Gradients", gradient_steepness)
                possible_steepest_gradient_idxs = (gradient_steepness == torch.max(gradient_steepness)).nonzero()
                n_idxs = possible_steepest_gradient_idxs.shape[0]
                choosen_gradient_idx = possible_steepest_gradient_idxs[random.randint(0, n_idxs-1)]
                action_idx = choosen_gradient_idx[2:5]
                if action_idx[0] == 0:
                    if clear[action_idx[1]] and clear[action_idx[2]] and current_state[action_idx[1], action_idx[2]] == 0:
                        break
                elif action_idx[0] == 1:
                        if clear[action_idx[1]] and current_state[action_idx[1], action_idx[2]] == 1:
                            break
                gradient_steepness[*choosen_gradient_idx] = 0.0
            if choosen_gradient_idx is None:
                print("No gradients available!")
                self.internal_action = None
                return
            print("Current trace", my_derivatives.quantity_traces[choosen_gradient_idx[0]])
            print("Current Action", action_idx)
            print("non abs val", my_derivatives.derivatives_tensor[*choosen_gradient_idx])
            self.internal_action = torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device)
            self.internal_action[*action_idx] = 1
            self.tried_actions[current_state] = torch.logical_or(already_tried_mask, self.internal_action)
            print(self.internal_action)
            self.wait_for_action_counter = 0
        else:
            self.wait_for_action_counter += 1
            self.internal_action = None

    def initial_definitions(self):
        self.quantities["action_blocks"] = torch.zeros((2, NUM_BLOCKS,NUM_BLOCKS), dtype=self.dtype, device=self.device)

    def initialize_quantities(self) -> bool:
        self.quantities["action_blocks"] = torch.ones((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device) * 1.0
        return True