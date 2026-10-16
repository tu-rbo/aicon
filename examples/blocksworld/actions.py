"""
Action Components for the Blocks World Domain

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import random
from typing import Dict, Callable, Union, Tuple

import torch

from aicon_base.components import ActionComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.components.util import collect_derivatives
from .global_params import NUM_BLOCKS


class BlockPuttingAction(ActionComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "action_blocks": (2, NUM_BLOCKS, NUM_BLOCKS),
    }

    def __init__(
        self,
        name: str,
        connections: Dict[str, ActiveInterconnection],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.get_default_device(),
        mockbuild: bool = False,
        send_action_func: Union[None, Dict[str, Callable]] = None,
    ):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.internal_action = None
        self.send_actions = False
        self.wait_for_action_counter = 0
        self.tried_actions = dict()
        self._send_action_func: Union[None, Callable] = send_action_func

    def _start(self):
        self.send_actions = self._send_action_func is not None

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
            return
        if self.wait_for_action_counter > 5:
            oldest_stamp_for_each_grad = torch.cat(
                [torch.min(torch.cat(stamps)).unsqueeze(0) for stamps in my_derivatives.timestamps]
            )
            age_of_newest_grad = self.timestamp - torch.max(oldest_stamp_for_each_grad)
            if age_of_newest_grad > 0.2:
                return
            gradient_steepness = my_derivatives.derivatives_tensor
            with self.connections["BelowLikelihoodDummySensing"].lock:
                current_state = self.connections["BelowLikelihoodDummySensing"].connected_quantities[
                    "below_state_sensed"
                ]
                # keyed by a snapshot: the sensed state tensor is updated in place
                state_key = tuple((current_state > 0.5).flatten().tolist())
                already_tried_mask = self.tried_actions.get(
                    state_key, torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=torch.bool, device=self.device)
                )
                clear = 1 - torch.clip(torch.norm(current_state, dim=0), 0, 1)
            gradient_steepness[:, :, already_tried_mask] = 0.0
            choosen_gradient_idx = None
            # only moves the goals pull towards; invalid ones are zeroed until none is left
            while torch.max(gradient_steepness) > 0:
                possible_steepest_gradient_idxs = (gradient_steepness == torch.max(gradient_steepness)).nonzero()
                n_idxs = possible_steepest_gradient_idxs.shape[0]
                choosen_gradient_idx = possible_steepest_gradient_idxs[random.randint(0, n_idxs - 1)]
                action_idx = choosen_gradient_idx[2:5]
                if action_idx[0] == 0:
                    if (
                        action_idx[1] != action_idx[2]
                        and clear[action_idx[1]]
                        and clear[action_idx[2]]
                        and current_state[action_idx[1], action_idx[2]] == 0
                    ):
                        break
                elif action_idx[0] == 1:
                    if clear[action_idx[1]] and current_state[action_idx[1], action_idx[2]] == 1:
                        break
                gradient_steepness[tuple(choosen_gradient_idx.tolist())] = 0.0
            if choosen_gradient_idx is None or torch.max(gradient_steepness) <= 0:
                self.internal_action = None
                return
            print(
                "Action",
                action_idx.tolist(),
                "following gradient path",
                " > ".join(my_derivatives.quantity_traces[choosen_gradient_idx[0]]),
            )
            self.internal_action = torch.zeros((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device)
            self.internal_action[action_idx[0], action_idx[1], action_idx[2]] = 1
            self.tried_actions[state_key] = torch.logical_or(already_tried_mask, self.internal_action.bool())
            self.wait_for_action_counter = 0
        else:
            self.wait_for_action_counter += 1
            self.internal_action = None

    def initialize_quantities(self) -> bool:
        self.quantities["action_blocks"] = (
            torch.ones((2, NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device) * 1.0
        )
        return True
