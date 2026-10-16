"""
Actions for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Tuple, Dict

import torch

from aicon_base.components import ActionComponent
from .env import env_instance


class PusherAction(ActionComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "pusher_point_action": (2,),
    }

    def _start(self):
        pass

    def _stop(self):
        pass

    def send_action_values(self) -> None:
        action = self.quantities["pusher_point_action"].cpu().numpy()
        env_instance.perform_action(env_instance.observation["agent_pos"] + action)

    def determine_new_actions(self):
        gain = 10
        last_action = self.quantities["pusher_point_action"]
        steepest_grad = self.get_steepest_plus_projected("pusher_point_action")
        noise = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) * 20).sample()
        if torch.norm(steepest_grad) == 0:
            # no gradient yet: explore with the noise alone
            new_action = last_action + noise
        else:
            new_action = last_action - gain * steepest_grad / torch.norm(steepest_grad) + noise
        new_action = new_action / torch.norm(new_action) * 2.0
        self.quantities["pusher_point_action"] = new_action

    def initialize_quantities(self) -> bool:
        self.quantities["pusher_point_action"] = torch.zeros(2, dtype=self.dtype, device=self.device) * 1.0
        return True
