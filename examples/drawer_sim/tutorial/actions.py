"""
Actions for the Simulated Drawer Examples

The simulation loop reads the action quantities and passes them to the environment, so the components send nothing
themselves.

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import ActionComponent
from aicon_base.derivatives import gradient_descent


class SimulatedAction(ActionComponent):
    """An action whose values the simulation loop reads from `self.quantities`."""

    def _start(self):
        pass

    def _stop(self):
        pass

    def send_action_values(self) -> None:
        pass


class VeloEEAction(SimulatedAction):
    """Translational end-effector velocity, a gradient descent step along the steepest goal gradient."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"action_velo_ee": (3,)}
    v_max = 0.3
    gain = 0.1

    def initialize_quantities(self) -> bool:
        self.quantities["action_velo_ee"] = torch.zeros(3, dtype=self.dtype, device=self.device)
        return True

    def determine_new_actions(self):
        steepest_grad, _, _ = self.get_steepest_gradient("action_velo_ee", time_threshold=self.timestamp - 2.0)
        velocity = gradient_descent(self.gain, self.quantities["action_velo_ee"], steepest_grad[:3])
        if torch.norm(velocity) > self.v_max:
            velocity = velocity / torch.norm(velocity) * self.v_max
        self.quantities["action_velo_ee"] = torch.nan_to_num(velocity)


class GripperAction(SimulatedAction):
    """Gripper activation, 0.01 (open) or 0.99 (closed), switched by the sign of the steepest goal gradient."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"gripper_activation": (1,)}

    def initialize_quantities(self) -> bool:
        self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
        return True

    def determine_new_actions(self):
        steepest_grad, _, _ = self.get_steepest_gradient("gripper_activation")
        if steepest_grad < -1e-15 and self.quantities["gripper_activation"] != 0.99:
            print("Closing the gripper")
            self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.99
        elif steepest_grad > 1e-15 and self.quantities["gripper_activation"] != 0.01:
            print("Opening the gripper")
            self.quantities["gripper_activation"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
