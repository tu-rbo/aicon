"""
Actions for the drawer experiment.
This module defines action components for controlling the robot in the drawer manipulation environment,
including end-effector velocity control and hand synergy actions.
"""

import torch
from loguru import logger

from aicon.base_classes.components import gradient_descent
from aicon.base_classes.components import ActionComponent


class VeloEEAction(ActionComponent):
    """
    End-effector velocity control action for the drawer experiment.
    This action component controls the robot's end-effector using translational velocity commands,
    with safety limits and gradient-based optimization.
    """

    state_dim = 3
    v_max_trans = 0.3

    def _start(self):
        """
        Initialize the action by activating impedance control and getting the initial pose.
        """
        # will be handled by the simulation
        pass

    def _stop(self):
        """
        Stop the action by sending zero velocity commands.
        """
        # will be handled by the simulation
        pass

    def determine_new_actions(self):
        """
        Determine new velocity commands based on gradient descent optimization.
        """
        last_action = self.quantities["action_velo_ee"]
        steepest_grad, timestamps, trace = self.get_steepest_gradient("action_velo_ee",
                                                                    time_threshold=self.timestamp - torch.ones_like(self.timestamp) * 2.0)

        t_part = self.perform_gradient_descent(last_action, steepest_grad)

        t_part = self.safety_limiting(t_part)

        new_action_limited = torch.nan_to_num(t_part)


        self.quantities["action_velo_ee"] = new_action_limited

    def perform_gradient_descent(self, last_action: torch.Tensor, steepest_grad: torch.Tensor) -> torch.Tensor:
        """
        Perform gradient descent to update the translational velocity commands.
        
        Args:
            last_action: Previous translational velocity command
            steepest_grad: Steepest gradient for optimization
        
        Returns:
            torch.Tensor: Updated translational velocity (t_part)
        """
        gain = 0.1
        t_part = gradient_descent(gain, last_action[:3], steepest_grad[:3])
        return t_part

    def safety_limiting(self, t_part: torch.Tensor) -> torch.Tensor:
        """
        Apply safety limits to translational velocity commands.
        
        Args:
            t_part: Translational velocity
        
        Returns:
            torch.Tensor: Limited translational velocity (t_part)
        """
        # safety limiting
        if torch.norm(t_part) > self.v_max_trans:
            t_part = t_part / torch.norm(t_part) * self.v_max_trans
        return t_part

    def send_action_values(self) -> None:
        """
        Send the computed velocity commands to the robot controller.
        """
        # We do nothing here because the simulation will handle the velocity commands
        pass

    def initialize_quantities(self) -> bool:
        """
        Initialize the action quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        new_action = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.quantities["action_velo_ee"] = new_action
        return True

    def initial_definitions(self):
        """
        Define initial values for the action quantities.
        """
        self.quantities["action_velo_ee"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)


class GripperAction(ActionComponent):
    """
    Gripper action for controlling the robot's gripper in the drawer experiment.
    This action component manages the gripper's opening and closing based on gradient information.
    """

    state_dim = 1
    
    def _start(self):
        """
        Initialize the action by setting the gripper activation to 0.01.
        """
        pass

    def _stop(self):
        """
        Stop the action by setting the gripper activation to 0.01.
        """
        pass

    def determine_new_actions(self):
        """
        Determine new gripper actions based on gradient information.
        """
        steepest_grad, timestamps, trace = self.get_steepest_gradient("gripper_activation")
        print("Steepest gripper grad", steepest_grad, self.quantities["gripper_activation"])
        if steepest_grad < -1e-15:
            if self.quantities["gripper_activation"] != 0.99:
                print("closing hand")
                self.quantities["gripper_activation"] = 0.99
        elif steepest_grad > 1e-15:
            if self.quantities["gripper_activation"] != 0.01:
                print("opening gripper")
                self.quantities["gripper_activation"] = 0.01

    def initial_definitions(self):
        """
        Define initial values for the gripper quantities.
        """
        self.quantities["gripper_activation"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)

    def initialize_quantities(self) -> bool:
        """
        Initialize the gripper quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        self.quantities["gripper_activation"] = torch.ones(self.state_dim, dtype=self.dtype, device=self.device) * 0.01
        return True

    def send_action_values(self) -> None:
        """
        Send the computed gripper activation to the robot controller.
        """
        # We do nothing here because the simulation will handle the gripper activation
        pass
