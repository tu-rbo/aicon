"""
Actions for the drawer experiment.
This module defines action components for controlling the robot in the drawer manipulation environment,
including end-effector velocity control and hand synergy actions.
"""

import torch

from aicon.actions.panda_controllers import BasicPandaControlAction
from aicon.actions.rbo_hand import RBOHandSynergyAction
from aicon.base_classes.components import gradient_descent
from aicon.math.util_3d import exponential_map_se3, log_map_se3


class VeloEEAction(BasicPandaControlAction):
    """
    End-effector velocity control action for the drawer experiment.
    This action component controls the robot's end-effector using velocity commands,
    with safety limits and gradient-based optimization.
    """

    state_dim = 6

    def _start(self):
        """
        Initialize the action by activating impedance control and getting the initial pose.
        """
        self.cf_controller.activate_blackboard_impedance_vel_ha()
        self.pose_sensor.update(self.timestamp)
        self.start_pose = exponential_map_se3(self.pose_sensor.quantities["pose_measured_ee_internal"])

    def _stop(self):
        """
        Stop the action by sending zero velocity commands.
        """
        self.cf_controller.send_new_blackboard_goal(torch.zeros(6).cpu().numpy(),
                                                  torch.zeros(6).cpu().numpy(),
                                                  self.comp_time,
                                                  stiff_velo=True,
                                                  compliant_pos=False)

    def determine_new_actions(self):
        """
        Determine new velocity commands based on gradient descent optimization.
        """
        last_action = self.quantities["action_velo_ee"]
        steepest_grad, timestamps, trace = self.get_steepest_gradient("action_velo_ee",
                                                                    time_threshold=self.timestamp - torch.ones_like(self.timestamp) * 2.0)

        r_part, t_part = self.perform_gradient_descent(last_action, steepest_grad)

        # if action is 0
        # if torch.norm(last_action) == 0:
        #     # generate some random action
        #     t_part, r_part = self.generate_random_action()

        t_part, r_part = self.safety_limiting(t_part, r_part)

        # r_part = torch.zeros_like(r_part)
        new_action_limited = torch.nan_to_num(torch.concatenate([t_part, r_part]))

        # print("Last Action")
        # print(last_action)
        # print("Gradient")
        # print(steepest_grad)
        # print(trace)
        # print("New Action")
        # print(new_action_limited)

        self.quantities["action_velo_ee"] = new_action_limited

    def perform_gradient_descent(self, last_action: torch.Tensor, steepest_grad: torch.Tensor) -> tuple:
        """
        Perform gradient descent to update the velocity commands.
        
        Args:
            last_action: Previous velocity command
            steepest_grad: Steepest gradient for optimization
        
        Returns:
            tuple: (r_part, t_part) where r_part is rotational velocity and t_part is translational velocity
        """
        gain = 0.1
        t_part = gradient_descent(gain, last_action[:3], steepest_grad[:3])
        if torch.norm(steepest_grad[3:]) == 0 and torch.norm(last_action[3:]) > 0:
            r_part = last_action[3:]
        else:
            r_part = gradient_descent(gain, last_action[3:], steepest_grad[3:])
        if torch.sum(t_part) > 0 and torch.sum(r_part) == 0:
            r_part = torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
        return r_part, t_part

    def generate_random_action(self) -> tuple:
        """
        Generate a random velocity command for exploration.
        
        Returns:
            tuple: (t_part, r_part) random translational and rotational velocities
        """
        t_part = torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
        r_part = torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
        return t_part, r_part

    def safety_limiting(self, t_part: torch.Tensor, r_part: torch.Tensor) -> tuple:
        """
        Apply safety limits to velocity commands.
        
        Args:
            t_part: Translational velocity
            r_part: Rotational velocity
        
        Returns:
            tuple: (t_part, r_part) limited velocities
        """
        # safety limiting
        if torch.norm(t_part) > self.v_max_trans:
            t_part = t_part / torch.norm(t_part) * self.v_max_trans
        if torch.norm(r_part) > self.v_max_rot:
            r_part = r_part / torch.norm(r_part) * self.v_max_rot
        return t_part, r_part

    def send_action_values(self) -> None:
        """
        Send the computed velocity commands to the robot controller.
        """
        dt = self.comp_time
        self.pose_sensor.update(self.timestamp)
        pose_ee = self.pose_sensor.quantities["pose_measured_ee_internal"]
        new_pose = torch.einsum("ij,jk->ik", exponential_map_se3(pose_ee), exponential_map_se3(self.quantities["action_velo_ee"] * dt))
        # prevent drift while only x,z control
        # new_pose[3, 1] = self.start_pose[3, 1]
        # new_pose[:3,:3] = self.start_pose[:3,:3]
        # print(exponential_map_se3(pose_ee))
        # print(new_pose)

        if not torch.any(torch.isnan(new_pose)):
            self.cf_controller.send_new_blackboard_goal(log_map_se3(new_pose).cpu().numpy(),
                                                      torch.zeros(6).cpu().numpy(),
                                                      self.comp_time,
                                                      stiff_velo=True,
                                                      compliant_pos=False)
        # # get actual current velocity
        # filtered_states = self.cf_controller.get_filtered_states()
        # self.quantities["action_velo_ee"] = torch.tensor(filtered_states["base_v_ee"], device=self.device, dtype=self.dtype)

    def initialize_quantities(self) -> bool:
        """
        Initialize the action quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        new_action = torch.zeros(6, dtype=self.dtype, device=self.device)
        self.quantities["action_velo_ee"] = new_action
        return True

    def initial_definitions(self):
        """
        Define initial values for the action quantities.
        """
        self.quantities["action_velo_ee"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)


class HandSynergyAction(RBOHandSynergyAction):
    """
    Hand synergy action for controlling the robot's hand in the drawer experiment.
    This action component manages the hand's opening and closing based on gradient information.
    """

    state_dim = 1

    def determine_new_actions(self):
        """
        Determine new hand synergy actions based on gradient information.
        """
        steepest_grad, timestamps, trace = self.get_steepest_gradient("hand_synergy_activation")
        print("Steepest hand grad", steepest_grad, self.quantities["hand_synergy_activation"])
        if steepest_grad < -0.00000001:
            if self.quantities["hand_synergy_activation"] != 0.99:
                print("closing hand")
                self.next_controller_cmd = "close_base_tip_sequentially_movement_fast_plus_thumb"
                self.quantities["hand_synergy_activation"] = 0.99
        elif steepest_grad > 0.000000000000000001:
            if self.quantities["hand_synergy_activation"] != 0.01:
                print("opening hand")
                self.next_controller_cmd = "reset"
                self.quantities["hand_synergy_activation"] = 0.01

    def initial_definitions(self):
        """
        Define initial values for the hand synergy quantities.
        """
        self.quantities["hand_synergy_activation"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)

    def initialize_quantities(self) -> bool:
        """
        Initialize the hand synergy quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        self.quantities["hand_synergy_activation"] = torch.ones(self.state_dim, dtype=self.dtype, device=self.device) * 0.01
        return True