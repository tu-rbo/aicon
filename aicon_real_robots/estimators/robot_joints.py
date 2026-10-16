"""
AICON Estimators for Robot Joint State

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch
from typing import Dict, Tuple

from aicon_base.components import EstimationComponent


class JointStateEstimator(EstimationComponent):
    state_dim = 7
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"robot_joint_states": (7,)}

    def initialize_quantities(self):
        if not self.connections["DirectMeasurementJoints"].connected_quantities_initialized["robot_joint_states_meas"]:
            return False
        self.quantities["robot_joint_states"] = self.connections["DirectMeasurementJoints"].connected_quantities[
            "robot_joint_states_meas"
        ]
        return True

    def define_estimation_function_f(self):
        c_proprio = self.connections["DirectMeasurementJoints"].c_func
        c_action = self.connections["LinearVelocityJoints"].c_func

        self.alpha["robot_joint_states"] = 1.0

        def f_func(robot_joint_states, robot_joint_states_meas, action_velo_joints):
            robot_joint_states_pred = c_action(robot_joint_states, action_velo_joints)
            robot_joint_states_delta = c_proprio(robot_joint_states_pred, robot_joint_states_meas).detach()
            robot_joint_states_new = robot_joint_states_pred + robot_joint_states_delta
            return (robot_joint_states_new,), (robot_joint_states_new,)

        self.differentiate_against = ("action_velo_joints",)
        return f_func, ["robot_joint_states"], ["robot_joint_states"]


class JointLimitEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_joint_at_limit": (7,),
        "joint_close_to_limit": (7,),
    }

    def initialize_quantities(self):
        if not self.connections["JointLimitLikelihoodConnection"].connected_quantities_initialized[
            "robot_joint_states"
        ]:
            return False
        self.quantities["likelihood_joint_at_limit"] = -self.connections["JointLimitLikelihoodConnection"].c_func(
            self.connections["JointLimitLikelihoodConnection"].connected_quantities["robot_joint_states"],
            torch.zeros((7,), dtype=self.dtype, device=self.device),
            torch.zeros((7,), dtype=self.dtype, device=self.device),
        )
        self.quantities["joint_close_to_limit"] = (self.quantities["likelihood_joint_at_limit"] != 0).to(
            dtype=self.dtype, device=self.device
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["JointLimitLikelihoodConnection"].c_func

        def f_func(likelihood_joint_at_limit, joint_close_to_limit, robot_joint_states):
            new_likelihood = likelihood_joint_at_limit - c_func(
                robot_joint_states, likelihood_joint_at_limit, joint_close_to_limit
            )
            joint_close_to_limit_new = (new_likelihood != 0).to(dtype=self.dtype, device=self.device)
            return (new_likelihood,), (new_likelihood, joint_close_to_limit_new)

        self.differentiate_against = ("robot_joint_states",)
        return f_func, ["likelihood_joint_at_limit"], ["likelihood_joint_at_limit", "joint_close_to_limit"]


class PostureEstimator(EstimationComponent):

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"joint_range_utilization": (7,)}

    def initialize_quantities(self):
        if not self.connections["JointRangeUtilization"].connected_quantities_initialized["robot_joint_states"]:
            return False
        self.quantities["joint_range_utilization"] = -self.connections["JointRangeUtilization"].c_func(
            torch.zeros((7,), dtype=self.dtype, device=self.device),
            self.connections["JointRangeUtilization"].connected_quantities["robot_joint_states"],
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["JointRangeUtilization"].c_func

        def f_func(joint_range_utilization, robot_joint_states):
            new_utilization = joint_range_utilization - c_func(joint_range_utilization, robot_joint_states)
            return (new_utilization,), (new_utilization,)

        self.differentiate_against = ("robot_joint_states",)
        return f_func, ["joint_range_utilization"], ["joint_range_utilization"]
