"""
AICON Estimators for EE Pose

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import predict_ekf_other_quantity, update_ekf


class CartesianControlEEPoseEstimator(EstimationComponent):
    state_dim = 6
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pose_ee": (6,), "uncertainty_ee": (6, 6)}

    def define_estimation_function_f(self):
        c_proprio = self.connections["DirectMeasurementEE"].c_func
        c_action = self.connections["RBForwardKinematics"].c_func

        def f_func(pose_ee, uncertainty_ee, pose_measured_ee, action_velo_ee):
            mu_pred, Sigma_pred = predict_ekf_other_quantity(
                c_action,
                pose_ee,
                uncertainty_ee,
                action_velo_ee,
                torch.zeros(6, 6, dtype=self.dtype, device=self.device),
                torch.eye(6, dtype=self.dtype, device=self.device) * 0.01,
            )
            mu_new, Sigma_new = update_ekf(
                c_proprio,
                mu_pred,
                Sigma_pred,
                pose_measured_ee,
                torch.eye(6, dtype=self.dtype, device=self.device) * 0.01,
                torch.eye(6, dtype=self.dtype, device=self.device) * 0.01,
            )
            Sigma_new = Sigma_new.detach()  # currently no grad possible
            # snap the mean onto the measurement
            delta = c_proprio(mu_new, pose_measured_ee).detach()
            mu_new = mu_new + delta
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        self.differentiate_against = ("action_velo_ee",)
        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]

    def initialize_quantities(self):
        c_proprio = self.connections["DirectMeasurement"]
        with c_proprio.lock:
            if c_proprio.connected_quantities_initialized["pose_measured_ee"]:
                pose_measured_ee = c_proprio.connected_quantities["pose_measured_ee"]
            else:
                return False
        self.quantities["pose_ee"] = pose_measured_ee
        self.quantities["uncertainty_ee"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * 0.001
        return True


class JointControlEEPoseEstimator(CartesianControlEEPoseEstimator):
    def define_estimation_function_f(self):
        c_joints = self.connections["RobotForwardKinematics"].c_func

        self.alpha["pose_ee"] = 1.0

        def f_func(pose_ee, uncertainty_ee, robot_joint_states):
            mu_new = pose_ee + c_joints(pose_ee, robot_joint_states)
            Sigma_new = uncertainty_ee
            return (mu_new,), (mu_new, Sigma_new)

        self.differentiate_against = ("robot_joint_states",)
        return (
            f_func,
            [
                "pose_ee",
            ],
            ["pose_ee", "uncertainty_ee"],
        )

    def initialize_quantities(self):
        c_joints = self.connections["RobotForwardKinematics"]
        with c_joints.lock:
            if c_joints.connected_quantities_initialized["robot_joint_states"]:
                current_joints = c_joints.connected_quantities["robot_joint_states"]
                pose_ee = c_joints.c_func(
                    torch.zeros(self.state_dim, dtype=self.dtype, device=self.device), current_joints
                )
            else:
                return False
        self.quantities["pose_ee"] = pose_ee
        self.quantities["uncertainty_ee"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * 0.001
        return True
