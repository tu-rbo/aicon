"""
Estimators for the Drawer Tutorial

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import (
    update_ekf,
    predict_ekf_other_quantity,
    predict_ekf,
    update_switching_ekf,
    update_shifting_ekf,
)
from aicon_base.inference.util import gradient_preserving_clipping


class EEPoseEstimator(EstimationComponent):
    """End-effector position: predicted with the commanded velocity, corrected by the measured position."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pose_ee": (3,), "uncertainty_ee": (3, 3)}

    def initialize_quantities(self):
        c_proprio = self.connections["DirectMeasurement"]
        with c_proprio.lock:
            if not c_proprio.connected_quantities_initialized["ee_pos_meas"]:
                return False
            pose_measured_ee = c_proprio.connected_quantities["ee_pos_meas"]
        self.quantities["pose_ee"] = pose_measured_ee
        self.quantities["uncertainty_ee"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_proprio = self.connections["DirectMeasurement"].c_func
        c_action = self.connections["ForwardKinematics"].c_func

        def f_func(pose_ee, uncertainty_ee, ee_pos_meas, action_velo_ee):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            mu_pred, Sigma_pred = predict_ekf_other_quantity(
                c_action,
                pose_ee,
                uncertainty_ee,
                action_velo_ee,
                torch.zeros(3, 3, dtype=self.dtype, device=self.device),
                eye * 0.01,
            )
            mu_new, Sigma_new = update_ekf(c_proprio, mu_pred, Sigma_pred, ee_pos_meas, eye * 0.01, eye * 0.01)
            Sigma_new = Sigma_new.detach()
            # The measurement is exact, so the value is set to it; the gradient still flows through the prediction,
            # i.e. towards the commanded velocity.
            mu_new = mu_new + c_proprio(mu_new, ee_pos_meas).detach()
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]


class DrawerPositionEstimator(EstimationComponent):
    """Position of the drawer handle: from the direct measurement, and from the end-effector while grasped."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"position_drawer": (3,), "uncertainty_drawer": (3, 3)}

    def initialize_quantities(self):
        c_meas = self.connections["DrawerDirectMeasurement"]
        with c_meas.lock:
            if not c_meas.connected_quantities_initialized["drawer_pos_meas"]:
                return False
            drawer_pos_meas = c_meas.connected_quantities["drawer_pos_meas"]
        self.quantities["position_drawer"] = drawer_pos_meas
        self.quantities["uncertainty_drawer"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_grasped = self.connections["GraspedDrawerKinematics"].c_func
        c_direct_meas = self.connections["DrawerDirectMeasurement"].c_func

        def forward_drawer(mu):
            return mu  # drawers do not move by themselves

        def f_func(
            position_drawer, uncertainty_drawer, pose_ee, uncertainty_ee, drawer_pos_meas, likelihood_grasped_drawer, dt
        ):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            # a grasped drawer may move with the end-effector
            forward_noise = likelihood_grasped_drawer * 0.15 * dt + 0.005 * dt
            mu_new, Sigma_new = predict_ekf(forward_drawer, position_drawer, uncertainty_drawer, eye * forward_noise)
            mu_new, Sigma_new = update_switching_ekf(
                c_grasped,
                mu_new,
                Sigma_new,
                pose_ee,
                uncertainty_ee,
                eye * 0.03,
                likelihood_grasped_drawer,
                outlier_rejection_threshold=0.05,
                prevent_uncertainty_state_grad=False,
            )
            mu_new, Sigma_new = update_ekf(
                c_direct_meas,
                mu_new,
                Sigma_new,
                drawer_pos_meas,
                eye * 0.03,
                eye * 0.03,
                prevent_uncertainty_state_grad=False,
            )
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["position_drawer", "uncertainty_drawer"], ["position_drawer", "uncertainty_drawer"]


class GraspedEstimator(EstimationComponent):
    """Likelihood that the drawer is grasped."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_grasped_drawer": (1,)}

    def initialize_quantities(self):
        self.quantities["likelihood_grasped_drawer"] = torch.ones(1, dtype=self.dtype, device=self.device) * 0.01
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            innovation = c_func(
                likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation
            )
            new_likelihood = gradient_preserving_clipping(likelihood_grasped_drawer + innovation, 1e-10, 0.9999999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_grasped_drawer"], ["likelihood_grasped_drawer"]


class KinematicJointEstimator(EstimationComponent):
    """The drawer's prismatic joint [azimuth, elevation, joint value, initial point (3)]. While the drawer is not
    grasped the initial point follows the drawer position; while it is grasped the axis and the joint value do."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"kinematic_joint": (6,), "uncertainty_joint": (6, 6)}

    def initialize_quantities(self):
        c = self.connections["DrawerKinematics"]
        with c.lock:
            if not c.connected_quantities_initialized["position_drawer"]:
                return False
            position_drawer = c.connected_quantities["position_drawer"]
            uncertainty_drawer = c.connected_quantities["uncertainty_drawer"]
        joint = torch.zeros(6, dtype=self.dtype, device=self.device)
        joint[0] = -torch.pi / 2 * 0.5  # initial guess of the azimuth
        joint[1] = -torch.pi / 2  # the axis lies in the xy plane
        joint[3:6] = position_drawer
        uncertainty = torch.eye(6, dtype=self.dtype, device=self.device)
        uncertainty[3:6, 3:6] = uncertainty_drawer
        self.quantities["kinematic_joint"] = joint
        self.quantities["uncertainty_joint"] = uncertainty
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["DrawerKinematics"].c_func

        def forward_joint(kinematic_joint):
            return kinematic_joint  # joints do not move by themselves

        R_additive_grasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.02
        R_additive_ungrasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.00005
        Q_diag = torch.tensor([0.001, 0.001, 0.1, 0.02, 0.02, 0.02], device=self.device, dtype=self.dtype)

        def f_func(
            kinematic_joint, uncertainty_joint, position_drawer, uncertainty_drawer, likelihood_grasped_drawer, dt
        ):
            # Below a grasp likelihood of 0.2 the drawer position informs the initial point, above it the axis and
            # the joint value.
            grasped = gradient_preserving_clipping(likelihood_grasped_drawer, 0.000000000001, 1.0)
            not_grasped = gradient_preserving_clipping((0.2 - likelihood_grasped_drawer) / 0.2, 0.0000000001, 1.0)
            shift = torch.diag(torch.cat([grasped, grasped, grasped, not_grasped, not_grasped, not_grasped]))
            shift = gradient_preserving_clipping(shift, 0.0, 1.0)
            mu_new, Sigma_new = predict_ekf(forward_joint, kinematic_joint, uncertainty_joint, shift * Q_diag * dt)
            R_additive = R_additive_grasped * grasped + R_additive_ungrasped * not_grasped
            mu_new, Sigma_new = update_shifting_ekf(
                c_func,
                mu_new,
                Sigma_new,
                position_drawer,
                uncertainty_drawer,
                R_additive,
                shift,
                outlier_rejection_threshold=1.0,
            )
            # axis and joint value are kept independent of the initial point
            Sigma_new = torch.cat(
                [
                    torch.cat([Sigma_new[:3, :3], torch.zeros_like(Sigma_new[:3, :3])], dim=0),
                    torch.cat([torch.zeros_like(Sigma_new[3:, 3:]), Sigma_new[3:, 3:]], dim=0),
                ],
                dim=1,
            )
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["kinematic_joint", "uncertainty_joint"], ["kinematic_joint", "uncertainty_joint"]
