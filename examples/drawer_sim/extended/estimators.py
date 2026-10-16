"""
Estimators for the Extended Drawer Example

Authors: James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import math
from typing import Dict, Callable, Tuple

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import (
    update_ekf,
    predict_ekf_other_quantity,
    predict_ekf,
    update_switching_ekf_triple_connection,
    update_switching_ekf,
    update_shifting_ekf,
)
from aicon_base.inference.util import gradient_preserving_clipping
from aicon_base.math.se3_and_SE3 import homogeneous_transform_inverse
from ..util import ee_to_camera, likelihood_dist_func, likelihood_func_visible, pose_vec_to_homogeneous


class EEPoseEstimator(EstimationComponent):
    """End-effector pose [x, y, z, rx, ry, rz]: predicted with the commanded velocity, corrected by the measured
    pose."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pose_ee": (6,), "uncertainty_ee": (6, 6)}

    def initialize_quantities(self):
        c_proprio = self.connections["DirectMeasurement"]
        with c_proprio.lock:
            if not c_proprio.connected_quantities_initialized["ee_pos_meas"]:
                return False
            pose_measured_ee = c_proprio.connected_quantities["ee_pos_meas"]
        self.quantities["pose_ee"] = pose_measured_ee
        self.quantities["uncertainty_ee"] = torch.eye(6, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        c_proprio = self.connections["DirectMeasurement"].c_func
        c_action = self.connections["ForwardKinematics"].c_func

        def f_func(pose_ee, uncertainty_ee, ee_pos_meas, action_velo_ee):
            eye = torch.eye(6, dtype=self.dtype, device=self.device)
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
            # the value is set to the measurement; the gradient still flows through the prediction
            mu_new = mu_new + c_proprio(mu_new, ee_pos_meas).detach()
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]


class DrawerPositionEstimator(EstimationComponent):
    """Position of the drawer handle, from the camera and from the end-effector while grasped.

    With `triangulation`, the camera measures only the bearing to the handle. The estimate is initialised along the
    first bearing at a depth prior (the true depth plus Gaussian noise of `depth_prior_noise` [m], with a large
    uncertainty in all directions) and is corrected by bearings from the viewpoints the robot moves through. Bearing
    updates are weighted by the likelihood that the handle is visible and not occluded by the gripper (i.e. not
    grasped), and their noise grows when the handle is outside the field of view or far from 0.5 m.

    Without triangulation, the handle position is measured directly.
    """

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"position_drawer": (3,), "uncertainty_drawer": (3, 3)}

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        env,
        triangulation: bool = True,
        depth_prior_noise: float = 0.01,
        **kwargs,
    ):
        self.env = env
        self.triangulation = triangulation
        self.depth_prior_noise = depth_prior_noise
        super().__init__(name, connections, **kwargs)

    def initialize_quantities(self):
        if not self.triangulation:
            c_meas = self.connections["DrawerDirectMeasurement"]
            with c_meas.lock:
                if not c_meas.connected_quantities_initialized["drawer_pos_meas"]:
                    return False
                drawer_pos_meas = c_meas.connected_quantities["drawer_pos_meas"]
            self.quantities["position_drawer"] = drawer_pos_meas
            self.quantities["uncertainty_drawer"] = torch.eye(3, dtype=self.dtype, device=self.device) * 0.001
            return True

        c_cam = self.connections["ProjectiveGeometry"]
        with c_cam.lock:
            if not (
                c_cam.connected_quantities_initialized["relative_position_in_CF_drawer"]
                and c_cam.connected_quantities_initialized["pose_ee"]
            ):
                return False
            pose_ee = c_cam.connected_quantities["pose_ee"]
            measured_bearing = c_cam.connected_quantities["relative_position_in_CF_drawer"].clone()
        if torch.any(torch.isnan(measured_bearing)) or measured_bearing[2] <= torch.finfo(self.dtype).eps:
            return False
        H_ee_to_cam = ee_to_camera(self.dtype, self.device)
        H_world_to_cam = torch.einsum("ij,jk->ik", pose_vec_to_homogeneous(pose_ee), H_ee_to_cam)
        # depth prior: the handle's true depth along the optical axis, with noise
        handle = torch.as_tensor(self.env.env.get_drawer_handle_pos(), dtype=self.dtype, device=self.device)
        handle_in_camera = torch.einsum(
            "ki,ij,j->k",
            homogeneous_transform_inverse(H_ee_to_cam),
            homogeneous_transform_inverse(pose_vec_to_homogeneous(pose_ee)),
            torch.cat([handle, torch.ones(1, dtype=self.dtype, device=self.device)]),
        )[:3]
        depth = handle_in_camera[2] + self.depth_prior_noise * torch.randn((), dtype=self.dtype, device=self.device)
        point_in_camera = measured_bearing * (depth / measured_bearing[2])
        self.quantities["position_drawer"] = torch.einsum(
            "ij,j->i", H_world_to_cam, torch.cat([point_in_camera, torch.ones(1, dtype=self.dtype, device=self.device)])
        )[:3]
        # uncertainty of 1.5 m^2 in every direction of the camera frame
        R = H_world_to_cam[:3, :3]
        self.quantities["uncertainty_drawer"] = torch.einsum(
            "ij,jk,kl->il", R, torch.eye(3, dtype=self.dtype, device=self.device) * 1.5, R.transpose(0, 1)
        )
        return True

    def define_estimation_function_f(self):
        c_grasped = self.connections["GraspedDrawerKinematics"].c_func
        if self.triangulation:
            c_cam = self.connections["ProjectiveGeometry"].c_func
        else:
            c_direct_meas = self.connections["DrawerDirectMeasurement"].c_func
        H_ee_to_cam = ee_to_camera(self.dtype, self.device)

        def forward_drawer(mu):
            return mu  # drawers do not move by themselves

        def predict_and_grasp(
            mu,
            Sigma,
            pose_ee,
            uncertainty_ee,
            likelihood_grasped_drawer,
            time_since_hand_change,
            dt,
            measurement_update,
        ):
            eye = torch.eye(3, dtype=self.dtype, device=self.device)
            # a grasped drawer may move with the end-effector
            forward_noise = likelihood_grasped_drawer * 10.0 * dt + 0.005 * dt
            mu, Sigma = predict_ekf(forward_drawer, mu, Sigma, eye * forward_noise)
            mu, Sigma = measurement_update(mu, Sigma)
            mu, Sigma = update_switching_ekf(
                c_grasped,
                mu,
                Sigma,
                pose_ee,
                uncertainty_ee,
                eye * 1e-6,
                likelihood_grasped_drawer,
                outlier_rejection_threshold=0.05,
                prevent_uncertainty_state_grad=False,
            )
            # shortly after opening the gripper the grasp may be lost
            if time_since_hand_change[0] < 0.5 and time_since_hand_change[1] == 0 and likelihood_grasped_drawer != 0.0:
                Sigma = Sigma + eye * dt
            return (mu, Sigma), (mu, Sigma)

        if self.triangulation:

            def f_func(
                position_drawer,
                uncertainty_drawer,
                pose_ee,
                uncertainty_ee,
                relative_position_in_CF_drawer,
                likelihood_grasped_drawer,
                likelihood_visible_drawer,
                time_since_hand_change,
                dt,
            ):
                def bearing_update(mu, Sigma):
                    eye = torch.eye(3, dtype=self.dtype, device=self.device)
                    R_add = 5.0 * (
                        eye * (1.0 - likelihood_func_visible(pose_ee, mu, H_ee_to_cam, steepness=5.0))
                        + eye * (1.0 - likelihood_dist_func(pose_ee, mu, H_ee_to_cam))
                    )
                    # a usable bearing needs the handle in view and not hidden by the gripper
                    likelihood_measurement = likelihood_visible_drawer * (1.0 - likelihood_grasped_drawer.detach())
                    return update_switching_ekf_triple_connection(
                        c_cam,
                        mu,
                        Sigma,
                        pose_ee,
                        uncertainty_ee,
                        relative_position_in_CF_drawer,
                        eye * 0.025,
                        R_add,
                        likelihood_measurement,
                    )

                return predict_and_grasp(
                    position_drawer,
                    uncertainty_drawer,
                    pose_ee,
                    uncertainty_ee,
                    likelihood_grasped_drawer,
                    time_since_hand_change,
                    dt,
                    bearing_update,
                )
        else:

            def f_func(
                position_drawer,
                uncertainty_drawer,
                pose_ee,
                uncertainty_ee,
                drawer_pos_meas,
                likelihood_grasped_drawer,
                time_since_hand_change,
                dt,
            ):
                def direct_update(mu, Sigma):
                    eye = torch.eye(3, dtype=self.dtype, device=self.device)
                    return update_ekf(
                        c_direct_meas,
                        mu,
                        Sigma,
                        drawer_pos_meas,
                        eye * 0.03,
                        eye * 0.03,
                        prevent_uncertainty_state_grad=False,
                    )

                return predict_and_grasp(
                    position_drawer,
                    uncertainty_drawer,
                    pose_ee,
                    uncertainty_ee,
                    likelihood_grasped_drawer,
                    time_since_hand_change,
                    dt,
                    direct_update,
                )

        return f_func, ["position_drawer", "uncertainty_drawer"], ["position_drawer", "uncertainty_drawer"]


class DistEEDrawerEstimator(EstimationComponent):
    """Distance between end-effector and handle, with the summed uncertainty of both positions."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"distance_ee_drawer": (2,), "uncertainty_dist": (1,)}

    def initialize_quantities(self):
        c_dist = self.connections["E[dist]"]
        with c_dist.lock:
            if not (
                c_dist.connected_quantities_initialized["position_drawer"]
                and c_dist.connected_quantities_initialized["pose_ee"]
            ):
                return False
            pose_ee = c_dist.connected_quantities["pose_ee"]
            position_drawer = c_dist.connected_quantities["position_drawer"]
            uncertainty_ee = c_dist.connected_quantities["uncertainty_ee"]
            uncertainty_drawer = c_dist.connected_quantities["uncertainty_drawer"]
        self.quantities["distance_ee_drawer"] = c_dist.c_func(
            torch.zeros(2, dtype=self.dtype, device=self.device), pose_ee, position_drawer
        )
        self.quantities["uncertainty_dist"] = (torch.trace(uncertainty_drawer) + torch.trace(uncertainty_ee)).reshape(1)
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["E[dist]"].c_func

        def f_func(distance_ee_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            new_dist = distance_ee_drawer + c_func(distance_ee_drawer, pose_ee, position_drawer)
            uncertainty = (torch.trace(uncertainty_drawer) + torch.trace(uncertainty_ee)).reshape(1)
            return (new_dist, uncertainty), (new_dist, uncertainty)

        return f_func, ["distance_ee_drawer", "uncertainty_dist"], ["distance_ee_drawer", "uncertainty_dist"]


class GraspedEstimator(EstimationComponent):
    """Likelihood that the drawer is grasped, and the time since the gripper last opened / closed
    (time_since_hand_change = [time open, time closed])."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_grasped_drawer": (1,),
        "time_since_hand_change": (2,),
    }

    def initialize_quantities(self):
        c = self.connections["GraspedLikelihood"]
        with c.lock:
            if not all(
                c.connected_quantities_initialized[k]
                for k in ("distance_ee_drawer", "gripper_activation", "ee_force_mag_meas", "uncertainty_dist")
            ):
                return False
            distance_ee_drawer = c.connected_quantities["distance_ee_drawer"]
            gripper_activation = c.connected_quantities["gripper_activation"]
            uncertainty_dist = c.connected_quantities["uncertainty_dist"]
        likelihood = c.c_func(
            torch.tensor([0.05], dtype=self.dtype, device=self.device),
            distance_ee_drawer,
            uncertainty_dist,
            gripper_activation,
            torch.zeros(3, dtype=self.dtype, device=self.device),
            torch.zeros(2, dtype=self.dtype, device=self.device),
        )
        self.quantities["likelihood_grasped_drawer"] = torch.clip(likelihood, 0.02, 0.98)
        # the gripper starts open (for 2 s, so a grasp attempt is possible right away)
        self.quantities["time_since_hand_change"] = torch.tensor([2.0, 0.0], dtype=self.dtype, device=self.device)
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(
            likelihood_grasped_drawer,
            distance_ee_drawer,
            gripper_activation,
            ee_force_mag_meas,
            uncertainty_dist,
            time_since_hand_change,
            dt,
        ):
            time_since_hand_change_new = torch.clone(time_since_hand_change).detach()
            if gripper_activation > 0.5:
                time_since_hand_change_new[0] = 0.0
                time_since_hand_change_new[1] = dt + time_since_hand_change[1]
            else:
                time_since_hand_change_new[0] = time_since_hand_change[0] + dt
                time_since_hand_change_new[1] = 0.0
            innovation = c_func(
                likelihood_grasped_drawer,
                distance_ee_drawer,
                uncertainty_dist,
                gripper_activation,
                ee_force_mag_meas,
                time_since_hand_change_new,
            )
            if innovation < -0.05:
                # losing the grasp is concluded slowly
                new_likelihood = likelihood_grasped_drawer + 0.1 * innovation
            else:
                new_likelihood = likelihood_grasped_drawer + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 1e-10, 0.9999999)
            return (new_likelihood,), (new_likelihood, time_since_hand_change_new)

        return f_func, ["likelihood_grasped_drawer"], ["likelihood_grasped_drawer", "time_since_hand_change"]


class VisibleEstimator(EstimationComponent):
    """Likelihood that the handle is in the wrist camera's field of view."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_visible_drawer": (1,)}

    def initialize_quantities(self):
        c = self.connections["VisibleLikelihood"]
        with c.lock:
            if not all(
                c.connected_quantities_initialized[k]
                for k in ("pose_ee", "position_drawer", "uncertainty_ee", "uncertainty_drawer")
            ):
                return False
            pose_ee = c.connected_quantities["pose_ee"]
            position_drawer = c.connected_quantities["position_drawer"]
        likelihood = c.c_func(torch.tensor([0.95], dtype=self.dtype, device=self.device), pose_ee, position_drawer)
        self.quantities["likelihood_visible_drawer"] = torch.clip(likelihood, 0.02, 0.98)
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["VisibleLikelihood"].c_func

        def f_func(likelihood_visible_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            new_likelihood = likelihood_visible_drawer + 2.2 * c_func(
                likelihood_visible_drawer, pose_ee, position_drawer
            )
            new_likelihood = torch.clip(new_likelihood, 1e-9, 0.9999999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_visible_drawer"], ["likelihood_visible_drawer"]


class KinematicJointEstimator(EstimationComponent):
    """The drawer's prismatic joint [azimuth, elevation, joint value, initial point (3)]. While the drawer is not
    grasped the initial point follows the drawer position; while it is grasped the axis and the joint value do."""

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"kinematic_joint": (6,), "uncertainty_joint": (6, 6)}
    # initial guess of the axis (the true axis is azimuth = elevation = pi / 2)
    initial_azimuth = math.pi / 2 - 0.2
    initial_elevation = math.pi / 2 - 0.2

    def initialize_quantities(self):
        c = self.connections["DrawerKinematics"]
        with c.lock:
            if not c.connected_quantities_initialized["position_drawer"]:
                return False
            position_drawer = c.connected_quantities["position_drawer"]
            uncertainty_drawer = c.connected_quantities["uncertainty_drawer"]
        joint = torch.zeros(6, dtype=self.dtype, device=self.device)
        joint[0] = self.initial_azimuth
        joint[1] = self.initial_elevation
        joint[3:6] = position_drawer
        uncertainty = torch.eye(6, dtype=self.dtype, device=self.device)
        uncertainty[2, 2] = 1e-3
        uncertainty[3:6, 3:6] = uncertainty_drawer
        self.quantities["kinematic_joint"] = joint
        self.quantities["uncertainty_joint"] = uncertainty
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["DrawerKinematics"].c_func

        def forward_joint(kinematic_joint):
            return kinematic_joint  # joints do not move by themselves

        R_additive_grasped = torch.eye(3, device=self.device, dtype=self.dtype) * 1e-2
        R_additive_ungrasped = torch.eye(3, device=self.device, dtype=self.dtype) * 5e-5
        Q_diag = torch.tensor([1e-2, 1e-2, 1e1, 2e-2, 2e-2, 2e-2], device=self.device, dtype=self.dtype)

        def f_func(
            kinematic_joint, uncertainty_joint, position_drawer, uncertainty_drawer, likelihood_grasped_drawer, dt
        ):
            # Below a grasp likelihood of 0.5 the drawer position informs the initial point, above it the axis and
            # the joint value.
            grasped = gradient_preserving_clipping(likelihood_grasped_drawer, 1e-12, 1.0)
            not_grasped = gradient_preserving_clipping((0.5 - likelihood_grasped_drawer) / 0.5, 1e-10, 1.0)
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
