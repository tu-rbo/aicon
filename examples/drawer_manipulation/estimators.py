"""
Estimators for Drawer Manipulation with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union, Callable, Dict, Iterable, Tuple

import torch

from aicon_base.components import EstimationComponent
from .constants import NUM_OBSTACLE_POINTS
from aicon_real_robots.constants import NUM_BODY_POINTS
from .util import likelihood_dist_func, likelihood_func_visible, get_sine_of_angles
from aicon_base.inference.ekf import (
    predict_ekf,
    update_switching_ekf_triple_connection,
    update_switching_ekf,
    update_shifting_ekf,
    update_ekf_triple_connection,
)
from aicon_base.inference.util import gradient_preserving_clipping, make_safe_for_inversion
from aicon_base.math.se3_and_SE3 import exponential_map_se3, homogeneous_transform_inverse
from aicon_ros_python_wrapping.util_ros import wait_for_tf_frame


class DrawerPositionEstimator(EstimationComponent):
    state_dim = 3
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"position_drawer": (3,), "uncertainty_drawer": (3, 3)}

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        initial_depth: Union[float, None] = None,
        initial_uncertainty_scale: Union[float, None] = None,
        sample_init_mean: bool = False,
        meas_noise_factor: float = 0.025,
        initial_drawer_pos: Union[None, Iterable] = None,
    ):
        self.initial_depth = initial_depth
        self.initial_uncertainty_scale = initial_uncertainty_scale
        self.sample_init_mean = sample_init_mean
        self.meas_noise_factor = meas_noise_factor
        self.initial_drawer_pos = initial_drawer_pos
        super().__init__(
            name=name,
            connections=connections,
            goals=goals,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            no_differentiation=no_differentiation,
            prevent_loops_in_differentiation=prevent_loops_in_differentiation,
            max_length_differentiation_trace=max_length_differentiation_trace,
        )

        # TODO estimate drawer handle orientation

    def initialize_quantities(self):
        c_cam = self.connections["ProjectiveGeometry"]
        with c_cam.lock:
            if (
                c_cam.connected_quantities_initialized["relative_position_in_CF_drawer"]
                and c_cam.connected_quantities_initialized["pose_ee"]
            ):
                pose_ee = c_cam.connected_quantities["pose_ee"]
                relative_position = c_cam.connected_quantities["relative_position_in_CF_drawer"]
            else:
                print(
                    "At least one is missing: Drawer Measurement ",
                    not c_cam.connected_quantities_initialized["relative_position_in_CF_drawer"],
                    " or EE Pose ",
                    not c_cam.connected_quantities_initialized["pose_ee"],
                )
                return False
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
        except AssertionError:
            print("DrawerEstimator not initialized, because it could not determine EE to Cam Transform")
            return False
        H = torch.einsum("ij,jk->ik", exponential_map_se3(pose_ee), H_ee_to_cam)
        if self.initial_drawer_pos is not None:
            initial_mu = torch.tensor(self.initial_drawer_pos, dtype=self.dtype, device=self.device)
        else:
            if torch.all(torch.isnan(relative_position)):
                print("Drawer Measurement is NaN")
                return False
            if self.initial_depth is None:
                relative_position[2] = 0.8  # prior for where it should be
            else:
                relative_position[2] = self.initial_depth
            initial_mu = torch.einsum(
                "ij,j->i", H, torch.concat([relative_position, torch.ones(1, dtype=self.dtype, device=self.device)])
            )[:3]
        if self.initial_uncertainty_scale is None:
            Sigma = torch.eye(3, dtype=self.dtype, device=self.device) * 0.1
            Sigma[2, 2] = 1.2
            initial_Sigma = torch.einsum("ij,jk,kl->il", H[:3, :3], Sigma, torch.transpose(H[:3, :3], dim0=0, dim1=1))
        else:
            Sigma = torch.eye(3, dtype=self.dtype, device=self.device) * 0.2
            Sigma[2, 2] = 1.0  # 1.2
            initial_Sigma = torch.einsum(
                "ij,jk,kl->il",
                H[:3, :3],
                Sigma * self.initial_uncertainty_scale,
                torch.transpose(H[:3, :3], dim0=0, dim1=1),
            )
        print("initial mean", initial_mu, pose_ee, relative_position)
        if self.sample_init_mean:
            # torch sampling is crazy, usually gives values outside 95% confidence interval
            likelihood = 0.0
            dist = 0.0
            while likelihood < 0.6 or dist < 0.25:
                sampled_mu = torch.distributions.multivariate_normal.MultivariateNormal(
                    initial_mu, initial_Sigma
                ).sample()
                relative_pos = torch.einsum(
                    "ki,ij,j->k",
                    homogeneous_transform_inverse(H_ee_to_cam),
                    homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                    torch.cat([sampled_mu, torch.ones(1, dtype=sampled_mu.dtype, device=sampled_mu.device)]),
                )[:3]
                likelihood = likelihood_func_visible(pose_ee, sampled_mu, H_ee_to_cam)
                dist = relative_pos[2]
            initial_mu = sampled_mu
            initial_Sigma = initial_Sigma * 2.0
            print("sampled mean", initial_mu, likelihood)
        self.quantities["position_drawer"] = initial_mu
        self.quantities["uncertainty_drawer"] = initial_Sigma
        return True

    def define_estimation_function_f(self):

        def forward_drawer(mu):
            return mu  # Drawers do not naturally move by themselves

        c_cam = self.connections["ProjectiveGeometry"].c_func
        c_grasped = self.connections["GraspedDrawerKinematics"].c_func
        c_depth_cam = self.connections["DepthCameraProjection"].c_func
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=5.0).to(
                dtype=self.dtype, device=self.device
            )
        except AssertionError:
            print("Fallback on saved transform")
            H_ee_to_cam = torch.tensor(
                [
                    [2.5214e-01, -9.6767e-01, 6.4494e-03, -6.3752e-02],
                    [9.6769e-01, 2.5213e-01, -2.1942e-03, -1.6633e-02],
                    [4.9714e-04, 6.7943e-03, 9.9998e-01, 4.0590e-02],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                ],
                dtype=self.dtype,
                device=self.device,
            )

        def process_visual_measurement(
            Sigma,
            likelihood_grasped_drawer,
            likelihood_visible_drawer,
            mu,
            pose_ee,
            relative_position_in_CF_drawer,
            uncertainty_ee,
            dt,
            dtype,
            device,
        ):

            # compute uncertainty contributors
            additive_noise_visibility = 1.0 - likelihood_func_visible(pose_ee, mu, H_ee_to_cam, steepness=5.0)
            additive_noise_distance = 1.0 - likelihood_dist_func(pose_ee, mu, H_ee_to_cam)
            R_add = 5 * (
                torch.eye(3, dtype=dtype, device=device) * additive_noise_visibility
                + torch.eye(3, dtype=dtype, device=device) * additive_noise_distance
            )
            R_meas = torch.eye(3, dtype=dtype, device=device) * self.meas_noise_factor
            likelihood_existent_measurement = likelihood_visible_drawer * (1.0 - likelihood_grasped_drawer.detach())

            # attempt to integrate measurement
            if torch.any(torch.isnan(relative_position_in_CF_drawer)):
                # we are mocking a measurement to determine current simulated gradient
                mock_relative_position_in_CF_drawer = -c_cam(
                    mu, pose_ee, torch.zeros_like(relative_position_in_CF_drawer)
                ).detach()
                # always reject mocked measurement though
                mu_new, Sigma_new = update_switching_ekf_triple_connection(
                    c_cam,
                    mu,
                    Sigma,
                    pose_ee,
                    uncertainty_ee,
                    mock_relative_position_in_CF_drawer,
                    R_meas,
                    R_add * 0.01,
                    likelihood_existent_measurement,
                    outlier_rejection_threshold=0.0,
                )
            else:
                # just integrate measurement
                mu_new, Sigma_new = update_switching_ekf_triple_connection(
                    c_cam,
                    mu,
                    Sigma,
                    pose_ee,
                    uncertainty_ee,
                    relative_position_in_CF_drawer,
                    R_meas,
                    R_add,
                    likelihood_existent_measurement,
                    outlier_rejection_threshold=0.5,
                )
            # TODO figure out outlier rejection noise increase
            # TODO this seems to currently not work well for 6d
            # increase uncertainty if we received a good measurement when we should not have or vice versa
            measurement_not_integrated = torch.equal(mu, mu_new)
            if measurement_not_integrated and likelihood_existent_measurement > 0.5:
                # visual measurements while grasped can be deceiving due to the hand being similarly blue
                print("No visual Measurement integration", likelihood_existent_measurement)
                if likelihood_grasped_drawer < 0.1:
                    pass
            elif (
                (measurement_not_integrated)
                and (likelihood_existent_measurement < 0.5)
                and (not torch.all(torch.isnan(relative_position_in_CF_drawer)))
            ):
                print("Visual Measurement even though there should not be one", likelihood_existent_measurement)
                if (
                    likelihood_grasped_drawer < 0.1 and not likelihood_existent_measurement == 0.0
                ):  # exactly zero means probbaly not fully initialized
                    Sigma_new = Sigma + torch.eye(3, dtype=dtype, device=device) * 0.1 * dt * (
                        1.0 - likelihood_existent_measurement
                    )
            return Sigma_new, mu_new

        R_add_depth = 0.1 * torch.eye(1, dtype=self.dtype, device=self.device)

        def process_depth_measurement(Sigma, mu, depth_in_CF_drawer, pose_ee, uncertainty_ee):
            if torch.any(torch.isnan(depth_in_CF_drawer)) or torch.sum(depth_in_CF_drawer) == 0:
                return mu, Sigma
            return update_ekf_triple_connection(
                c_depth_cam,
                mu,
                Sigma,
                pose_ee,
                uncertainty_ee,
                depth_in_CF_drawer.unsqueeze(0),
                R_add_depth,
                R_add_depth,
                outlier_rejection_threshold=0.5,
            )

        def f_func(
            position_drawer,
            uncertainty_drawer,
            pose_ee,
            uncertainty_ee,
            relative_position_in_CF_drawer,
            likelihood_grasped_drawer,
            likelihood_visible_drawer,
            time_since_hand_change,
            depth_in_CF_drawer,
            dt,
        ):
            forward_noise = likelihood_grasped_drawer * 0.15 * dt + 0.0005 * dt
            mu_pred, Sigma_new = predict_ekf(
                forward_drawer,
                position_drawer,
                uncertainty_drawer,
                torch.eye(3, dtype=self.dtype, device=self.device) * forward_noise,
            )
            # TODO forward noise depending on graped
            Sigma_new, mu_new = process_visual_measurement(
                Sigma_new,
                likelihood_grasped_drawer,
                likelihood_visible_drawer,
                mu_pred,
                pose_ee,
                relative_position_in_CF_drawer,
                uncertainty_ee,
                dt,
                self.dtype,
                self.device,
            )
            mu_new, Sigma_new = process_depth_measurement(
                Sigma_new, mu_new, depth_in_CF_drawer, pose_ee, uncertainty_ee
            )
            mu_new = mu_pred + (mu_new - mu_pred).detach()
            mu_new, Sigma_new = update_switching_ekf(
                c_grasped,
                mu_new,
                Sigma_new,
                pose_ee,
                uncertainty_ee,
                torch.eye(3, 3, dtype=self.dtype, device=self.device) * 0.03,
                likelihood_grasped_drawer,
                outlier_rejection_threshold=0.05,
                prevent_uncertainty_state_grad=False,
            )  # 0.1 -> more than 1% outlier chance
            if time_since_hand_change[0] < 0.5 and time_since_hand_change[1] == 0 and likelihood_grasped_drawer != 0.0:
                # possibly in the process of loosing grasp
                Sigma_new = Sigma_new + torch.eye(3, dtype=self.dtype, device=self.device) * dt
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        self.differentiate_against = ("pose_ee", "likelihood_grasped_drawer", "likelihood_visible_drawer")
        return f_func, ["position_drawer", "uncertainty_drawer"], ["position_drawer", "uncertainty_drawer"]


class DistEEDrawerEstimator(EstimationComponent):
    state_dim = 2
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"distance_ee_drawer": (2,), "uncertainty_dist": (1,)}

    def initialize_quantities(self):
        c_dist = self.connections["E[dist]"]
        with c_dist.lock:
            if (
                c_dist.connected_quantities_initialized["position_drawer"]
                and c_dist.connected_quantities_initialized["pose_ee"]
            ):
                pose_ee = c_dist.connected_quantities["pose_ee"]
                position_drawer = c_dist.connected_quantities["position_drawer"]
                uncertainty_ee = c_dist.connected_quantities["uncertainty_ee"]
                uncertainty_drawer = c_dist.connected_quantities["uncertainty_drawer"]
                c_func = c_dist.c_func
            else:
                return False
        # Just take innovation fully
        self.quantities["distance_ee_drawer"] = c_func(
            torch.zeros(self.state_dim, dtype=self.dtype, device=self.device), pose_ee, position_drawer
        )
        self.quantities["uncertainty_dist"] = torch.sum(torch.trace(uncertainty_drawer)) + torch.sum(
            torch.trace(uncertainty_ee)
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["E[dist]"].c_func

        def f_func(distance_ee_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            innovation = c_func(distance_ee_drawer, pose_ee, position_drawer)
            new_dist = distance_ee_drawer + innovation
            uncertainty = torch.sum(torch.trace(uncertainty_drawer)) + torch.sum(torch.trace(uncertainty_ee))
            return (new_dist, uncertainty), (new_dist, uncertainty)

        self.differentiate_against = ("pose_ee", "position_drawer", "uncertainty_drawer")
        return f_func, ["distance_ee_drawer", "uncertainty_dist"], ["distance_ee_drawer", "uncertainty_dist"]


class GraspedEstimator(EstimationComponent):
    state_dim = 1
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_grasped_drawer": (1,),
        "time_since_hand_change": (2,),
    }

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        initially_grasped: bool = False,
    ):
        self.initially_grasped = initially_grasped
        super().__init__(
            name=name,
            connections=connections,
            goals=goals,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            no_differentiation=no_differentiation,
            prevent_loops_in_differentiation=prevent_loops_in_differentiation,
            max_length_differentiation_trace=max_length_differentiation_trace,
        )

    def initialize_quantities(self):
        c_grasp = self.connections["GraspedLikelihood"]
        with c_grasp.lock:
            if (
                c_grasp.connected_quantities_initialized["distance_ee_drawer"]
                and c_grasp.connected_quantities_initialized["hand_synergy_activation"]
            ):
                distance_ee_drawer = c_grasp.connected_quantities["distance_ee_drawer"]
                uncertainty_dist = c_grasp.connected_quantities["uncertainty_dist"]
                hand_synergy_activation = c_grasp.connected_quantities["hand_synergy_activation"]
                c_func = c_grasp.c_func
            else:
                return False
        if self.initially_grasped:
            self.quantities["likelihood_grasped_drawer"] = (
                torch.ones((self.state_dim,), dtype=self.dtype, device=self.device) * 0.99
            )
            self.quantities["time_since_hand_change"] = torch.zeros(2, dtype=self.dtype, device=self.device)
        else:
            start_likelihood = torch.zeros((1,), dtype=self.dtype, device=self.device)
            start_likelihood[0] = 0.01
            self.quantities["likelihood_grasped_drawer"] = torch.clip(
                c_func(
                    start_likelihood,
                    distance_ee_drawer,
                    uncertainty_dist,
                    hand_synergy_activation,
                    torch.zeros(3, dtype=self.dtype, device=self.device),
                    torch.zeros(2, dtype=self.dtype, device=self.device),
                ),
                0.02,
                0.98,
            )
            self.quantities["time_since_hand_change"] = torch.zeros(2, dtype=self.dtype, device=self.device)
            self.quantities["time_since_hand_change"][0] = 2.0
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(
            likelihood_grasped_drawer,
            distance_ee_drawer,
            hand_synergy_activation,
            ee_force_external,
            uncertainty_dist,
            time_since_hand_change,
            dt,
        ):
            time_since_hand_change_new = torch.clone(time_since_hand_change).detach()
            if hand_synergy_activation > 0.5:
                time_since_hand_change_new[0] = 0.0
                time_since_hand_change_new[1] = dt + time_since_hand_change[1]
            else:
                time_since_hand_change_new[0] = dt + time_since_hand_change[0]
                time_since_hand_change_new[1] = 0.0
            innovation = c_func(
                likelihood_grasped_drawer,
                distance_ee_drawer,
                uncertainty_dist,
                hand_synergy_activation,
                ee_force_external,
                time_since_hand_change_new,
            )
            if innovation < -0.05:
                new_likelihood = (
                    likelihood_grasped_drawer + 0.1 * innovation
                )  # slowly decide that it isnt grasped anymore
            else:
                new_likelihood = likelihood_grasped_drawer + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.0000000000, 0.9999999)
            return (new_likelihood,), (new_likelihood, time_since_hand_change_new)

        self.differentiate_against = ("distance_ee_drawer", "hand_synergy_activation", "uncertainty_dist")
        return f_func, ["likelihood_grasped_drawer"], ["likelihood_grasped_drawer", "time_since_hand_change"]


class VisibleEstimator(EstimationComponent):
    state_dim = 1
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_visible_drawer": (1,),
    }

    def initialize_quantities(self):
        c_vis = self.connections["VisibleLikelihood"]
        with c_vis.lock:
            if (
                c_vis.connected_quantities_initialized["pose_ee"]
                and c_vis.connected_quantities_initialized["position_drawer"]
            ):
                pose_ee = c_vis.connected_quantities["pose_ee"]
                position_drawer = c_vis.connected_quantities["position_drawer"]
                c_func = c_vis.c_func
            else:
                return False
        start_likelihood = torch.zeros((1,), dtype=self.dtype, device=self.device)
        start_likelihood[0] = 0.9
        self.quantities["likelihood_visible_drawer"] = torch.clip(
            c_func(start_likelihood, pose_ee, position_drawer), 0.02, 0.98
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["VisibleLikelihood"].c_func

        def f_func(likelihood_visible_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            # TODO depend on no vision likelihood
            innovation = c_func(likelihood_visible_drawer, pose_ee, position_drawer)
            new_likelihood = likelihood_visible_drawer + 0.5 * innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.000000001, 0.9999999)
            return (new_likelihood,), (new_likelihood,)

        self.differentiate_against = ("pose_ee",)
        return f_func, ["likelihood_visible_drawer"], ["likelihood_visible_drawer"]


class KinematicJointEstimator(EstimationComponent):
    state_dim = 1 + 1 + 1 + 3  # azimuth elevation state initial_point
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"kinematic_joint": (6,), "uncertainty_joint": (6, 6)}

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        initial_rotation_xy: Union[float, None] = None,
        initial_uncertainty_scale: Union[float, None] = None,
        sample_init_mean: bool = False,
    ):
        self.initial_rotation_xy = initial_rotation_xy
        self.initial_uncertainty_scale = initial_uncertainty_scale
        self.sample_init_mean = sample_init_mean
        super().__init__(
            name=name,
            connections=connections,
            goals=goals,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            no_differentiation=no_differentiation,
            prevent_loops_in_differentiation=prevent_loops_in_differentiation,
            max_length_differentiation_trace=max_length_differentiation_trace,
        )

    def initialize_quantities(self):
        if not self.connections["DrawerKinematics"].connected_quantities_initialized["position_drawer"]:
            return False
        self.quantities["kinematic_joint"] = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        self.quantities["kinematic_joint"][1] = -torch.pi / 2  # is always on the xy plane
        if self.initial_rotation_xy is None:
            self.quantities["kinematic_joint"][0] = -torch.pi / 2 * 0.5
            pass
        else:
            # depends on rotation of robot to drawer cabinet
            self.quantities["kinematic_joint"][0] = self.initial_rotation_xy
        self.quantities["kinematic_joint"][3:6] = self.connections["DrawerKinematics"].connected_quantities[
            "position_drawer"
        ]
        if self.initial_uncertainty_scale is None:
            self.quantities["uncertainty_joint"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device)
        else:
            self.quantities["uncertainty_joint"] = (
                torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * self.initial_uncertainty_scale
            )
        self.quantities["uncertainty_joint"][3:6, 3:6] = self.connections["DrawerKinematics"].connected_quantities[
            "uncertainty_drawer"
        ]
        if self.sample_init_mean:
            init_angles = torch.distributions.multivariate_normal.MultivariateNormal(
                self.quantities["kinematic_joint"][0:2], self.quantities["uncertainty_joint"][:2, :2]
            ).sample()
            self.quantities["kinematic_joint"][0:2] = init_angles
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["DrawerKinematics"].c_func

        def forward_joint(kinematic_joint):
            return kinematic_joint  # Joints do not naturally move by themselves

        R_additive_grasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.02
        R_additive_ungrasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.00005
        # TODO angles grow waay too quickly; rather make that smaller and increase measurement noise
        # TODO Similar for the point, maybe also dependent on grasp likelihood?
        Q_diag = torch.tensor([0.001, 0.001, 0.1, 0.02, 0.02, 0.02], device=self.device, dtype=self.dtype)

        def f_func(
            kinematic_joint, uncertainty_joint, position_drawer, uncertainty_drawer, likelihood_grasped_drawer, dt
        ):
            # starting from 0.5 grasp likelihood we do not want to integrate any starting point info anymore
            # but instead only about the joint
            unlikelihood_grasped = gradient_preserving_clipping(
                (0.2 - likelihood_grasped_drawer) / 0.2, 0.0000000001, 1.0
            )
            likelihood_grasped = gradient_preserving_clipping(likelihood_grasped_drawer, 0.000000000001, 1.0)
            shift_diagonal_matrix = torch.diag(
                torch.cat(
                    [
                        likelihood_grasped,
                        likelihood_grasped,
                        likelihood_grasped,
                        unlikelihood_grasped,
                        unlikelihood_grasped,
                        unlikelihood_grasped,
                    ]
                )
            )
            shift_diagonal_matrix = gradient_preserving_clipping(shift_diagonal_matrix, 0.0, 1.0)
            mu_new, Sigma_new = predict_ekf(
                forward_joint, kinematic_joint, uncertainty_joint, shift_diagonal_matrix * Q_diag * dt
            )
            # azi, ele, and state are highly influenced when grasped, initial point else
            R_additive = R_additive_grasped * likelihood_grasped + R_additive_ungrasped * unlikelihood_grasped
            mu_new, Sigma_new = update_shifting_ekf(
                c_func,
                mu_new,
                Sigma_new,
                position_drawer,
                uncertainty_drawer,
                R_additive,
                shift_diagonal_matrix,
                outlier_rejection_threshold=1.0,
            )
            Sigma_new = torch.cat(
                [
                    torch.cat([Sigma_new[:3, :3], torch.zeros_like(Sigma_new[:3, :3])], dim=0),
                    torch.cat([torch.zeros_like(Sigma_new[3:, 3:]), Sigma_new[3:, 3:]], dim=0),
                ],
                dim=1,
            )
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        self.differentiate_against = ("position_drawer", "uncertainty_drawer", "likelihood_grasped_drawer")
        return f_func, ["kinematic_joint", "uncertainty_joint"], ["kinematic_joint", "uncertainty_joint"]


class OcclusionAndObstaclePointsEstimator(EstimationComponent):
    state_dim = 3
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"obstacle_points": (NUM_OBSTACLE_POINTS, 3)}

    def initialize_quantities(self):
        c = self.connections["Environment2ObstacleConnection"]
        if (
            c.connected_quantities_initialized["position_drawer"]
            and c.connected_quantities_initialized["environment_point_positions"]
            and c.connected_quantities_initialized["body_points"]
            and c.connected_quantities_initialized["likelihood_grasped_drawer"]
        ):
            new_points = -c.c_func(
                self.quantities["obstacle_points"],
                c.connected_quantities["position_drawer"],
                c.connected_quantities["environment_point_positions"],
                c.connected_quantities["environment_point_tracked"],
                c.connected_quantities["body_points"],
                c.connected_quantities["likelihood_grasped_drawer"],
            )
            self.quantities["obstacle_points"] = new_points
            return True
        else:
            return False

    def define_estimation_function_f(self):
        c_func = self.connections["Environment2ObstacleConnection"].c_func

        def f_func(
            obstacle_points,
            position_drawer,
            environment_point_positions,
            environment_point_tracked,
            body_points,
            likelihood_grasped_drawer,
        ):
            new_points = obstacle_points - c_func(
                obstacle_points,
                position_drawer,
                environment_point_positions,
                environment_point_tracked,
                body_points,
                likelihood_grasped_drawer,
            )
            return (new_points,), (new_points,)

        self.no_differentiation = True
        return f_func, ["obstacle_points"], ["obstacle_points"]


class ObstacleCollisionEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_obstacle_collision": (NUM_OBSTACLE_POINTS, NUM_BODY_POINTS)
    }

    def initialize_quantities(self):
        if (not self.connections["ObstacleCollision"].connected_quantities_initialized["body_points"]) or (
            not self.connections["ObstacleCollision"].connected_quantities_initialized["obstacle_points"]
        ):
            return False
        self.quantities["likelihood_obstacle_collision"] = self.connections["ObstacleCollision"].c_func(
            self.connections["ObstacleCollision"].connected_quantities["body_points"],
            self.connections["ObstacleCollision"].connected_quantities["obstacle_points"],
            torch.zeros((NUM_OBSTACLE_POINTS, NUM_BODY_POINTS), dtype=self.dtype, device=self.device),
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["ObstacleCollision"].c_func

        def f_func(likelihood_obstacle_collision, body_points, obstacle_points):
            new_likelihood = likelihood_obstacle_collision + c_func(
                body_points, obstacle_points, likelihood_obstacle_collision
            )
            return (new_likelihood,), (new_likelihood,)

        self.differentiate_against = ("body_points",)
        return f_func, ["likelihood_obstacle_collision"], ["likelihood_obstacle_collision"]


class EnvironmentPointEstimator(EstimationComponent):
    state_dim = 3
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "environment_point_positions": (
            0,
            3,
        ),
        "environment_point_uncertainties": (0, 3, 3),
        "environment_point_tracked": (0,),
        "environment_point_time_since_last_update": (0,),
        "environment_point_tracked_time_total": (0,),
    }

    def initialize_quantities(self):
        c_cam = self.connections["ProjectiveGeometryEnvironment"]
        with c_cam.lock:
            if not c_cam.connected_quantities_initialized["relative_position_in_CF_environment_points"]:
                return False
            if not c_cam.connected_quantities_initialized["pose_ee"]:
                return False

        self.quantities["environment_point_positions"] = torch.zeros(
            (0, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.quantities["environment_point_uncertainties"] = torch.zeros(
            (0, self.state_dim, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.quantities["environment_point_time_since_last_update"] = torch.zeros(
            (0), device=self.device, dtype=self.dtype
        )
        self.quantities["environment_point_tracked"] = torch.zeros(0, device=self.device, dtype=torch.bool)
        self.quantities["environment_point_tracked_time_total"] = torch.zeros(0, device=self.device, dtype=self.dtype)
        return True

    def define_estimation_function_f(self):
        c_cam = self.connections["ProjectiveGeometryEnvironment"]
        H_ee_to_cam = c_cam.H_ee_to_cam
        Q = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0) * 0.005
        R_additive_2d = torch.eye(3, dtype=self.dtype, device=self.device) * 1.0
        R_additive_depth = torch.eye(1, dtype=self.dtype, device=self.device) * 0.05

        def simplified_connection_func_2d(matrix, point, meas):
            expected_measurement = get_sine_of_angles(torch.mv(matrix, point)[:3])
            return (meas - expected_measurement,), (meas - expected_measurement,)

        def single_inference_function_2d(matrix, point, uncertainty, meas):
            (H_this,), (innovation,) = torch.func.jacfwd(
                simplified_connection_func_2d, argnums=(1), randomness="same", has_aux=True
            )(matrix, point, meas)
            H_this = H_this[:3, :3]
            S = torch.mm(torch.mm(H_this, uncertainty), H_this.transpose(0, 1)) + R_additive_2d
            K = torch.mm(
                torch.mm(uncertainty, H_this.transpose(0, 1)), torch.inverse(make_safe_for_inversion(S, is_batch=False))
            )
            change = torch.mv(K, innovation)
            point_new = point[:3] - change
            uncertainty_new = uncertainty - torch.mm(K, torch.mm(H_this, uncertainty))
            # TODO maybe outlier, rejection, maybe grad prevention
            return point_new, uncertainty_new

        def simplified_connection_func_depth(matrix, point, meas):
            expected_measurement = torch.mv(matrix, point)[2]
            return (meas - expected_measurement,), (meas - expected_measurement,)

        def single_inference_function_depth(matrix, point, uncertainty, meas):
            (H_this,), (innovation,) = torch.func.jacfwd(
                simplified_connection_func_depth, argnums=(1), randomness="same", has_aux=True
            )(matrix, point, meas)
            innovation = innovation.unsqueeze(0)
            H_this = H_this[:3].unsqueeze(0)
            innovation_uncertainty = torch.mm(torch.mm(H_this, uncertainty), H_this.transpose(0, 1))
            mahalanobis = torch.einsum("i,ij,j->", innovation, innovation_uncertainty, innovation)
            outlier_rejection_factor = mahalanobis < 0.5
            S = innovation_uncertainty + R_additive_depth
            K = torch.mm(
                torch.mm(uncertainty, H_this.transpose(0, 1)), torch.inverse(make_safe_for_inversion(S, is_batch=False))
            )
            change = torch.mv(K, innovation)
            point_new = point[:3] - change * outlier_rejection_factor
            uncertainty_new = uncertainty - torch.mm(K, torch.mm(H_this, uncertainty)) * outlier_rejection_factor
            # TODO maybe outlier, rejection, maybe grad prevention
            return point_new, uncertainty_new

        batched_inference_2d = torch.vmap(single_inference_function_2d)
        batched_inference_depth = torch.vmap(single_inference_function_depth)
        assumed_dist = 0.8
        init_Sigma = torch.eye(3, dtype=self.dtype, device=self.device) * 0.5
        init_Sigma[2, 2] = 3.0

        def f_func(
            environment_point_positions,
            environment_point_uncertainties,
            pose_ee,
            relative_position_in_CF_environment_points,
            present_measurement_points_mask,
            environment_point_tracked,
            dt,
            environment_point_time_since_last_update,
            environment_point_tracked_time_total,
            depth_in_CF_environment_points,
            body_points,
            all_depth_values,
        ):
            uncertainties = environment_point_uncertainties + Q * dt
            environment_point_time_since_last_update = environment_point_time_since_last_update + dt
            if torch.sum(present_measurement_points_mask) != relative_position_in_CF_environment_points.shape[0]:
                raise IndexError("Received number of measurement points and measurement point matches differ")

            present_measurement_points_mask = present_measurement_points_mask.bool()
            n_add_points = present_measurement_points_mask.shape[0] - environment_point_positions.shape[0]
            present_depth_meas = depth_in_CF_environment_points > 0
            if n_add_points > 0:
                H = torch.einsum("ij,jk->ik", exponential_map_se3(pose_ee), H_ee_to_cam)
                relative_position_naive = torch.ones(
                    present_measurement_points_mask.shape[0], 3, device=self.device, dtype=self.dtype
                )
                relative_position_naive[present_measurement_points_mask] = relative_position_in_CF_environment_points
                relative_position_naive[:, 2] = assumed_dist
                relative_position_naive[present_measurement_points_mask][present_depth_meas, 2] = (
                    depth_in_CF_environment_points[present_depth_meas]
                )
                relative_position = relative_position_naive[-n_add_points:]
                initial_points = torch.einsum(
                    "ij,bj->bi",
                    H,
                    torch.concat(
                        [
                            relative_position,
                            torch.ones(relative_position.shape[0], 1, dtype=self.dtype, device=self.device),
                        ],
                        dim=1,
                    ),
                )[:, :3]
                points = torch.cat([environment_point_positions, initial_points])
                initial_Sigma = torch.einsum(
                    "ij,jk,kl->il", H[:3, :3], init_Sigma, torch.transpose(H[:3, :3], dim0=0, dim1=1)
                )
                uncertainties = torch.cat([uncertainties, initial_Sigma.unsqueeze(0).expand(n_add_points, -1, -1)])
                environment_point_tracked = torch.cat(
                    [environment_point_tracked, present_measurement_points_mask[-n_add_points:]]
                )
                environment_point_time_since_last_update = torch.cat(
                    [
                        environment_point_time_since_last_update,
                        present_measurement_points_mask[-n_add_points:].to(self.dtype) * 0.0,
                    ]
                )
                environment_point_tracked_time_total = torch.cat(
                    [
                        environment_point_tracked_time_total,
                        present_measurement_points_mask[-n_add_points:].to(self.dtype) * 0.0,
                    ]
                )
            else:
                points = environment_point_positions
            if all_depth_values.shape[0] < points.shape[0]:
                all_depth_values = torch.cat(
                    [
                        all_depth_values,
                        torch.zeros(points.shape[0] - all_depth_values.shape[0], dtype=self.dtype, device=self.device),
                    ],
                    dim=0,
                )
            elif all_depth_values.shape[0] > points.shape[0]:
                all_depth_values = all_depth_values[: points.shape[0] - 1]
            print("depth shape", all_depth_values.shape, points.shape[0])
            to_track_mask = torch.logical_and(present_measurement_points_mask, environment_point_tracked)
            if torch.sum(to_track_mask) > 0:
                points_for_update = points[to_track_mask]
                uncertainties_for_update = uncertainties[to_track_mask] + Q * dt * 1.5
                environment_point_time_since_last_update[to_track_mask] = 0.0
                environment_point_tracked_time_total[to_track_mask] = (
                    environment_point_tracked_time_total[to_track_mask] + dt
                )
                meas = relative_position_in_CF_environment_points[
                    environment_point_tracked[present_measurement_points_mask]
                ]
                n_points = points_for_update.shape[0]
                present_depth_meas_short = present_depth_meas[:n_points]
                depth_in_CF_environment_points_short = depth_in_CF_environment_points[:n_points]
                matrix = torch.einsum(
                    "ki,ij->kj",
                    homogeneous_transform_inverse(H_ee_to_cam),
                    homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                )
                matrix_batched = matrix.unsqueeze(0).expand(n_points, -1, -1)
                points_h = torch.cat(
                    [
                        points_for_update,
                        torch.ones(
                            (n_points, 1),
                            dtype=environment_point_positions.dtype,
                            device=environment_point_positions.device,
                        ),
                    ],
                    dim=1,
                )
                new_points, new_uncertainties = batched_inference_2d(
                    matrix_batched, points_h, uncertainties_for_update, meas
                )
                points_h = torch.cat(
                    [
                        new_points,
                        torch.ones(
                            (n_points, 1),
                            dtype=environment_point_positions.dtype,
                            device=environment_point_positions.device,
                        ),
                    ],
                    dim=1,
                )
                new_points[present_depth_meas_short], new_uncertainties[present_depth_meas_short] = (
                    batched_inference_depth(
                        matrix_batched[present_depth_meas_short],
                        points_h[present_depth_meas_short],
                        new_uncertainties[present_depth_meas_short],
                        depth_in_CF_environment_points_short[present_depth_meas_short],
                    )
                )
                points[to_track_mask] = new_points
                uncertainties[to_track_mask] = new_uncertainties
                points_h = torch.cat(
                    [
                        points,
                        torch.ones(
                            (points.shape[0], 1),
                            dtype=environment_point_positions.dtype,
                            device=environment_point_positions.device,
                        ),
                    ],
                    dim=1,
                )
                point_depths = torch.einsum("ij,bj->bj", matrix, points_h)[:, 2]
                lost = torch.logical_and(point_depths - 0.5 > all_depth_values, all_depth_values > 0)
                too_far = torch.logical_and(
                    torch.norm(points, dim=1) > 1.2, environment_point_time_since_last_update > 3.0
                )
                too_little_observation = torch.logical_and(
                    environment_point_tracked_time_total < 1.0, environment_point_time_since_last_update > 5.0
                )
                body_dists = torch.min(torch.norm(body_points.unsqueeze(0) - points.unsqueeze(1), dim=2), dim=1)[0]
                not_too_close_to_body = body_dists > 0.125
                environment_point_tracked = torch.logical_and(
                    torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(environment_point_tracked, torch.logical_not(too_far)),
                            torch.logical_not(too_little_observation),
                        ),
                        not_too_close_to_body,
                    ),
                    torch.logical_not(lost),
                )
            return (points, uncertainties), (
                points,
                uncertainties,
                environment_point_tracked,
                environment_point_time_since_last_update,
                environment_point_tracked_time_total,
            )

        self.no_differentiation = True
        return (
            f_func,
            ["environment_point_positions", "environment_point_uncertainties"],
            [
                "environment_point_positions",
                "environment_point_uncertainties",
                "environment_point_tracked",
                "environment_point_time_since_last_update",
                "environment_point_tracked_time_total",
            ],
        )
