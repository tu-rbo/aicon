"""
Connections for Drawer Manipulation with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_real_robots.connections.connections import (
    EEVeloConnection,
    EEProprioConnection,
    JointsVeloConnection,
    PandaJoint2EEConnection,
    PandaWithHandJoint2BodyPointsConnection,
    BodyPoints2SelfCollisionConnection,
    JointsProprioConnection,
    EEFTWithHandConnection,
)
from aicon_real_robots.constants import NUM_BODY_POINTS
from aicon_base.connections import ActiveInterconnection
from .constants import NUM_OBSTACLE_POINTS
from .util import get_sine_of_angles, likelihood_func_visible
from aicon_base.inference.util import gradient_preserving_clipping
from aicon_base.math.se3_and_SE3 import exponential_map_se3, homogeneous_transform_inverse
from aicon_ros_python_wrapping.util_ros import wait_for_tf_frame


def build_connections():
    connections = {
        "GraspedDrawerKinematics": lambda device, dtype, mockbuild: EEDrawerGraspedConnection(
            "GraspedDrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "E[dist]": lambda device, dtype, mockbuild: DistDrawerEEConnection(
            "E[dist]", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "GraspedLikelihood": lambda device, dtype, mockbuild: DistGraspHandConnection(
            "GraspedLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ProjectiveGeometry": lambda device, dtype, mockbuild: CameraAndPointsConnection(
            "ProjectiveGeometry", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "RBForwardKinematics": lambda device, dtype, mockbuild: EEVeloConnection(
            "RBForwardKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DirectMeasurementEE": lambda device, dtype, mockbuild: EEProprioConnection(
            "DirectMeasurementEE", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "VisibleLikelihood": lambda device, dtype, mockbuild: VisibleEEDrawerConnection(
            "VisibleLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "OwnWeightCompensation": lambda device, dtype, mockbuild: EEFTWithHandConnection(
            "OwnWeightCompensation", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DrawerKinematics": lambda device, dtype, mockbuild: KinematicJointConnection(
            "DrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "LinearVelocityJoints": lambda device, dtype, mockbuild: JointsVeloConnection(
            "LinearVelocityJoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DirectMeasurementJoints": lambda device, dtype, mockbuild: JointsProprioConnection(
            "DirectMeasurementJoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "RobotForwardKinematics": lambda device, dtype, mockbuild: PandaJoint2EEConnection(
            "RobotForwardKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "PandaJoint2BodyPoints": lambda device, dtype, mockbuild: PandaWithHandJoint2BodyPointsConnection(
            "PandaJoint2BodyPoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "BodyPoints2SelfCollision": lambda device, dtype, mockbuild: BodyPoints2SelfCollisionConnection(
            "BodyPoints2SelfCollision", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ObstacleCollision": lambda device, dtype, mockbuild: ObstacleCollisionConnection(
            "ObstacleCollision", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ProjectiveGeometryEnvironment": lambda device, dtype, mockbuild: CameraAndPointsConnection(
            "ProjectiveGeometryEnvironment", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "Environment2ObstacleConnection": lambda device, dtype, mockbuild: Environment2ObstacleConnection(
            "Environment2ObstacleConnection", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DepthCameraProjection": lambda device, dtype, mockbuild: DepthCameraConnection(
            "DepthCameraProjection", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
    return connections


class EEDrawerGraspedConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "likelihood_grasped_drawer": (1,),
                "time_since_hand_change": (2,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            # for on/off switching connection we can require gradient signs
            # if we want to move the position of the drawer more we want it to be grasped
            # if we want to change ee more, we do not want it to be grasped
            required_signs_dict={
                "position_drawer": {"likelihood_grasped_drawer": 1},
                "uncertainty_drawer": {"likelihood_grasped_drawer": 1},
                "pose_ee": {"likelihood_grasped_drawer": -1},
            },
        )

    def define_implicit_connection_function(self):
        try:
            H_grasping_offset = wait_for_tf_frame("panda_link8", "middle_base_link0", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("no hand fallback", H_grasping_offset)
        except AssertionError:
            print("Fallback on saved transform for hand")
            H_grasping_offset = torch.tensor(
                [
                    [
                        2.74110045007598390754e-02,
                        3.82684273295470278597e-01,
                        9.23472459689284597140e-01,
                        9.77863788666019223550e-02,
                    ],
                    [
                        1.22155932520496002258e-02,
                        -9.23878840994039940426e-01,
                        3.82490086727761857688e-01,
                        2.38479587074937604552e-02,
                    ],
                    [
                        9.99549606629786091538e-01,
                        7.96326458243235002499e-04,
                        -2.99991624951785329500e-02,
                        7.91443584734091709620e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )

        def connection_func(position_drawer, pose_ee):
            H_ee = exponential_map_se3(pose_ee)
            grasping_point = torch.einsum("ij,jk->ik", H_ee, H_grasping_offset)[:3, 3]
            relative_vector = grasping_point - position_drawer
            return relative_vector

        return connection_func


class DistDrawerEEConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.H_grasping_offset = None
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "distance_ee_drawer": (2,),
                "uncertainty_dist": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):

        try:
            H_grasping_offset = wait_for_tf_frame("panda_link8", "middle_base_link0", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("no hand fallback", H_grasping_offset)
        except AssertionError:
            print("Fallback on saved transform for hand")
            H_grasping_offset = torch.tensor(
                [
                    [
                        2.74110045007598390754e-02,
                        3.82684273295470278597e-01,
                        9.23472459689284597140e-01,
                        9.77863788666019223550e-02,
                    ],
                    [
                        1.22155932520496002258e-02,
                        -9.23878840994039940426e-01,
                        3.82490086727761857688e-01,
                        2.38479587074937604552e-02,
                    ],
                    [
                        9.99549606629786091538e-01,
                        7.96326458243235002499e-04,
                        -2.99991624951785329500e-02,
                        7.91443584734091709620e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        H_grasping_offset[2, 3] += 0.02

        def connection_func(distance_ee_drawer, pose_ee, position_drawer):
            H_ee = exponential_map_se3(pose_ee)
            grasping_point = torch.einsum("ij,jk->ik", H_ee, H_grasping_offset)[:3, 3]
            relative_vector = grasping_point - position_drawer
            dist_mean = torch.norm(relative_vector)
            # good grasp oritation means z axis of ee points perfectly down
            xy_magnitude = torch.norm(H_ee[:2, 2])
            angle_to_good_grasp_pose = torch.arcsin(xy_magnitude)
            dist_combined = torch.cat([dist_mean.unsqueeze(0), angle_to_good_grasp_pose.unsqueeze(0)])
            return dist_combined - distance_ee_drawer

        return connection_func


class DistGraspHandConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "distance_ee_drawer": (2,),
                "uncertainty_dist": (1,),
                "likelihood_grasped_drawer": (1,),
                "hand_synergy_activation": (1,),
                "ee_force_external": (3,),
                "time_since_hand_change": (2,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):

        def connection_func(
            likelihood_grasped_drawer,
            distance_ee_drawer,
            uncertainty_dist,
            hand_synergy_activation,
            ee_force_external,
            time_since_hand_change,
        ):
            likelihood_given_uncertainty = torch.clip(torch.exp(-(uncertainty_dist - 0.2) * 5), max=1.0)
            innovation_from_uncertainty = likelihood_given_uncertainty - likelihood_grasped_drawer

            dist_relevance = torch.clip(
                torch.exp(-(uncertainty_dist - 0.2) * 20), max=1.0
            )  # * gradient_preserving_clipping(1 - torch.pow(distance_ee_drawer[1]  * 3, 2), 0.0, 1.0)
            expected_dist = gradient_preserving_clipping(distance_ee_drawer[0], 0, 0.35) + gradient_preserving_clipping(
                torch.pow(distance_ee_drawer[1] * 0.5, 2), 0, 1
            )  # + uncertainty_dist # TODO angle contrib?
            likelihood_given_dist = torch.exp(-expected_dist * 10) * dist_relevance
            innovation_from_dist = likelihood_given_dist - likelihood_given_uncertainty.detach()

            # TODO figure out right force scaling for the likelihood
            # TODO then bring back negative hand gradient then
            force_magnitude = torch.norm(ee_force_external)
            # hand and force measurements are only relevant if we are close (otherwise from other source...)
            if (distance_ee_drawer[0] < 0.04 and uncertainty_dist < 0.25 and time_since_hand_change[0] > 1.0) or (
                hand_synergy_activation > 0.5
            ):
                # under 2N is just FT noise
                FT_tresh = torch.minimum(
                    time_since_hand_change[1] * 0.75, torch.ones_like(time_since_hand_change[1]) * 2.25
                )
                likelihood_from_hand_and_force = (
                    torch.clip(1 - torch.exp(-(force_magnitude - FT_tresh)), 0, 1) * hand_synergy_activation
                )
                # but we need an open hand to increase this likelihood
                # so we generate a negative gradient
                if (
                    (likelihood_grasped_drawer < 0.1)
                    and (likelihood_from_hand_and_force < 0.1)
                    and (hand_synergy_activation > 0.5)
                ):
                    likelihood_from_hand_and_force = (
                        likelihood_from_hand_and_force.detach()
                        - hand_synergy_activation
                        + hand_synergy_activation.detach()
                    )
            else:
                # essentially no likelihood
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_uncertainty) * 0.0000000

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()

            return innovation_from_uncertainty + innovation_from_dist + innovation_from_hand_and_force

        return connection_func


class VisibleEEDrawerConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "likelihood_visible_drawer": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("No fallback:", H_ee_to_cam)
        except AssertionError:
            print("Fallback on saved transform for cam")
            H_ee_to_cam = torch.tensor(
                [
                    [
                        7.02207423630366811551e-01,
                        -7.11958278188459137148e-01,
                        4.48824211993193410153e-03,
                        -6.35840416956092446821e-02,
                    ],
                    [
                        7.11972251600688443851e-01,
                        7.02188897710080417447e-01,
                        -5.12492764322124497167e-03,
                        1.74961830646664512345e-02,
                    ],
                    [
                        4.97140873857276118647e-04,
                        6.79426608449542474011e-03,
                        9.99976795130429363390e-01,
                        4.05902968729656976610e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )

        def connection_func(likelihood_visible_drawer, pose_ee, position_drawer):
            likelihood = likelihood_func_visible(pose_ee, position_drawer, H_ee_to_cam)
            return likelihood - likelihood_visible_drawer

        return connection_func


class DrawerCameraEEConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.H_ee_to_cam = None
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "relative_position_in_CF_drawer": (3,),
                "likelihood_visible_drawer": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("No fallback:", H_ee_to_cam)
        except AssertionError:
            print("Fallback on saved transform for cam")
            H_ee_to_cam = torch.tensor(
                [
                    [
                        7.02207423630366811551e-01,
                        -7.11958278188459137148e-01,
                        4.48824211993193410153e-03,
                        -6.35840416956092446821e-02,
                    ],
                    [
                        7.11972251600688443851e-01,
                        7.02188897710080417447e-01,
                        -5.12492764322124497167e-03,
                        1.74961830646664512345e-02,
                    ],
                    [
                        4.97140873857276118647e-04,
                        6.79426608449542474011e-03,
                        9.99976795130429363390e-01,
                        4.05902968729656976610e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)

        def connection_func(position_drawer, pose_ee, relative_position_in_CF_drawer):
            relative_pos = torch.einsum(
                "ki,ij,j->k",
                homogeneous_transform_inverse(H_ee_to_cam),
                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]),
            )[:3]
            only_angles_sin_pred = get_sine_of_angles(relative_pos)
            # angles should be the same, dist of the measured point is always set for unit length because unknown (RGB)
            return relative_position_in_CF_drawer - only_angles_sin_pred

        return connection_func


class ObstacleCollisionConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "obstacle_points": (NUM_OBSTACLE_POINTS, 3),
                "likelihood_obstacle_collision": (NUM_OBSTACLE_POINTS, NUM_BODY_POINTS),
                "body_points": (NUM_BODY_POINTS, 3),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(body_points, obstacle_points, likelihood_obstacle_collision):
            # because of panda kinematics we only need to care about
            # parts above the elbow colliding with parts below it
            dists = torch.norm(body_points.unsqueeze(0) - obstacle_points.unsqueeze(1), dim=2)
            print("Smallest dist", torch.min(dists))
            likelihoods = torch.erf((0.08 - dists) * 5) / 2.0 + 0.5
            return likelihoods - likelihood_obstacle_collision

        return connection_func


class KinematicJointConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "kinematic_joint": (6,),
                "uncertainty_joint": (
                    6,
                    6,
                ),
                "position_drawer": (3,),
                "uncertainty_drawer": (
                    3,
                    3,
                ),
                "likelihood_grasped_drawer": (1,),
                "time_since_hand_change": (2,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(kinematic_joint, position_drawer):
            azimuth, elevation, state, initial_point = (
                kinematic_joint[0:1],
                kinematic_joint[1:2],
                kinematic_joint[2:3],
                kinematic_joint[3:],
            )
            orientation_vector = torch.cat(
                [
                    torch.cos(azimuth) * torch.sin(elevation),
                    torch.sin(azimuth) * torch.sin(elevation),
                    torch.cos(elevation),
                ],
                dim=0,
            )
            translation = orientation_vector * state
            predicted_point = initial_point + translation
            return predicted_point - position_drawer

        return connection_func


class CameraAndPointsConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.H_ee_to_cam = None
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "relative_position_in_CF_drawer": (3,),
                "depth_in_CF_drawer": (1,),
                "likelihood_visible_drawer": (1,),
                "coarse_bounding_box_drawer": (4,),
                "environment_point_positions": (0, 3),
                "environment_point_uncertainties": (0, 3, 3),
                "relative_position_in_CF_environment_points": (0, 3),
                "present_measurement_points_mask": (0,),
                "environment_point_tracked": (0,),
                "depth_in_CF_environment_points": (0,),
                "all_depth_values": (0,),
                "body_points": (NUM_BODY_POINTS, 3),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("No fallback:", H_ee_to_cam)
        except AssertionError:
            print("Fallback on saved transform for cam")
            H_ee_to_cam = torch.tensor(
                [
                    [
                        7.02207423630366811551e-01,
                        -7.11958278188459137148e-01,
                        4.48824211993193410153e-03,
                        -6.35840416956092446821e-02,
                    ],
                    [
                        7.11972251600688443851e-01,
                        7.02188897710080417447e-01,
                        -5.12492764322124497167e-03,
                        1.74961830646664512345e-02,
                    ],
                    [
                        4.97140873857276118647e-04,
                        6.79426608449542474011e-03,
                        9.99976795130429363390e-01,
                        4.05902968729656976610e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)

        def connection_func(position_drawer, pose_ee, relative_position_in_CF_drawer):
            relative_pos = torch.einsum(
                "ki,ij,j->k",
                homogeneous_transform_inverse(H_ee_to_cam),
                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]),
            )[:3]
            only_angles_sin_pred = get_sine_of_angles(relative_pos)
            # angles should be the same, dist of the measured point is always set for unit length because unknown (RGB)
            return relative_position_in_CF_drawer - only_angles_sin_pred

        return connection_func


class DepthCameraConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.H_ee_to_cam = None
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "depth_in_CF_drawer": (1,),
                "likelihood_visible_drawer": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera1_color_optical_frame", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            print("No fallback:", H_ee_to_cam)
        except AssertionError:
            print("Fallback on saved transform for cam")
            H_ee_to_cam = torch.tensor(
                [
                    [
                        7.02207423630366811551e-01,
                        -7.11958278188459137148e-01,
                        4.48824211993193410153e-03,
                        -6.35840416956092446821e-02,
                    ],
                    [
                        7.11972251600688443851e-01,
                        7.02188897710080417447e-01,
                        -5.12492764322124497167e-03,
                        1.74961830646664512345e-02,
                    ],
                    [
                        4.97140873857276118647e-04,
                        6.79426608449542474011e-03,
                        9.99976795130429363390e-01,
                        4.05902968729656976610e-02,
                    ],
                    [
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        0.00000000000000000000e00,
                        1.00000000000000000000e00,
                    ],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)

        def connection_func(position_drawer, pose_ee, depth_in_CF_drawer):
            depth_expected = torch.einsum(
                "ki,ij,j->k",
                homogeneous_transform_inverse(H_ee_to_cam),
                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]),
            )[2:3]
            return depth_in_CF_drawer - depth_expected

        return connection_func


class Environment2ObstacleConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "position_drawer": (3,),
                "obstacle_points": (NUM_OBSTACLE_POINTS, 3),
                "environment_point_positions": (0, 3),
                "environment_point_tracked": (0,),
                "body_points": (NUM_BODY_POINTS, 3),
                "likelihood_grasped_drawer": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):

        def connection_func(
            obstacle_points,
            position_drawer,
            environment_point_positions,
            environment_point_tracked,
            body_points,
            likelihood_grasped_drawer,
        ):
            if environment_point_positions.shape[0] > environment_point_tracked.shape[0]:
                environment_point_tracked = torch.cat(
                    [
                        environment_point_tracked,
                        torch.ones(
                            environment_point_positions.shape[0] - environment_point_tracked.shape[0],
                            device=self.device,
                            dtype=bool,
                        ),
                    ]
                )
            elif environment_point_positions.shape[0] < environment_point_tracked.shape[0]:
                environment_point_tracked = environment_point_tracked[: environment_point_positions.shape[0]]
            relevant_points = environment_point_positions[environment_point_tracked.bool()]
            if likelihood_grasped_drawer < 0.5:
                # leverage contact close to object
                dist_to_interaction_point = torch.norm(position_drawer.unsqueeze(0) - relevant_points, dim=1)
                relevant_points = relevant_points[dist_to_interaction_point > 0.2]
            dists = torch.norm(
                body_points[6:].unsqueeze(0) - relevant_points.unsqueeze(1), dim=2
            )  # first 4 are irrelevant because almost static
            min_dists = torch.min(dists, dim=1)[0]
            idxs = torch.argsort(min_dists)
            current_obstacle_points = relevant_points[idxs][:NUM_OBSTACLE_POINTS]
            if current_obstacle_points.shape[0] < NUM_OBSTACLE_POINTS:
                current_obstacle_points = torch.cat(
                    [
                        current_obstacle_points,
                        torch.zeros(
                            (NUM_OBSTACLE_POINTS - current_obstacle_points.shape[0], 3),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    ]
                )
            return obstacle_points - current_obstacle_points

        return connection_func
