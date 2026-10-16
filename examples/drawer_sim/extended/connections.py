"""
Active Interconnections for the Extended Drawer Example

Compared to the tutorial, the end-effector pose is 6D, the handle is seen by a wrist camera, and the grasp
likelihood is inferred from an explicit end-effector-to-handle distance estimate and its uncertainty.

Authors: James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection
from ..tutorial.connections import DrawerDirectMeasurementConnection, KinematicJointConnection
from ..util import bearing, ee_to_camera, likelihood_func_visible, point_in_camera_frame, pose_vec_to_homogeneous

# point in the end-effector frame whose distance to the handle decides whether a grasp is possible [m]
PREGRASP_POINT = [0.0, 0.0, -0.005]


def build_connections():
    return {
        "GraspedDrawerKinematics": lambda device, dtype, mockbuild: EEDrawerGraspedConnection(
            "GraspedDrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "E[dist]": lambda device, dtype, mockbuild: DistDrawerEEConnection(
            "E[dist]", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "GraspedLikelihood": lambda device, dtype, mockbuild: DistGraspHandConnection(
            "GraspedLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ForwardKinematics": lambda device, dtype, mockbuild: EEVeloConnection(
            "ForwardKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DirectMeasurement": lambda device, dtype, mockbuild: EEProprioConnection(
            "DirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DrawerKinematics": lambda device, dtype, mockbuild: KinematicJointConnection(
            "DrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DrawerDirectMeasurement": lambda device, dtype, mockbuild: DrawerDirectMeasurementConnection(
            "DrawerDirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ProjectiveGeometry": lambda device, dtype, mockbuild: DrawerCameraEEConnection(
            "ProjectiveGeometry", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "VisibleLikelihood": lambda device, dtype, mockbuild: VisibleEEDrawerConnection(
            "VisibleLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }


class EEDrawerGraspedConnection(ActiveInterconnection):
    """A grasped drawer moves with the end-effector: the handle is at the end-effector's position."""

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
            # Moving the drawer requires a grasp; moving the end-effector freely requires none.
            required_signs_dict={
                "position_drawer": {"likelihood_grasped_drawer": 1},
                "uncertainty_drawer": {"likelihood_grasped_drawer": 1},
                "pose_ee": {"likelihood_grasped_drawer": -1},
            },
        )

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, pose_ee):
            return pose_ee[:3] - position_drawer

        return connection_func


class DistDrawerEEConnection(ActiveInterconnection):
    """Distance between the pre-grasp point of the end-effector and the handle. The second entry is reserved for
    the orientation error to a good grasp and is always zero here."""

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
                "distance_ee_drawer": (2,),
                "uncertainty_dist": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_pregrasp = torch.eye(4, dtype=self.dtype, device=self.device)
        H_pregrasp[:3, 3] = torch.tensor(PREGRASP_POINT, dtype=self.dtype, device=self.device)

        def connection_func(distance_ee_drawer, pose_ee, position_drawer):
            pregrasp_point = torch.einsum("ij,jk->ik", pose_vec_to_homogeneous(pose_ee), H_pregrasp)[:3, 3]
            dist = torch.norm(pregrasp_point - position_drawer)
            orientation_error = torch.zeros((), dtype=self.dtype, device=self.device)
            return torch.cat([dist.unsqueeze(0), orientation_error.unsqueeze(0)]) - distance_ee_drawer

        return connection_func


class DistGraspHandConnection(ActiveInterconnection):
    """Grasp likelihood from the estimated distance to the handle and its uncertainty, the force at the
    end-effector and the gripper.

    A grasp is only plausible if the handle position is certain enough. Force and gripper count once the
    end-effector is close and the gripper has been open for a while, or while the gripper is closed; the force
    threshold rises from 0 to 6 N during the first second after closing.
    """

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
                "gripper_activation": (1,),
                "ee_force_mag_meas": (1,),
                "time_since_hand_change": (2,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        dist_decay = 4.0
        close_dist = 0.03  # [m]
        max_uncertainty_dist = 0.4
        open_time_before_grasp = 1.0  # [s] the gripper has to be open this long before a grasp attempt counts
        gripper_closed = 0.5
        low_likelihood = 0.1
        uncertainty_bias = 0.2
        force_threshold_rate, force_threshold_max = 6.0, 6.0  # [N/s], [N]

        def connection_func(
            likelihood_grasped_drawer,
            distance_ee_drawer,
            uncertainty_dist,
            gripper_activation,
            ee_force_mag_meas,
            time_since_hand_change,
        ):
            likelihood_given_uncertainty = torch.clip(torch.exp(-(uncertainty_dist - uncertainty_bias) * 5.0), max=1.0)
            innovation_from_uncertainty = likelihood_given_uncertainty - likelihood_grasped_drawer

            dist_relevance = (1 - torch.sigmoid((distance_ee_drawer[1] - 0.5) * 10.0)) * torch.clip(
                torch.exp(-(uncertainty_dist - uncertainty_bias) * 20.0), max=1.0
            )
            expected_dist = torch.sum(distance_ee_drawer) + uncertainty_dist
            likelihood_given_dist = torch.exp(-expected_dist * dist_decay) * dist_relevance
            innovation_from_dist = likelihood_given_dist - likelihood_given_uncertainty.detach()

            force_magnitude = torch.norm(ee_force_mag_meas)
            if (
                distance_ee_drawer[0] < close_dist
                and uncertainty_dist < max_uncertainty_dist
                and time_since_hand_change[0] > open_time_before_grasp
            ) or (gripper_activation > gripper_closed):
                force_threshold = torch.minimum(
                    time_since_hand_change[1] * force_threshold_rate,
                    torch.ones_like(time_since_hand_change[1]) * force_threshold_max,
                )
                likelihood_from_hand_and_force = (
                    torch.clip(1 - torch.exp(-(force_magnitude - force_threshold)), 0, 1) * gripper_activation
                )
                # Closing on nothing: the gradient on the gripper points towards opening it, which a grasp needs.
                if (
                    (likelihood_grasped_drawer < low_likelihood)
                    and (likelihood_from_hand_and_force < low_likelihood)
                    and (gripper_activation > gripper_closed)
                ):
                    likelihood_from_hand_and_force = (
                        likelihood_from_hand_and_force.detach() - gripper_activation + gripper_activation.detach()
                    )
            else:
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_uncertainty) * 1e-8

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()
            return innovation_from_uncertainty + innovation_from_dist + innovation_from_hand_and_force

        return connection_func


class VisibleEEDrawerConnection(ActiveInterconnection):
    """Likelihood that the handle is in the wrist camera's field of view."""

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
        H_ee_to_cam = ee_to_camera(self.dtype, self.device)

        def connection_func(likelihood_visible_drawer, pose_ee, position_drawer):
            return likelihood_func_visible(pose_ee, position_drawer, H_ee_to_cam) - likelihood_visible_drawer

        return connection_func


class DrawerCameraEEConnection(ActiveInterconnection):
    """Projective geometry of the wrist camera: the measured bearing to the handle equals the bearing of the
    estimated handle position from the estimated camera pose. A bearing has no depth, so the handle's position is
    only determined by bearings from different viewpoints (triangulation)."""

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
                "relative_position_in_CF_drawer": (3,),
                "likelihood_visible_drawer": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee_to_cam = ee_to_camera(self.dtype, self.device)

        def connection_func(position_drawer, pose_ee, relative_position_in_CF_drawer):
            return relative_position_in_CF_drawer - bearing(
                point_in_camera_frame(pose_ee, position_drawer, H_ee_to_cam)
            )

        return connection_func


class EEVeloConnection(ActiveInterconnection):
    """The commanded translational velocity moves the end-effector; its orientation is kept."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "action_velo_ee": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee):
            return torch.cat([pose_ee[:3] + action_velo_ee, pose_ee[3:]])

        return connection_func


class EEProprioConnection(ActiveInterconnection):
    """The measured end-effector pose equals the estimated one."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "ee_pos_meas": (6,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, ee_pos_meas):
            return ee_pos_meas - pose_ee

        return connection_func
