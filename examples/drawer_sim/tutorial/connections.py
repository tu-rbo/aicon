"""
Active Interconnections for the Drawer Tutorial

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection


def build_connections():
    return {
        "GraspedDrawerKinematics": lambda device, dtype, mockbuild: EEDrawerGraspedConnection(
            "GraspedDrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild
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
    }


class EEDrawerGraspedConnection(ActiveInterconnection):
    """A grasped drawer moves with the end-effector: their positions coincide."""

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
                "pose_ee": (3,),
                "uncertainty_ee": (3, 3),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "likelihood_grasped_drawer": (1,),
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
            return position_drawer - pose_ee

        return connection_func


class DistGraspHandConnection(ActiveInterconnection):
    """Grasp likelihood from the distance between end-effector and drawer, the force at the end-effector and the
    gripper. The force only counts close to the drawer, and an open gripper is needed for a grasp to start."""

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
                "pose_ee": (3,),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "uncertainty_ee": (3, 3),
                "likelihood_grasped_drawer": (1,),
                "gripper_activation": (1,),
                "ee_force_mag_meas": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            expected_dist = torch.norm(position_drawer - pose_ee)
            likelihood_given_dist = torch.exp(-expected_dist * 4)
            innovation_from_dist = likelihood_given_dist - likelihood_grasped_drawer

            if expected_dist < 0.05 and ee_force_mag_meas > 10:
                # forces below 5 N are sensor noise
                likelihood_from_hand_and_force = (
                    torch.clip(1 - torch.exp(-(ee_force_mag_meas - 5)), 0, 1) * gripper_activation
                )
                # Closing on nothing: the gradient on the gripper points towards opening it, which a grasp needs.
                if (
                    (likelihood_grasped_drawer < 0.1)
                    and (likelihood_from_hand_and_force < 0.1)
                    and (gripper_activation > 0.5)
                ):
                    likelihood_from_hand_and_force = (
                        likelihood_from_hand_and_force.detach() - gripper_activation + gripper_activation.detach()
                    )
            else:
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_dist) * 0.00000001

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()
            return innovation_from_dist + innovation_from_hand_and_force

        return connection_func


class EEVeloConnection(ActiveInterconnection):
    """The commanded velocity moves the end-effector (one tick of the action's time step)."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (3,), "uncertainty_ee": (3, 3), "action_velo_ee": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee):
            return pose_ee + action_velo_ee

        return connection_func


class EEProprioConnection(ActiveInterconnection):
    """The measured end-effector position equals the estimated one."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (3,), "uncertainty_ee": (3, 3), "ee_pos_meas": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, ee_pos_meas):
            return ee_pos_meas - pose_ee

        return connection_func


class DrawerDirectMeasurementConnection(ActiveInterconnection):
    """The measured drawer handle position equals the estimated one."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"position_drawer": (3,), "uncertainty_drawer": (3, 3), "drawer_pos_meas": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, drawer_pos_meas):
            return drawer_pos_meas - position_drawer

        return connection_func


class KinematicJointConnection(ActiveInterconnection):
    """A prismatic joint [azimuth, elevation, joint value, initial point (3)]: the drawer lies at the initial point
    moved by the joint value along the axis given by azimuth and elevation."""

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
                "uncertainty_joint": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
                "likelihood_grasped_drawer": (1,),
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
            axis = torch.cat(
                [
                    torch.cos(azimuth) * torch.sin(elevation),
                    torch.sin(azimuth) * torch.sin(elevation),
                    torch.cos(elevation),
                ],
                dim=0,
            )
            return initial_point + axis * state - position_drawer

        return connection_func
