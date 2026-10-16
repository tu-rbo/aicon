"""
Drawer Tutorial: Graph Specification

Five estimators (end-effector, drawer handle, grasp likelihood, prismatic joint) between three simulated sensors
and two actions (end-effector velocity, gripper). The only goal is on the drawer's joint value; reaching the handle,
grasping it and pulling it out emerge from the gradients of that goal (see TUTORIAL.md).

Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from .actions import GripperAction, VeloEEAction
from .connections import build_connections
from .estimators import DrawerPositionEstimator, EEPoseEstimator, GraspedEstimator, KinematicJointEstimator
from .goals import DrawerOpenViaJointGoal
from .sensors import DrawerPoseSensor, EEForceSensor, EEPoseSensor

# initial arm joint positions (Panda, 7 joints)
INITIAL_QPOS = [-0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7]
# target joint value of the drawer [m]; its joint range is [-0.04, 0]
OPEN_VALUE = -0.1


def get_building_functions(env):
    """Builders for the tutorial graph, with the sensors reading from `env` (see env.make_env)."""
    connection_builders = build_connections()
    cpu, double = torch.device("cpu"), torch.double

    def connections(*names):
        return {k: connection_builders[k] for k in names}

    # the sequential middleware updates the components in this order
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction(
            "GripperAction", connections("GraspedLikelihood"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor(
            "EEForceSensor", connections("GraspedLikelihood"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPosSensor": lambda mockbuild: EEPoseSensor(
            "EEPosSensor", connections("DirectMeasurement"), env, device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "DrawerPosSensor": lambda mockbuild: DrawerPoseSensor(
            "DrawerPosSensor",
            connections("DrawerDirectMeasurement"),
            env,
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "EEVelocities": lambda mockbuild: VeloEEAction(
            "EEVelocities", connections("ForwardKinematics"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator(
            "EEPoseEstimator",
            connections("ForwardKinematics", "DirectMeasurement", "GraspedDrawerKinematics", "GraspedLikelihood"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator(
            "DrawerPosEstimator",
            connections("DrawerDirectMeasurement", "GraspedDrawerKinematics", "DrawerKinematics"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator(
            "GraspLikelihoodEstimator",
            connections("GraspedDrawerKinematics", "GraspedLikelihood", "DrawerKinematics"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "KinematicJointEstimator": lambda mockbuild: KinematicJointEstimator(
            "KinematicJointEstimator",
            connections("DrawerKinematics"),
            goals={
                "ReduceJointStateDifference": lambda d, t, mockbuild=False: DrawerOpenViaJointGoal(
                    is_active=True, open_value=OPEN_VALUE, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
    }
    frame_rates = {k: 10 for k in component_builders}
    return component_builders, connection_builders, frame_rates
