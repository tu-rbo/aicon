"""
Extended Drawer Example: Graph Specification

Builds on the tutorial graph with a 6D end-effector pose, a wrist camera and explicit estimates of the
end-effector-to-handle distance and of the handle's visibility. With triangulation the camera only measures the
bearing to the handle, so the handle's depth has to be inferred from several viewpoints; the robot moves the camera
to make the handle visible and its position certain enough before it grasps. Without triangulation the handle's
position is measured directly and the visibility estimate is not needed.

Authors: James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers
License: MIT
"""

__author__ = "James Reynaldo, Adrian Pfisterer, Patrick Lowin, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from ..tutorial.actions import GripperAction, VeloEEAction
from ..tutorial.goals import DrawerOpenViaJointGoal
from .connections import build_connections
from .estimators import (
    DistEEDrawerEstimator,
    DrawerPositionEstimator,
    EEPoseEstimator,
    GraspedEstimator,
    KinematicJointEstimator,
    VisibleEstimator,
)
from .sensors import BearingSensor, DrawerPoseSensor, EEForceSensor, EEPoseSensor

# initial arm joint positions (Panda, 7 joints); the handle is in the camera's view
INITIAL_QPOS = [-0.58865829, 0.70879424, 0.0393395, -1.81579848, 1.08319597, 1.37122098, -0.23145539]
# target joint value of the drawer [m]; its joint range is [-0.04, 0]
OPEN_VALUE = -0.1


def get_building_functions(env, triangulation: bool = True, noise_scale: float = 1.0, depth_prior_noise: float = 0.01):
    """Builders for the extended graph, with the sensors reading from `env` (see env.make_env). `noise_scale` scales
    the sensor noise, `depth_prior_noise` [m] is the noise of the initial depth guess when triangulating."""
    connection_builders = build_connections()
    cpu, double = torch.device("cpu"), torch.double

    def connections(*names):
        return {k: connection_builders[k] for k in names}

    # where the handle position comes from
    if triangulation:
        handle_sensor = (
            "BearingSensor",
            lambda mockbuild: BearingSensor(
                "BearingSensor",
                connections("ProjectiveGeometry"),
                env,
                noise_scale=noise_scale,
                device=cpu,
                dtype=double,
                mockbuild=mockbuild,
            ),
        )
        camera = ("ProjectiveGeometry", "VisibleLikelihood")
    else:
        handle_sensor = (
            "DrawerPosSensor",
            lambda mockbuild: DrawerPoseSensor(
                "DrawerPosSensor",
                connections("DrawerDirectMeasurement"),
                env,
                noise_scale=noise_scale,
                device=cpu,
                dtype=double,
                mockbuild=mockbuild,
            ),
        )
        camera = ()
    handle_connections = camera if triangulation else ("DrawerDirectMeasurement",)

    # the sequential middleware updates the components in this order
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction(
            "GripperAction", connections("GraspedLikelihood"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor(
            "EEForceSensor",
            connections("GraspedLikelihood"),
            env,
            noise_scale=noise_scale,
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "EEPosSensor": lambda mockbuild: EEPoseSensor(
            "EEPosSensor",
            connections("DirectMeasurement"),
            env,
            noise_scale=noise_scale,
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        handle_sensor[0]: handle_sensor[1],
        "EEVelocities": lambda mockbuild: VeloEEAction(
            "EEVelocities", connections("ForwardKinematics"), device=cpu, dtype=double, mockbuild=mockbuild
        ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator(
            "EEPoseEstimator",
            connections("ForwardKinematics", "DirectMeasurement", "GraspedDrawerKinematics", "E[dist]", *camera),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator(
            "DrawerPosEstimator",
            connections("GraspedDrawerKinematics", "DrawerKinematics", "E[dist]", *handle_connections),
            env,
            triangulation=triangulation,
            depth_prior_noise=depth_prior_noise,
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
        ),
        "DistanceEstimator": lambda mockbuild: DistEEDrawerEstimator(
            "DistanceEstimator",
            connections("E[dist]", "GraspedLikelihood"),
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
    }
    if triangulation:
        component_builders["VisibleEstimator"] = lambda mockbuild: VisibleEstimator(
            "VisibleEstimator",
            connections("VisibleLikelihood", "ProjectiveGeometry"),
            device=cpu,
            dtype=double,
            mockbuild=mockbuild,
            # its only quantity is one-dimensional, so its gradients cannot be combined by projection
            no_backwards_combinations=True,
        )
    component_builders["KinematicJointEstimator"] = lambda mockbuild: KinematicJointEstimator(
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
    )
    frame_rates = {k: 10 for k in component_builders}
    return component_builders, connection_builders, frame_rates
