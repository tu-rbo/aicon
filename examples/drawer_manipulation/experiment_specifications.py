"""
Experiment Specification for Drawer Manipulation with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union, Iterable

import torch

from aicon_real_robots.actions.panda_controllers import JointVeloPandaControlViaROSCtrl
from aicon_real_robots.actions.rbo_hand_controller import SimpleHandSynergyAction
from .connections import build_connections
from aicon_real_robots.estimators import (
    ExternalForceEstimator,
    JointControlEEPoseEstimator,
    JointStateEstimator,
    BodyEstimator,
    SelfCollisionEstimator,
)
from .estimators import (
    DrawerPositionEstimator,
    DistEEDrawerEstimator,
    GraspedEstimator,
    VisibleEstimator,
    OcclusionAndObstaclePointsEstimator,
    ObstacleCollisionEstimator,
    EnvironmentPointEstimator,
)
from .goals import ObstacleAvoidanceGoal, DrawerOpenGoal
from aicon_real_robots.goals.default_panda_goals import JointLimitOnStateGoal, SelfCollisionGoal
from .sensors import CameraPointTracking, DLObjectInCameraSensor
from .visualization import DrawerExperimentVisualizer
from aicon_real_robots.sensors.camera_sensors import get_camtopic_name
from aicon_real_robots.sensors.ft_sensor import ForceTorqueSensor
from aicon_real_robots.sensors.robot_joint_state_sensor import RobotJointSensor
from aicon_real_robots.constants import FT_BIAS_RH3


def get_basic_building_functions_joint_level(
    initial_rotation_xy_joint: Union[float, None] = None,
    initial_uncertainty_scale_joint: Union[float, None] = None,
    sample_init_mean_joint: bool = False,
    initial_depth_drawer: Union[float, None] = None,
    initial_uncertainty_scale_drawer: Union[float, None] = None,
    sample_init_mean_drawer: bool = False,
    meas_noise_factor_drawer: float = 0.2,
    initially_grasped: bool = False,
    initial_drawer_pos: Union[None, Iterable] = None,
):
    connection_builders = build_connections()
    component_builders = {
        "ExternalForceEstimator": lambda mockbuild: ExternalForceEstimator(
            "ExternalForceEstimator",
            connections={k: connection_builders[k] for k in ("OwnWeightCompensation",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            no_differentiation=True,
        ),
        "JointVelocities": lambda mockbuild: JointVeloPandaControlViaROSCtrl(
            "JointVelocities",
            connections={k: connection_builders[k] for k in ("LinearVelocityJoints",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "SimpleHandSynergyAction": lambda mockbuild: SimpleHandSynergyAction(
            "SimpleHandSynergyAction",
            connections={k: connection_builders[k] for k in ("GraspedLikelihood",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            initially_grasped=initially_grasped,
        ),
        "JointStateEstimator": lambda mockbuild: JointStateEstimator(
            "JointStateEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "LinearVelocityJoints",
                    "DirectMeasurementJoints",
                    "RobotForwardKinematics",
                )
            },
            goals={
                "JointLimitGoal": lambda d, t, mockbuild=False: JointLimitOnStateGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "EEPoseEstimator": lambda mockbuild: JointControlEEPoseEstimator(
            "EEPoseEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "RobotForwardKinematics",
                    "ProjectiveGeometry",
                    "E[dist]",
                    "GraspedDrawerKinematics",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator(
            "DrawerPosEstimator",
            connections={
                k: connection_builders[k]
                for k in ("ProjectiveGeometry", "E[dist]", "GraspedDrawerKinematics", "DepthCameraProjection")
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            initial_depth=initial_depth_drawer,
            initial_uncertainty_scale=initial_uncertainty_scale_drawer,
            sample_init_mean=sample_init_mean_drawer,
            meas_noise_factor=meas_noise_factor_drawer,
            initial_drawer_pos=initial_drawer_pos,
            goals={
                "ReduceDistToPoint": lambda d, t, mockbuild=False: DrawerOpenGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
        ),
        "DistanceEstimator": lambda mockbuild: DistEEDrawerEstimator(
            "DistanceEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "E[dist]",
                    "GraspedLikelihood",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator(
            "GraspLikelihoodEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "GraspedDrawerKinematics",
                    "GraspedLikelihood",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
            initially_grasped=initially_grasped,
        ),
        "VisibleLikelihoodEstimator": lambda mockbuild: VisibleEstimator(
            "VisibleLikelihoodEstimator",
            connections={k: connection_builders[k] for k in ("VisibleLikelihood",)},
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "OcclusionAndObstaclePointsEstimator": lambda mockbuild: OcclusionAndObstaclePointsEstimator(
            "OcclusionAndObstaclePointsEstimator",
            connections={k: connection_builders[k] for k in ("ObstacleCollision", "Environment2ObstacleConnection")},
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "BodyEstimator": lambda mockbuild: BodyEstimator(
            "BodyEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "ObstacleCollision",
                    "PandaJoint2BodyPoints",
                    "BodyPoints2SelfCollision",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "SelfCollisionEstimator": lambda mockbuild: SelfCollisionEstimator(
            "SelfCollisionEstimator",
            connections={k: connection_builders[k] for k in ("BodyPoints2SelfCollision",)},
            goals={
                "SelfCollisionGoal": lambda d, t, mockbuild=False: SelfCollisionGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "ObstacleCollisionEstimator": lambda mockbuild: ObstacleCollisionEstimator(
            "ObstacleCollisionEstimator",
            connections={k: connection_builders[k] for k in ("ObstacleCollision",)},
            goals={
                "ObstacleAvoidanceGoal": lambda d, t, mockbuild=False: ObstacleAvoidanceGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "EnvironmentPointEstimator": lambda mockbuild: EnvironmentPointEstimator(
            "EnvironmentPointEstimator",
            connections={k: connection_builders[k] for k in ("ProjectiveGeometryEnvironment",)},
            device=torch.device("cuda:1"),
            dtype=torch.double,
            mockbuild=mockbuild,
            no_differentiation=True,
        ),
        "Visualizer": lambda mockbuild: DrawerExperimentVisualizer(
            "Visualizer", connections=connection_builders, dtype=torch.double, mockbuild=mockbuild
        ),
        # sensors: run by get_sensor_only_building_functions on the robot control and vision PCs, mocked here
        "CameraPointsSensor": lambda mockbuild: CameraPointTracking(
            "CameraPointsSensor",
            connections={k: connection_builders[k] for k in ("ProjectiveGeometry",)},
            ros_topic_base_color="",
            ros_topic_base_depth="",
            mockbuild=True,
        ),
        "CameraDrawerPositionSensor": lambda mockbuild: DLObjectInCameraSensor(
            "CameraDrawerPositionSensor",
            connections={k: connection_builders[k] for k in ("ProjectiveGeometry",)},
            ros_topic_base="",
            mockbuild=True,
        ),
        "ForceTorqueSensor": lambda mockbuild: ForceTorqueSensor(
            "ForceTorqueSensor",
            connections={k: connection_builders[k] for k in ("OwnWeightCompensation",)},
            quantity_name="ee_force_measured",
            ros_topic="",
            bias="",
            mockbuild=True,
        ),
        "JointStateSensor": lambda mockbuild: RobotJointSensor(
            "JointStateSensor",
            ros_topic_base="",
            connections={k: connection_builders[k] for k in ("DirectMeasurementJoints",)},
            mockbuild=True,
        ),
    }
    frame_rates = {
        "CameraDrawerPositionSensor": 10,
        "ForceTorqueSensor": 10,
        "ExternalForceEstimator": 10,
        "JointStateSensor": 30,
        "JointVelocities": 30,
        "JointStateEstimator": 30,
        "SimpleHandSynergyAction": 10,
        "EEPoseEstimator": 20,
        "DrawerPosEstimator": 10,
        "DistanceEstimator": 10,
        "GraspLikelihoodEstimator": 10,
        "VisibleLikelihoodEstimator": 5,
        "KinematicJointEstimator": 5,
        "EnvironmentPointEstimator": 10,
        "ObstacleCollisionEstimator": 20,
        "OcclusionAndObstaclePointsEstimator": 10,
        "BodyEstimator": 20,
        "SelfCollisionEstimator": 20,
    }
    return component_builders, connection_builders, frame_rates


def get_sensor_only_building_functions():
    connection_builders = build_connections()
    component_builders = {
        "CameraPointsSensor": lambda mockbuild: CameraPointTracking(
            "CameraPointsSensor",
            connections={k: connection_builders[k] for k in ("ProjectiveGeometry",)},
            ros_topic_base_color=get_camtopic_name("realsense", "rgb", camera_name="camera1"),
            ros_topic_base_depth=get_camtopic_name("realsense", "depth_aligned", camera_name="camera1"),
            device=torch.device("cuda:0"),
            dtype=torch.float,
            mockbuild=mockbuild,
            scaling_factor=0.2,
        ),
        "CameraDrawerPositionSensor": lambda mockbuild: DLObjectInCameraSensor(
            "CameraDrawerPositionSensor",
            connections={k: connection_builders[k] for k in ("ProjectiveGeometry",)},
            ros_topic_base=get_camtopic_name("realsense", "rgb", camera_name="camera1"),
            device=torch.device("cuda:1"),
            dtype=torch.float,
            mockbuild=mockbuild,
            scaling_factor=0.5,
        ),
        "ForceTorqueSensor": lambda mockbuild: ForceTorqueSensor(
            "ForceTorqueSensor",
            connections={k: connection_builders[k] for k in ("OwnWeightCompensation",)},
            quantity_name="ee_force_measured",
            ros_topic="/netft_data",
            bias=torch.tensor(FT_BIAS_RH3, dtype=torch.double),
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "JointStateSensor": lambda mockbuild: RobotJointSensor(
            "JointStateSensor",
            ros_topic_base="",
            connections={k: connection_builders[k] for k in ("DirectMeasurementJoints",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
    }
    frame_rates = {
        "CameraPointsSensor": 10,
        "CameraDrawerPositionSensor": 5,
        "ForceTorqueSensor": 10,
        "JointStateSensor": 20,
        "Visualizer": 5,
    }
    return component_builders, connection_builders, frame_rates
