"""
Component Graph for Real-Robot Object Pushing

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from aicon_real_robots.estimators.singularity import SingularityEstimator
from aicon_real_robots.goals.default_panda_goals import SingularityGoal
from .estimators import InteractionEstimator

# Per-joint gain and velocity-limit scale of the joint velocity controller. Joint 7 only rolls the
# camera about its optical axis, so it gets a larger share to keep the object in view without
# affecting the pusher tip.
J7_CONTRIBUTION_FACTORS = [1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 3.0]
J7_VMAX_SCALE = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.12]


def get_building_functions():
    """Estimators and the joint velocity controller, run on the robot PC. Sensors are mocked here and run via
    `get_sensor_only_building_functions` on the machines that own the hardware."""
    from .connections import build_connections
    from .estimators import (
        ContactsEstimator,
        PusherEstimator,
        ObjectEstimator,
        TwoDimKeypointEstimator,
        CorrectHeightEstimator,
    )
    from .goals import Push2GoalReal, GoalHeight
    from .sensors import ObjectPushingSensor
    from aicon_real_robots.actions.panda_controllers import JointVeloPandaControlViaROSCtrl
    from aicon_real_robots.estimators import (
        SelfCollisionEstimator,
        BodyEstimator,
        JointStateEstimator,
        ExternalForceEstimator,
        JointLimitEstimator,
        PostureEstimator,
    )
    from aicon_real_robots.sensors.ft_sensor import ForceTorqueSensor
    from aicon_real_robots.sensors.robot_joint_state_sensor import RobotJointSensor
    from aicon_real_robots.goals.default_panda_goals import SelfCollisionGoal, JointLimitOnLikelihoodGoal, PostureGoal
    from .estimators import AdaptedJointControlEEPoseEstimator

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
            gain=0.5,
            v_max=0.11,
            contribution_factors=J7_CONTRIBUTION_FACTORS,
            vmax_scale=J7_VMAX_SCALE,
        ),
        "JointStateEstimator": lambda mockbuild: JointStateEstimator(
            "JointStateEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "LinearVelocityJoints",
                    "DirectMeasurementJoints",
                    "RobotForwardKinematics",
                    "JointLimitLikelihoodConnection",
                    "PandaJoint2SingularityConnection",
                    "JointRangeUtilization",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "PostureEstimator": lambda mockbuild: PostureEstimator(
            "PostureEstimator",
            connections={k: connection_builders[k] for k in ("JointRangeUtilization",)},
            goals={
                "PostureGoal": lambda d, t, mockbuild=False: PostureGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "JointLimitEstimator": lambda mockbuild: JointLimitEstimator(
            "JointLimitEstimator",
            connections={k: connection_builders[k] for k in ("JointLimitLikelihoodConnection",)},
            goals={
                "JointLimitGoal": lambda d, t, mockbuild=False: JointLimitOnLikelihoodGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "EEPoseEstimator": lambda mockbuild: AdaptedJointControlEEPoseEstimator(
            "EEPoseEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "RobotForwardKinematics",
                    "EE2Pusher",
                    "EE2PusherHeight",
                    "KeyPoint2ObjectUncertainty",
                    "TwoDimKeypointsFrom3D",
                    "Keypoints2ObjectHeight",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "BodyEstimator": lambda mockbuild: BodyEstimator(
            "BodyEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "PandaJoint2BodyPoints",
                    "BodyPoints2SelfCollision",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "ContactsEstimator": lambda mockbuild: ContactsEstimator(
            "ContactsEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "KeyPoints2ContactPointsReal",
                    "ContactPoints2ContactLikelihoodReal",
                    "ContactPoints2PushVelocityDirectionsReal",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "PusherEstimator": lambda mockbuild: PusherEstimator(
            "PusherEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "EE2Pusher",
                    "EE2PusherHeight",
                    "HeightDifference",
                    "ObjectvsPush2Interactionlikelihood",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "ObjectEstimator": lambda mockbuild: ObjectEstimator(
            "ObjectEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "Contacts2ObjectPosition",
                    "KeyPoints2PositionReal",
                    "Keypoints2ObjectHeight",
                    "KeyPoint2ObjectUncertainty",
                    "HeightDifference",
                    "ObjectvsPush2Interactionlikelihood",
                    "PusherNonPenetration2ObjectPosition",
                )
            },
            goals={
                "Push2Goal": lambda d, t, mockbuild=False: Push2GoalReal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
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
        "TwoDimKeypointEstimator": lambda mockbuild: TwoDimKeypointEstimator(
            "ObjectEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "TwoDimKeypointsFrom3D",
                    "KeyPoints2PositionReal",
                    "Keypoints2ObjectHeight",
                    "KeyPoint2ObjectUncertainty",
                    "KeyPoints2ContactPointsReal",
                    "PusherNonPenetration2ObjectPosition",
                )
            },
            dtype=torch.float,
            mockbuild=mockbuild,
            no_differentiation=True,
        ),
        "CorrectHeightEstimator": lambda mockbuild: CorrectHeightEstimator(
            "CorrectHeightEstimator",
            connections={k: connection_builders[k] for k in ("HeightDifference",)},
            goals={
                "GoalHeight": lambda d, t, mockbuild=False: GoalHeight(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "SingularityEstimator": lambda mockbuild: SingularityEstimator(
            "SingularityEstimator",
            connections={k: connection_builders[k] for k in ("PandaJoint2SingularityConnection",)},
            goals={
                "SingularityGoal": lambda d, t, mockbuild=False: SingularityGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        "InteractionEstimator": lambda mockbuild: InteractionEstimator(
            "InteractionEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "Contacts2ObjectPosition",
                    "ObjectvsPush2Interactionlikelihood",
                )
            },
            dtype=torch.double,
            mockbuild=mockbuild,
        ),
        # sensors: run by get_sensor_only_building_functions on the robot's control PC and the vision PC
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
        "ObjectPushingSensor": lambda mockbuild: ObjectPushingSensor(
            "ObjectPushingSensor",
            connections={k: connection_builders[k] for k in ("Keypoints2ObjectHeight", "TwoDimKeypointsFrom3D")},
            ros_topic_base_color="",
            ros_topic_base_depth="",
            mockbuild=True,
        ),
    }
    frame_rates = {
        "ExternalForceEstimator": 45,
        "JointVelocities": 60,
        "JointStateEstimator": 45,
        "PostureEstimator": 45,
        "EEPoseEstimator": 30,
        "BodyEstimator": 15,
        "SelfCollisionEstimator": 15,
        "CorrectHeightEstimator": 30,
        "ContactsEstimator": 15,
        "PusherEstimator": 30,
        "ObjectEstimator": 15,
        "TwoDimKeypointEstimator": 15,
        "JointLimitEstimator": 30,
        "SingularityEstimator": 10,
        "InteractionEstimator": 10,
    }
    return component_builders, connection_builders, frame_rates


def get_sensor_only_building_functions():
    """Sensors and the visualizer, run on the vision / robot control PCs. The estimators are only mocked here so
    that the ROS middleware knows the graph's topology."""
    from .connections import build_connections
    from .estimators import (
        ContactsEstimator,
        PusherEstimator,
        ObjectEstimator,
        TwoDimKeypointEstimator,
        CorrectHeightEstimator,
        AdaptedJointControlEEPoseEstimator,
    )
    from .goals import Push2GoalReal, GoalHeight
    from .sensors import ObjectPushingSensor
    from .visualization import RealPusherExperimentVisualizer
    from aicon_real_robots.actions.panda_controllers import JointVeloPandaControlViaROSCtrl
    from aicon_real_robots.constants import FT_BIAS_PUSHER
    from aicon_real_robots.estimators import (
        SelfCollisionEstimator,
        BodyEstimator,
        JointStateEstimator,
        ExternalForceEstimator,
        JointLimitEstimator,
        PostureEstimator,
    )
    from aicon_real_robots.goals.default_panda_goals import SelfCollisionGoal, JointLimitOnLikelihoodGoal, PostureGoal
    from aicon_real_robots.sensors.camera_sensors import get_camtopic_name
    from aicon_real_robots.sensors.ft_sensor import ForceTorqueSensor
    from aicon_real_robots.sensors.robot_joint_state_sensor import RobotJointSensor

    connection_builders = build_connections()
    component_builders = {
        "ForceTorqueSensor": lambda mockbuild: ForceTorqueSensor(
            "ForceTorqueSensor",
            connections={k: connection_builders[k] for k in ("OwnWeightCompensation",)},
            quantity_name="ee_force_measured",
            ros_topic="/netft_data",
            bias=torch.tensor(FT_BIAS_PUSHER, dtype=torch.double),
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
        "ObjectPushingSensor": lambda mockbuild: ObjectPushingSensor(
            "ObjectPushingSensor",
            connections={k: connection_builders[k] for k in ("Keypoints2ObjectHeight", "TwoDimKeypointsFrom3D")},
            ros_topic_base_color=get_camtopic_name("realsense", "rgb", camera_name="camera2"),
            ros_topic_base_depth=get_camtopic_name("realsense", "depth_aligned", camera_name="camera2"),
            device=torch.device("cuda:0"),
            dtype=torch.float,
            mockbuild=mockbuild,
            scaling_factor=0.4,
        ),
        "Visualizer": lambda mockbuild: RealPusherExperimentVisualizer(
            "Visualizer", connections=connection_builders, dtype=torch.double, mockbuild=mockbuild
        ),
        # estimators and controllers: run by get_building_functions, mocked here
        "InteractionEstimator": lambda mockbuild: InteractionEstimator(
            "InteractionEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "Contacts2ObjectPosition",
                    "ObjectvsPush2Interactionlikelihood",
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "ExternalForceEstimator": lambda mockbuild: ExternalForceEstimator(
            "ExternalForceEstimator",
            connections={k: connection_builders[k] for k in ("OwnWeightCompensation",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
            no_differentiation=True,
        ),
        "JointVelocities": lambda mockbuild: JointVeloPandaControlViaROSCtrl(
            "JointVelocities",
            connections={k: connection_builders[k] for k in ("LinearVelocityJoints",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
            gain=10.0,
            v_max=0.12,
        ),
        "JointStateEstimator": lambda mockbuild: JointStateEstimator(
            "JointStateEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "LinearVelocityJoints",
                    "DirectMeasurementJoints",
                    "RobotForwardKinematics",
                    "JointLimitLikelihoodConnection",
                    "PandaJoint2SingularityConnection",
                    "JointRangeUtilization",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
        ),
        "PostureEstimator": lambda mockbuild: PostureEstimator(
            "PostureEstimator",
            connections={k: connection_builders[k] for k in ("JointRangeUtilization",)},
            goals={
                "PostureGoal": lambda d, t, mockbuild=False: PostureGoal(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
        ),
        "EEPoseEstimator": lambda mockbuild: AdaptedJointControlEEPoseEstimator(
            "EEPoseEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "RobotForwardKinematics",
                    "EE2Pusher",
                    "EE2PusherHeight",
                    "KeyPoint2ObjectUncertainty",
                    "TwoDimKeypointsFrom3D",
                    "Keypoints2ObjectHeight",
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
        ),
        "BodyEstimator": lambda mockbuild: BodyEstimator(
            "BodyEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "PandaJoint2BodyPoints",
                    "BodyPoints2SelfCollision",
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "ContactsEstimator": lambda mockbuild: ContactsEstimator(
            "ContactsEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "KeyPoints2ContactPointsReal",
                    "ContactPoints2ContactLikelihoodReal",
                    "ContactPoints2PushVelocityDirectionsReal",
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "PusherEstimator": lambda mockbuild: PusherEstimator(
            "PusherEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "EE2Pusher",
                    "EE2PusherHeight",
                    "HeightDifference",
                    "ObjectvsPush2Interactionlikelihood",
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "ObjectEstimator": lambda mockbuild: ObjectEstimator(
            "ObjectEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "Contacts2ObjectPosition",
                    "KeyPoints2PositionReal",
                    "Keypoints2ObjectHeight",
                    "KeyPoint2ObjectUncertainty",
                    "HeightDifference",
                    "Contacts2ObjectPosition",
                    "ObjectvsPush2Interactionlikelihood",
                    "PusherNonPenetration2ObjectPosition",
                )
            },
            goals={
                "Push2Goal": lambda d, t, mockbuild=False: Push2GoalReal(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "SelfCollisionEstimator": lambda mockbuild: SelfCollisionEstimator(
            "SelfCollisionEstimator",
            connections={k: connection_builders[k] for k in ("BodyPoints2SelfCollision",)},
            goals={
                "SelfCollisionGoal": lambda d, t, mockbuild=False: SelfCollisionGoal(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "TwoDimKeypointEstimator": lambda mockbuild: TwoDimKeypointEstimator(
            "ObjectEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "TwoDimKeypointsFrom3D",
                    "KeyPoints2PositionReal",
                    "Keypoints2ObjectHeight",
                    "KeyPoint2ObjectUncertainty",
                    "KeyPoints2ContactPointsReal",
                    "PusherNonPenetration2ObjectPosition",
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "CorrectHeightEstimator": lambda mockbuild: CorrectHeightEstimator(
            "CorrectHeightEstimator",
            connections={k: connection_builders[k] for k in ("HeightDifference",)},
            goals={
                "GoalHeight": lambda d, t, mockbuild=False: GoalHeight(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
        "JointLimitEstimator": lambda mockbuild: JointLimitEstimator(
            "JointLimitEstimator",
            connections={k: connection_builders[k] for k in ("JointLimitLikelihoodConnection",)},
            goals={
                "JointLimitGoal": lambda d, t, mockbuild=False: JointLimitOnLikelihoodGoal(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=True,
        ),
        "SingularityEstimator": lambda mockbuild: SingularityEstimator(
            "SingularityEstimator",
            connections={k: connection_builders[k] for k in ("PandaJoint2SingularityConnection",)},
            goals={
                "SingularityGoal": lambda d, t, mockbuild=False: SingularityGoal(
                    is_active=True, dtype=t, device=d, mockbuild=True
                )
            },
            dtype=torch.double,
            mockbuild=True,
        ),
    }
    frame_rates = {
        "ForceTorqueSensor": 60,
        "JointStateSensor": 45,
        "ObjectPushingSensor": 30,
        "Visualizer": 20,
    }
    return component_builders, connection_builders, frame_rates
