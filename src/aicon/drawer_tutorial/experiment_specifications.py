import torch

from aicon.drawer_tutorial.actions import GripperAction, VeloEEAction
from aicon.drawer_tutorial.connections import build_connections
from aicon.drawer_tutorial.estimators import (
    DrawerPositionEstimator,
    EEPoseEstimator,
    GraspedEstimator,
    KinematicJointEstimator,
)
from aicon.drawer_tutorial.goals import DrawerOpenViaJointGoal
from aicon.drawer_tutorial.sensors import DrawerPoseSenser, EEForceSensor, EEPoseSensor


def get_building_functions_basic_drawer_motion(sim_env_pointer):
    """
    Builds the connections and components for the basic drawer motion experiment.
    """
    connection_builders = build_connections()
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction("GripperAction",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor("EEForceSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "EEPosSensor": lambda mockbuild: EEPoseSensor("EEPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "DrawerPosSensor": lambda mockbuild: DrawerPoseSenser("DrawerPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DrawerDirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "EEVelocities": lambda mockbuild: VeloEEAction("EEVelocities",
                                                       connections={k: connection_builders[k] for k in
                                                                    ("ForwardKinematics",)},
                                                       device=torch.device("cpu"),
                                                       dtype=torch.double, mockbuild=mockbuild,
                                                       ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator("EEPoseEstimator",
                                                             connections={k: connection_builders[k] for k in
                                                                          ("ForwardKinematics",
                                                                           "DirectMeasurement",
                                                                           "GraspedDrawerKinematics",
                                                                           "GraspedLikelihood",
                                                                           )},
                                                             device=torch.device("cpu"),
                                                             dtype=torch.double, mockbuild=mockbuild),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator("DrawerPosEstimator",
                                                                        connections={k: connection_builders[k] for k in
                                                                                     ("DrawerDirectMeasurement",
                                                                                      "GraspedDrawerKinematics",
                                                                                      "DrawerKinematics",
                                                                                     )},
                                                                        device=torch.device("cpu"),
                                                                        dtype=torch.double, mockbuild=mockbuild,
                                                                        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator("GraspLikelihoodEstimator",
                                                                       connections={k: connection_builders[k] for k in
                                                                                    ("GraspedDrawerKinematics",
                                                                                     "GraspedLikelihood",
                                                                                     "DrawerKinematics",
                                                                                     )},
                                                                       dtype=torch.double, mockbuild=mockbuild,
                                                                       ),
        "KinematicJointEstimator": lambda mockbuild: KinematicJointEstimator("KinematicJointEstimator",
                                                                         connections={k: connection_builders[k] for k in
                                                                                      ("DrawerKinematics",
                                                                                       )},
                                                                         dtype=torch.double, mockbuild=mockbuild,
                                                                         goals={"ReduceJointStateDifference": lambda d, t,
                                                                                                            mockbuild=False: DrawerOpenViaJointGoal(
                                                                             is_active=True, dtype=t, device=d,
                                                                             mockbuild=mockbuild)},
                                                                         )
    }
    frame_rates = {
        "GripperAction": 10,
        "EEForceSensor": 10,
        "EEPosSensor": 10,
        "DrawerPosSensor": 10,
        "EEVelocities": 10,
        "EEPoseEstimator": 10,
        "DrawerPosEstimator": 10,
        "GraspLikelihoodEstimator": 10,
        "KinematicJointEstimator": 10,
    }
    return component_builders, connection_builders, frame_rates