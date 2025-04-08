from typing import Union, Iterable

import torch

from aicon.drawer_experiment.actions import VeloEEAction, HandSynergyAction
from aicon.drawer_experiment.connections import build_connections
from aicon.drawer_experiment.estimators import ExternalForceEstimator, EEPoseEstimator, DrawerPositionEstimator, \
    DistEEDrawerEstimator, GraspedEstimator, VisibleEstimator, KinematicJointEstimator
from aicon.drawer_experiment.goals import DrawerOpenViaJointGoal
from aicon.drawer_experiment.sensors import CameraDrawerPosSensor
from aicon.sensors.camera_sensors import get_camtopic_name
from aicon.sensors.ft_sensor import ForceTorqueSensor
from aicon.sensors.ros_tf_sensor import ROSTFSensor


def get_building_functions_basic_drawer_motion(initial_rotation_xy_joint : Union[float, None] = None,
                 initial_uncertainty_scale_joint : Union[float, None] = None,
                 sample_init_mean_joint : bool = False, initial_depth_drawer : Union[float, None] = None,
                 initial_uncertainty_scale_drawer : Union[float, None] = None,
                 sample_init_mean_drawer : bool = False,
                 meas_noise_factor_drawer : float = 0.025, initially_grasped : bool =False,
                 initial_drawer_pos : Union[None, Iterable] =None):
    connection_builders = build_connections()
    print("drawer params", initial_depth_drawer,
          initial_uncertainty_scale_drawer,
          sample_init_mean_drawer,
          meas_noise_factor_drawer)
    component_builders = {
        "CameraDrawerPositionSensor": lambda mockbuild: CameraDrawerPosSensor("CameraDrawerPositionSensor",
                                                                              connections={k: connection_builders[k] for
                                                                                           k in (
                                                                                               "ProjectiveGeometry",)},
                                                                              ros_topic_base=get_camtopic_name(
                                                                                  "realsense", "rgb"),
                                                                              lower_cam_threshold=torch.tensor(
                                                                                  (0, 70, 35)),
                                                                              upper_cam_threshold=torch.tensor(
                                                                                  (15, 120, 80,)),
                                                                              device=torch.device("cpu"),
                                                                              dtype=torch.double, mockbuild=mockbuild),
        "EEProprioSensor": lambda mockbuild: ROSTFSensor("EEProprioSensor",
                                                         connections={k: connection_builders[k] for k in
                                                                      ("DirectMeasurement",)},
                                                         source_frame="panda_link0", target_frame="panda_link8",
                                                         device=torch.device("cpu"), quantity_name="pose_measured_ee",
                                                         dtype=torch.double, mockbuild=mockbuild),
        "ForceTorqueSensor": lambda mockbuild: ForceTorqueSensor("ForceTorqueSensor",
                                                                 connections={k: connection_builders[k] for k in
                                                                              ("OwnWeightCompensation", )},
                                                                 quantity_name="ee_force_measured",
                                                                 ros_topic="/ft_sensor/data",
                                                                 bias=torch.tensor(
                                                                     [1.45336144, -2.50235184, 2.55458085, -0.0460468,
                                                                      0.01755776, 0.04572964, ], dtype=torch.double),
                                                                 device=torch.device("cpu"),
                                                                 dtype=torch.double, mockbuild=mockbuild),
        "ExternalForceEstimator": lambda mockbuild: ExternalForceEstimator("ExternalForceEstimator",
                                                                 connections={k: connection_builders[k] for k in
                                                                              ("OwnWeightCompensation", )},
                                                                 device=torch.device("cpu"),
                                                                 dtype=torch.double, mockbuild=mockbuild),
        "EEVelocities": lambda mockbuild: VeloEEAction("EEVelocities",
                                                       connections={k: connection_builders[k] for k in
                                                                    ("ForwardKinematics",)},
                                                       device=torch.device("cpu"),
                                                       dtype=torch.double, mockbuild=mockbuild),
        "HandSynergyAction": lambda mockbuild: HandSynergyAction("HandSynergyAction",
                                                                 connections={k: connection_builders[k] for k in
                                                                              ("GraspedLikelihood",)},
                                                                 device=torch.device("cpu"),
                                                                 dtype=torch.double, mockbuild=mockbuild,
                                                                 initially_grasped=initially_grasped),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator("EEPoseEstimator",
                                                             connections={k: connection_builders[k] for k in
                                                                          ("ForwardKinematics",
                                                                           "DirectMeasurement",
                                                                           "ProjectiveGeometry",
                                                                           "E[dist]",
                                                                           "GraspedDrawerKinematics",
                                                                           )},
                                                             # goals={"StayInWorkspace": lambda d, t,
                                                             #                                    mockbuild=False: WorkspaceGoal(
                                                             #     is_active=True, dtype=t, device=d,
                                                             #     mockbuild=mockbuild)},
                                                             device=torch.device("cpu"),
                                                             dtype=torch.double, mockbuild=mockbuild),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator("DrawerPosEstimator",
                                                                        connections={k: connection_builders[k] for k in
                                                                                     ("ProjectiveGeometry",
                                                                                      "E[dist]",
                                                                                      "GraspedDrawerKinematics",
                                                                                      )},
                                                                        # goals={"ReduceDistToPoint": lambda d, t,
                                                                        #                                    mockbuild=False: DrawerOpenGoal(
                                                                        #     is_active=True, dtype=t, device=d,
                                                                        #     mockbuild=mockbuild)},
                                                                        # goals={"MakeDrawerPositionCertain": lambda d, t,
                                                                        #                                    mockbuild=False: DrawerUncertaintyGoal(
                                                                        #     is_active=True, dtype=t, device=d,
                                                                        #     mockbuild=mockbuild)},
                                                                        device=torch.device("cpu"),
                                                                        dtype=torch.double, mockbuild=mockbuild,
                                                                        initial_depth = initial_depth_drawer,
                                                                        initial_uncertainty_scale = initial_uncertainty_scale_drawer,
                                                                        sample_init_mean = sample_init_mean_drawer,
                                                                        meas_noise_factor = meas_noise_factor_drawer,
                                                                        initial_drawer_pos = initial_drawer_pos,
                                                                        ),
        "DistanceEstimator": lambda mockbuild: DistEEDrawerEstimator("DistanceEstimator",
                                                                     connections={k: connection_builders[k] for k in
                                                                                  ("E[dist]",
                                                                                   "GraspedLikelihood",
                                                                                   )},
                                                                     device=torch.device("cpu"),
                                                                     dtype=torch.double, mockbuild=mockbuild),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator("GraspLikelihoodEstimator",
                                                                       connections={k: connection_builders[k] for k in
                                                                                    ("GraspedDrawerKinematics",
                                                                                     "GraspedLikelihood",
                                                                                     )},
                                                                       dtype=torch.double, mockbuild=mockbuild,
                                                                       initially_grasped=initially_grasped
                                                                       ),
        "VisibleLikelihoodEstimator": lambda mockbuild: VisibleEstimator("VisibleLikelihoodEstimator",
                                                                         connections={k: connection_builders[k] for k in
                                                                                      ("VisibleLikelihood",
                                                                                       )},
                                                                         dtype=torch.double, mockbuild=mockbuild
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
                                                                             # goals={"MakeJointCertain": lambda d, t,
                                                                             #                                    mockbuild=False: JointUncertaintyGoal(
                                                                             #     is_active=True, dtype=t, device=d,
                                                                             #     mockbuild=mockbuild)},
                                                                             initial_rotation_xy=initial_rotation_xy_joint,
                                                                            initial_uncertainty_scale=initial_uncertainty_scale_joint,
                                                                            sample_init_mean=sample_init_mean_joint,
                                                                         )
    }
    frame_rates = {
        "CameraDrawerPositionSensor": 10,
        "EEProprioSensor": 15,
        "ForceTorqueSensor": 10,
        "ExternalForceEstimator": 10,
        "EEVelocities": 30,
        "HandSynergyAction": 10,
        "EEPoseEstimator": 15,
        "DrawerPosEstimator": 10,
        "DistanceEstimator": 15,
        "GraspLikelihoodEstimator": 15,
        "VisibleLikelihoodEstimator": 10,
        "KinematicJointEstimator": 10,
    }
    return component_builders, connection_builders, frame_rates