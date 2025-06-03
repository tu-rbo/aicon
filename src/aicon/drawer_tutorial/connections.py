"""

Connections for the drawer experiment.
This module defines the connections between different components in the drawer manipulation environment,
including kinematics, sensing, and control connections.
"""

from typing import Union

import torch

from aicon.base_classes.connections import ActiveInterconnection
from aicon.drawer_experiment.util import get_sine_of_angles, likelihood_func_visible
from aicon.math.util_3d import exponential_map_se3, homogeneous_transform_inverse, log_map_se3
from aicon.middleware.util_ros import wait_for_tf_frame
from loguru import logger

# embodiment specific constants
GRASPING_ORIENTATION_DRAWER = [[-0.19754394844475631210, -0.01502825724660820927, 0.98017882977298076419],
                [-0.20747924073575563231, -0.97658970333946726328, -0.05678832084272268654],
                [ 0.95808598336199168877, -0.21458494869060928956, 0.18980133616634375926,]]
RELATIVE_PREGRASPING_POINT_DRAWER = [0.03, -0.01, 0.175]
RELATIVE_GRASPED_POINT_DRAWER = [0.005, -0.005, 0.155]
GRAVITY_ACC = 9.8067
FT_COM = [-0.001054, -0.016209, 0.090570] #[-0.022491, -0.006024, 0.076837]
FT_BIAS = [0, 0, 0]
FT_ROTATION = [[-0.70710361, -0.70710996, 0.0], [0.70710996, -0.70710361, 0.0], [0.0, 0.0, 1.0]]
EE_MASS = 0.5


def build_connections():
    """
    Build and return all connections needed for the drawer experiment.
    
    Returns:
        dict: Dictionary of connection builders
    """
    connections = {"GraspedDrawerKinematics": lambda device, dtype, mockbuild: EEDrawerGraspedConnection("GraspedDrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild),
                   "GraspedLikelihood": lambda device, dtype, mockbuild:DistGraspHandConnection("GraspedLikelihood", device=device, dtype=dtype, mockbuild=mockbuild),
                   "ForwardKinematics": lambda device, dtype, mockbuild:EEVeloConnection("ForwardKinematics", device=device, dtype=dtype, mockbuild=mockbuild),
                   "DirectMeasurement": lambda device, dtype, mockbuild:EEProprioConnection("DirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild),
                   "DrawerKinematics": lambda device, dtype, mockbuild: KinematicJointConnection("DrawerKinematics", device=device, dtype=dtype, mockbuild=mockbuild),
                   "DrawerDirectMeasurement": lambda device, dtype, mockbuild:DrawerDirectMeasurementConnection("DrawerDirectMeasurement", device=device, dtype=dtype, mockbuild=mockbuild)}
    return connections


class EEDrawerGraspedConnection(ActiveInterconnection):
    """
    Active Interconnection between the end-effector and the grasped drawer.
    The difference in position should be zero, because a grasped object moves together with the end-effector.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3, 3), "position_drawer": (3,),
                                "uncertainty_drawer": (3, 3), "likelihood_grasped_drawer": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,
                         # for on/off switching connection we can require gradient signs
                         # if we want to move the position of the drawer more we want it to be grasped
                         # if we want to change ee more, we do not want it to be grasped
                         required_signs_dict={"position_drawer" : {"likelihood_grasped_drawer": 1},
                                              "uncertainty_drawer": {"likelihood_grasped_drawer": 1},
                                              "pose_ee" : {"likelihood_grasped_drawer": -1}})

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, pose_ee):
            relative_vector = position_drawer - pose_ee
            return relative_vector

        return connection_func


class DistDrawerEEConnection(ActiveInterconnection):
    """
    Active Interconnection between the distance estimator, the end-effector and the drawer position.
    Our distance estimate should correspond to the distance between the end-effector and the drawer position.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (6,), "uncertainty_ee": (6, 6), "position_drawer": (3,),
                                "uncertainty_drawer": (3, 3), "distance_ee_drawer": (2,), "uncertainty_dist": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        H_grasping_offset = torch.eye(4, dtype=self.dtype, device=self.device)
        H_grasping_offset[:3, 3] = torch.tensor(RELATIVE_PREGRASPING_POINT_DRAWER, dtype=self.dtype, device=self.device)
        R_optimal_grasping = torch.tensor(
            GRASPING_ORIENTATION_DRAWER, dtype=self.dtype, device=self.device)

        def connection_func(distance_ee_drawer, pose_ee, position_drawer):
            H_ee = exponential_map_se3(pose_ee)
            grasping_point = torch.einsum("ij,jk->ik", H_ee, H_grasping_offset)[:3, 3]
            relative_vector = grasping_point - position_drawer
            dist_mean = torch.norm(relative_vector)
            # dist to good grasp pose also depends on a good orientation, but it only matters once we are somewhat close
            R_diff = torch.einsum("ij,jk->ik", H_ee[:3, :3], torch.transpose(R_optimal_grasping, 0, 1))
            angle_to_good_grasp_pose = torch.abs(
                torch.arccos(torch.clip((torch.trace(R_diff) - 1) / 2.0, -0.9999, 0.9999))) * 0.0
            dist_combined = torch.cat([dist_mean.unsqueeze(0), angle_to_good_grasp_pose.unsqueeze(0)])
            return dist_combined - distance_ee_drawer

        return connection_func


class DistGraspHandConnection(ActiveInterconnection):
    """
    Active Interconnection that links end-effector, drawer position, force measurements and gripper activation to the grasp likelihood.
    The likelihood is high if the end-effector is close to the drawer and the force measurements are high.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "position_drawer": (3,), "uncertainty_drawer": (3,3), "uncertainty_ee": (3,3),
                                "likelihood_grasped_drawer": (1,), "gripper_activation": (1,), "ee_force_mag_meas": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):

        def connection_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            expected_dist = torch.norm(position_drawer - pose_ee)
            likelihood_given_dist = torch.exp(-expected_dist * 4)
            innovation_from_dist = likelihood_given_dist - likelihood_grasped_drawer

            # hand and force measurements are only relevant if we are close (otherwise from other source...)
            if (expected_dist < 0.05 and ee_force_mag_meas > 10):
                # under 2N is just FT noise
                likelihood_from_hand_and_force = torch.clip(1 - torch.exp(-(ee_force_mag_meas - 5)), 0, 1) * gripper_activation
                # but we need an open hand to increase this likelihood
                # so we generate a negative gradient
                if (likelihood_grasped_drawer < 0.1) and (likelihood_from_hand_and_force < 0.1) and (gripper_activation > 0.5):
                    # print("Should open")
                    likelihood_from_hand_and_force = (likelihood_from_hand_and_force.detach() -
                                                      gripper_activation + gripper_activation.detach())
            else:
                # essentially no likelihood
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_dist) * 0.00000001

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()
            return innovation_from_dist + innovation_from_hand_and_force

        return connection_func


class VisibleEEDrawerConnection(ActiveInterconnection):
    """
    Active Interconnection between the end-effector, drawer position and drawer visibility likelihood.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (6,), "uncertainty_ee": (6, 6), "position_drawer": (3,), "uncertainty_drawer": (3, 3), "likelihood_visible_drawer": (1,),}, dtype=dtype,
                         device=device, mockbuild=mockbuild,)


    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera_color_optical_frame", timeout=5.0).to(dtype=self.dtype, device=self.device)
        except AssertionError:
            print("Fallback on saved transform")
            H_ee_to_cam = torch.tensor([[ 2.5214e-01, -9.6767e-01,  6.4494e-03, -6.3752e-02],
                                            [ 9.6769e-01,  2.5213e-01, -2.1942e-03, -1.6633e-02],
                                            [ 4.9714e-04,  6.7943e-03,  9.9998e-01,  4.0590e-02],
                                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
                                           dtype=self.dtype, device=self.device)

        def connection_func(likelihood_visible_drawer, pose_ee, position_drawer):
            likelihood = likelihood_func_visible(pose_ee, position_drawer, H_ee_to_cam)
            return likelihood - likelihood_visible_drawer

        return connection_func


class DrawerCameraEEConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (6,), "uncertainty_ee": (6, 6), "position_drawer": (3,), "uncertainty_drawer": (3, 3), "relative_position_in_CF_drawer":(3,), "likelihood_visible_drawer": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,)
        self.H_ee_to_cam = None

    def define_implicit_connection_function(self):
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera_color_optical_frame", timeout=5.0).to(dtype=self.dtype, device=self.device)
        except AssertionError:
            print("Fallback on saved transform")
            H_ee_to_cam = torch.tensor([[ 2.5214e-01, -9.6767e-01,  6.4494e-03, -6.3752e-02],
                                            [ 9.6769e-01,  2.5213e-01, -2.1942e-03, -1.6633e-02],
                                            [ 4.9714e-04,  6.7943e-03,  9.9998e-01,  4.0590e-02],
                                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
                                           dtype=self.dtype, device=self.device)

        def connection_func(position_drawer, pose_ee, relative_position_in_CF_drawer):
            relative_pos = torch.einsum("ki,ij,j->k",
                                        homogeneous_transform_inverse(H_ee_to_cam),
                                        homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                                        torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype, device=position_drawer.device)]))[:3]
            only_angles_sin_pred = get_sine_of_angles(relative_pos)
            # angles should be the same, dist of the measured point is always set for unit length because unknown (RGB)
            return relative_position_in_CF_drawer - only_angles_sin_pred

        return connection_func


class EEVeloConnection(ActiveInterconnection):
    """
    Links the end-effector estimator to the action component by applying the action velocity to the pose.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3,3), "action_velo_ee": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee):
            # Note pose is just position 
            # the velocity is applied to the pose
            # we are separating translation and rotation because otherwise there are artifacts from
            # the way they are coupled in Lee space and not the actual Euclidean space
            new_position = pose_ee + action_velo_ee
            return new_position

        return connection_func


class EEProprioConnection(ActiveInterconnection):
    """
    Active Interconnection between the estimated and measured end-effector position.
    the difference between both quantities should be zero.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3,3), "ee_pos_meas": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, ee_pos_meas):
            # the measured and the current pose should be the same
            return ee_pos_meas - pose_ee

        return connection_func

class DrawerDirectMeasurementConnection(ActiveInterconnection):
    """
    Active Interconnection between the drawer position and the measured drawer position.
    the difference between both quantities should be zero.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"position_drawer": (3,), "uncertainty_drawer": (3,3), "drawer_pos_meas": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, drawer_pos_meas):
            # the measured and the current pose should be the same
            return drawer_pos_meas - position_drawer

        return connection_func

class EEFTConnection(ActiveInterconnection):
    """
    Active Interconnection between the end-effector, the force measurements and the external force.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (6,), "uncertainty_ee": (6,6), "ee_force_measured": (6,), "ee_force_external": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):

        R_sensor = torch.tensor(FT_ROTATION,
                                dtype=self.dtype, device=self.device)
        H_com = torch.eye(4, dtype=self.dtype, device=self.device)
        H_com[:3, 3] = torch.tensor(FT_COM, dtype=self.dtype, device=self.device)
        expected_force = torch.tensor([0.0, 0.0, - GRAVITY_ACC * EE_MASS], dtype=self.dtype, device=self.device)
        FT_bias_term = torch.tensor(FT_BIAS[:3], dtype=self.dtype, device=self.device)

        def connection_func(pose_ee, ee_force_measured, ee_force_external):
            # the measured and the current pose should be the same
            H_ee = exponential_map_se3(pose_ee)
            rotated_meas_force = torch.einsum("ij,jk,k->i", H_ee[:3, :3], R_sensor, ee_force_measured[:3] - FT_bias_term)
            external_force_meas = rotated_meas_force - expected_force
            return ee_force_external - external_force_meas

        return connection_func


class KinematicJointConnection(ActiveInterconnection):
    """
    Active Interconnection between the kinematic joint and the drawer position.
    The postion of the drawer should correspond to the displacement along the kinematic joint axis.
    """
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"kinematic_joint": (6,), "uncertainty_joint": (6,6,),
                                "position_drawer": (3,), "uncertainty_drawer": (3,3,), "likelihood_grasped_drawer": (1,),
                                "time_since_hand_change": (2,)},
                         dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):

        def connection_func(kinematic_joint, position_drawer):
            azimuth, elevation, state, initial_point = kinematic_joint[0:1], kinematic_joint[1:2], kinematic_joint[2:3], kinematic_joint[3:]
            orientation_vector = torch.cat([torch.cos(azimuth) * torch.sin(elevation),
                                            torch.sin(azimuth) * torch.sin(elevation),
                                            torch.cos(elevation)], dim=0)
            translation = orientation_vector * state
            predicted_point = initial_point + translation
            return predicted_point - position_drawer

        return connection_func