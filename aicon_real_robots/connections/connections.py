"""
Real-World Robots -- Useful Connections for AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection
from aicon_base.inference.util import gradient_preserving_clipping
from aicon_real_robots.constants import (
    PANDA_DH_ALPHA,
    PANDA_DH_D,
    PANDA_DH_A,
    NUM_BODY_POINTS,
    FT_ROTATION_RH3,
    FT_COM_RH3,
    GRAVITY_ACC,
    EE_MASS_RH3,
    FT_ROTATION_PUSHER,
    EE_MASS_PUSHER,
    FT_COM_PUSHER,
)
from aicon_base.math.robot_kinematics import dh_params_to_htransform_batch
from aicon_base.math.se3_and_SE3 import exponential_map_se3, log_map_se3


class EEVeloConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "action_velo_ee": (6,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee):
            # the velocity is applied to the pose
            # we are separating translation and rotation because otherwise there are artifacts from
            # the way they are coupled in Lee space and not the actual Euclidean space
            pose = exponential_map_se3(pose_ee)
            velocity = exponential_map_se3(action_velo_ee)
            new_pose = torch.einsum("ij,jk->ik", pose, velocity)
            return log_map_se3(new_pose)

        return connection_func


class EEProprioConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "pose_measured_ee": (6,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, pose_measured_ee):
            # the measured and the current pose should be the same
            return pose_measured_ee - pose_ee

        return connection_func


class JointsVeloConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "action_velo_joints": (7,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        gains = torch.tensor([5.0, 5.0, 3.0, 3.0, 2.0, 1.0, 1.0], dtype=self.dtype, device=self.device)
        gains = torch.ones_like(gains)

        def connection_func(robot_joint_states, action_velo_joints):
            new_joint_states = robot_joint_states + gains * action_velo_joints
            return new_joint_states

        return connection_func


class PandaJoint2EEConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "pose_ee": (6,), "uncertainty_ee": (6, 6)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        append_zero = torch.zeros((1,), dtype=self.dtype, device=self.device)
        alpha = torch.tensor(PANDA_DH_ALPHA, dtype=self.dtype, device=self.device)
        d = torch.tensor(PANDA_DH_D, dtype=self.dtype, device=self.device)
        a = torch.tensor(PANDA_DH_A, dtype=self.dtype, device=self.device)

        def connection_func(pose_ee, robot_joint_states):
            transforms = dh_params_to_htransform_batch(
                theta=torch.cat([robot_joint_states, append_zero]), a=a, alpha=alpha, d=d
            )
            zero2one = torch.einsum("ij,jk->ik", transforms[0], transforms[1])
            zero2two = torch.einsum("ij,jk->ik", zero2one, transforms[2])
            zero2three = torch.einsum("ij,jk->ik", zero2two, transforms[3])
            zero2four = torch.einsum("ij,jk->ik", zero2three, transforms[4])
            zero2five = torch.einsum("ij,jk->ik", zero2four, transforms[5])
            # last joint of panda does not affect translation, but due to numerical issues often does...
            # this is a fix for that
            zero2six_trans = torch.einsum("ij,jk->ik", zero2five, transforms[6].detach())
            zero2six_rot = torch.einsum("ij,jk->ik", zero2five, transforms[6])
            zero2EE_trans = torch.einsum("ij,jk->ik", zero2six_trans, transforms[7])
            pose_ee_rot = log_map_se3(zero2six_rot)
            pose_ee_trans = log_map_se3(zero2EE_trans)
            return torch.cat([pose_ee_trans[:3], pose_ee_rot[3:]]) - pose_ee

        return connection_func


class PandaJoint2SingularityConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "likelihood_ee_singularity": (1,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        append_zero = torch.zeros((1,), dtype=self.dtype, device=self.device)
        alpha = torch.tensor(PANDA_DH_ALPHA, dtype=self.dtype, device=self.device)
        d = torch.tensor(PANDA_DH_D, dtype=self.dtype, device=self.device)
        a = torch.tensor(PANDA_DH_A, dtype=self.dtype, device=self.device)

        def joints2ee(robot_joint_states):
            transforms = dh_params_to_htransform_batch(
                theta=torch.cat([robot_joint_states, append_zero]), a=a, alpha=alpha, d=d
            )
            zero2one = torch.einsum("ij,jk->ik", transforms[0], transforms[1])
            zero2two = torch.einsum("ij,jk->ik", zero2one, transforms[2])
            zero2three = torch.einsum("ij,jk->ik", zero2two, transforms[3])
            zero2four = torch.einsum("ij,jk->ik", zero2three, transforms[4])
            zero2five = torch.einsum("ij,jk->ik", zero2four, transforms[5])
            zero2six = torch.einsum("ij,jk->ik", zero2five, transforms[6])
            zero2EE = torch.einsum("ij,jk->ik", zero2six, transforms[7])
            pose_ee = log_map_se3(zero2EE)
            return pose_ee

        w0 = 1e-3
        p = 2.0

        def connection_func(likelihood_ee_singularity, robot_joint_states):
            Jacobian = torch.func.jacfwd(joints2ee, argnums=(0,), has_aux=False, randomness="same")(robot_joint_states)[
                0
            ]
            manipulability = torch.sqrt(torch.det(torch.einsum("ij,kj->ik", Jacobian, Jacobian))).unsqueeze(0)
            if manipulability < 0.1:
                likelihood = 1 - torch.exp(-((w0 / gradient_preserving_clipping(manipulability, 1e-6, 1.0)) ** p))
            else:
                likelihood = torch.zeros_like(likelihood_ee_singularity)
            print("Manipulability", manipulability)
            return likelihood_ee_singularity - likelihood

        return connection_func


class PandaWithHandJoint2BodyPointsConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "body_points": (NUM_BODY_POINTS, 3)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        alpha = torch.tensor(PANDA_DH_ALPHA, dtype=self.dtype, device=self.device)
        d = torch.tensor(PANDA_DH_D, dtype=self.dtype, device=self.device)
        a = torch.tensor(PANDA_DH_A, dtype=self.dtype, device=self.device)
        base_point1 = torch.zeros(
            (
                1,
                3,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        base_point2 = torch.tensor([[-0.07, 0, 0.05]], dtype=self.dtype, device=self.device)
        append_zero = torch.zeros((1,), dtype=self.dtype, device=self.device)
        from aicon_ros_python_wrapping.util_ros import wait_for_tf_frame

        try:
            H_fingers = wait_for_tf_frame("panda_link8", "middle_tip_link10", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
            H_thumb = wait_for_tf_frame("panda_link8", "thumb_tip_link10", timeout=2.0).to(
                dtype=self.dtype, device=self.device
            )
        except AssertionError:
            print("No hand considered in body points!")
            H_fingers = torch.eye(4, dtype=self.dtype, device=self.device)
            H_thumb = torch.eye(4, dtype=self.dtype, device=self.device)

        def connection_func(body_points, robot_joint_states):
            # point names are defined based on frame names here
            # https://frankaemika.github.io/docs/control_parameters.html
            transforms = dh_params_to_htransform_batch(
                theta=torch.cat([robot_joint_states, append_zero]), a=a, alpha=alpha, d=d
            )
            zero2one = torch.einsum("ij,jk->ik", transforms[0], transforms[1])
            zero2two = torch.einsum("ij,jk->ik", zero2one, transforms[2])
            zero2three = torch.einsum("ij,jk->ik", zero2two, transforms[3])
            zero2four = torch.einsum("ij,jk->ik", zero2three, transforms[4])
            zero2five = torch.einsum("ij,jk->ik", zero2four, transforms[5])
            zero2six = torch.einsum("ij,jk->ik", zero2five, transforms[6])
            zero2ee = torch.einsum("ij,jk->ik", zero2six, transforms[7])
            zero2finger = torch.einsum("ij,jk->ik", zero2ee, H_fingers)
            zero2thumb = torch.einsum("ij,jk->ik", zero2ee, H_thumb)
            point_12 = zero2one[:3, 3]
            z_12 = zero2one[:3, 2]
            point_12_l = point_12 + z_12 * 0.07
            point_12_r = point_12 - z_12 * 0.07
            point_0third12 = point_12 / 3.0
            point_0twothird12 = 2.0 * point_12 / 3.0
            point_3 = zero2two[:3, 3]
            x_123 = point_3 - point_12
            point12third3 = point_12 + x_123 * 0.33
            point12twothird3 = point_12 + x_123 * 0.66
            point_4 = zero2three[:3, 3]
            z_4 = zero2three[:3, 2]
            point_4_l = zero2three[:3, 3] + z_4 * 0.05
            point_4_r = zero2three[:3, 3] - z_4 * 0.05
            x_4 = zero2three[:3, 0]
            y_5 = zero2four[:3, 1]
            point_56 = zero2four[:3, 3]
            point_56_l = point_56 + y_5 * 0.07
            point_56_r = point_56 - y_5 * 0.03
            point4quarter56 = point_4 + (point_56 - point_4) * 0.25 - x_4 * 0.05
            point4half56 = point_4 + (point_56 - point_4) * 0.5 - x_4 * 0.05 + y_5 * 0.025
            point4threequarter56 = point_4 + (point_56 - point_4) * 0.75 - x_4 * 0.02 + y_5 * 0.07
            point_7 = zero2six[:3, 3]
            point_f = zero2ee[:3, 3]
            point_finger = zero2finger[:3, 3]
            point_thumb = zero2thumb[:3, 3]
            relevant_points = torch.cat(
                [
                    base_point1,
                    base_point2,
                    point_0third12.unsqueeze(0),
                    point_0twothird12.unsqueeze(0),
                    point_12_l.unsqueeze(0),
                    point_12_r.unsqueeze(0),
                    point12third3.unsqueeze(0),
                    point12twothird3.unsqueeze(0),
                    point_4_l.unsqueeze(0),
                    point_4_r.unsqueeze(0),
                    point4quarter56.unsqueeze(0),
                    point4half56.unsqueeze(0),
                    point4threequarter56.unsqueeze(0),
                    point_56_l.unsqueeze(0),
                    point_56_r.unsqueeze(0),
                    point_7.unsqueeze(0),
                    point_f.unsqueeze(0),
                    point_finger.unsqueeze(0),
                    point_thumb.unsqueeze(0),
                ],
                dim=0,
            )
            return relevant_points - body_points

        return connection_func


class PandaWithPusherJoint2BodyPointsConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "body_points": (NUM_BODY_POINTS, 3)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        alpha = torch.tensor(PANDA_DH_ALPHA, dtype=self.dtype, device=self.device)
        d = torch.tensor(PANDA_DH_D, dtype=self.dtype, device=self.device)
        a = torch.tensor(PANDA_DH_A, dtype=self.dtype, device=self.device)
        base_point1 = torch.zeros(
            (
                1,
                3,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        base_point2 = torch.tensor([[-0.07, 0, 0.05]], dtype=self.dtype, device=self.device)
        append_zero = torch.zeros((1,), dtype=self.dtype, device=self.device)

        H_push_mid = torch.eye(4, dtype=self.dtype, device=self.device)
        H_push_mid[2, 3] = 0.16
        H_push_end = torch.eye(4, dtype=self.dtype, device=self.device)
        H_push_end[2, 3] = 0.33

        def connection_func(body_points, robot_joint_states):
            # point names are defined based on frame names here
            # https://frankaemika.github.io/docs/control_parameters.html
            transforms = dh_params_to_htransform_batch(
                theta=torch.cat([robot_joint_states, append_zero]), a=a, alpha=alpha, d=d
            )
            zero2one = torch.einsum("ij,jk->ik", transforms[0], transforms[1])
            zero2two = torch.einsum("ij,jk->ik", zero2one, transforms[2])
            zero2three = torch.einsum("ij,jk->ik", zero2two, transforms[3])
            zero2four = torch.einsum("ij,jk->ik", zero2three, transforms[4])
            zero2five = torch.einsum("ij,jk->ik", zero2four, transforms[5])
            zero2six = torch.einsum("ij,jk->ik", zero2five, transforms[6])
            zero2ee = torch.einsum("ij,jk->ik", zero2six, transforms[7])
            zero2finger = torch.einsum("ij,jk->ik", zero2ee, H_push_mid)
            zero2thumb = torch.einsum("ij,jk->ik", zero2ee, H_push_end)
            point_12 = zero2one[:3, 3]
            z_12 = zero2one[:3, 2]
            point_12_l = point_12 + z_12 * 0.07
            point_12_r = point_12 - z_12 * 0.07
            point_0third12 = point_12 / 3.0
            point_0twothird12 = 2.0 * point_12 / 3.0
            point_3 = zero2two[:3, 3]
            x_123 = point_3 - point_12
            point12third3 = point_12 + x_123 * 0.33
            point12twothird3 = point_12 + x_123 * 0.66
            point_4 = zero2three[:3, 3]
            z_4 = zero2three[:3, 2]
            point_4_l = zero2three[:3, 3] + z_4 * 0.05
            point_4_r = zero2three[:3, 3] - z_4 * 0.05
            x_4 = zero2three[:3, 0]
            y_5 = zero2four[:3, 1]
            point_56 = zero2four[:3, 3]
            point_56_l = point_56 + y_5 * 0.07
            point_56_r = point_56 - y_5 * 0.03
            point4quarter56 = point_4 + (point_56 - point_4) * 0.25 - x_4 * 0.05
            point4half56 = point_4 + (point_56 - point_4) * 0.5 - x_4 * 0.05 + y_5 * 0.025
            point4threequarter56 = point_4 + (point_56 - point_4) * 0.75 - x_4 * 0.02 + y_5 * 0.07
            point_7 = zero2six[:3, 3]
            point_f = zero2ee[:3, 3]
            point_pusher1 = zero2finger[:3, 3]
            point_pusher2 = zero2thumb[:3, 3]
            relevant_points = torch.cat(
                [
                    base_point1,
                    base_point2,
                    point_0third12.unsqueeze(0),
                    point_0twothird12.unsqueeze(0),
                    point_12_l.unsqueeze(0),
                    point_12_r.unsqueeze(0),
                    point12third3.unsqueeze(0),
                    point12twothird3.unsqueeze(0),
                    point_4_l.unsqueeze(0),
                    point_4_r.unsqueeze(0),
                    point4quarter56.unsqueeze(0),
                    point4half56.unsqueeze(0),
                    point4threequarter56.unsqueeze(0),
                    point_56_l.unsqueeze(0),
                    point_56_r.unsqueeze(0),
                    point_7.unsqueeze(0),
                    point_f.unsqueeze(0),
                    point_pusher1.unsqueeze(0),
                    point_pusher2.unsqueeze(0),
                ],
                dim=0,
            )
            return relevant_points - body_points

        return connection_func


class BodyPoints2SelfCollisionConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"likelihood_self_collision": (1,), "body_points": (NUM_BODY_POINTS, 3)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(body_points, likelihood_self_collision):
            # because of panda kinematics we only need to care about
            # parts above the elbow colliding with parts below it
            dists = torch.norm(body_points.unsqueeze(0) - body_points.unsqueeze(1), dim=2)
            relevant_dists = dists[:6, 11:].reshape(-1)
            min_dist = torch.min(relevant_dists).reshape((1,))
            if min_dist > 0.075:
                curr_likelihood = torch.zeros_like(min_dist)
            else:
                curr_likelihood = torch.erf((0.05 - min_dist) * 10) / 2.0 + 0.5
            return curr_likelihood - likelihood_self_collision

        return connection_func


class JointsProprioConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"robot_joint_states": (7,), "robot_joint_states_meas": (7,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(robot_joint_states, robot_joint_states_meas):
            return robot_joint_states_meas - robot_joint_states

        return connection_func


class EEFTWithHandConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "ee_force_measured": (6,), "ee_force_external": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        R_sensor = torch.tensor(FT_ROTATION_RH3, dtype=self.dtype, device=self.device)
        H_com = torch.eye(4, dtype=self.dtype, device=self.device)
        H_com[:3, 3] = torch.tensor(FT_COM_RH3, dtype=self.dtype, device=self.device)
        expected_force = torch.tensor([0.0, 0.0, -GRAVITY_ACC * EE_MASS_RH3], dtype=self.dtype, device=self.device)

        def connection_func(pose_ee, ee_force_measured, ee_force_external):
            # the measured and the current pose should be the same
            H_ee = exponential_map_se3(pose_ee)
            rotated_meas_force = torch.einsum("ij,jk,k->i", H_ee[:3, :3], R_sensor, ee_force_measured[:3])
            external_force_meas = rotated_meas_force - expected_force
            return ee_force_external - external_force_meas

        return connection_func


class EEFTWithPusherConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pose_ee": (6,), "uncertainty_ee": (6, 6), "ee_force_measured": (6,), "ee_force_external": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        R_sensor = torch.tensor(FT_ROTATION_PUSHER, dtype=self.dtype, device=self.device)
        H_com = torch.eye(4, dtype=self.dtype, device=self.device)
        H_com[:3, 3] = torch.tensor(FT_COM_PUSHER, dtype=self.dtype, device=self.device)
        expected_force = torch.tensor([0.0, 0.0, -GRAVITY_ACC * EE_MASS_PUSHER], dtype=self.dtype, device=self.device)

        def connection_func(pose_ee, ee_force_measured, ee_force_external):
            # the measured and the current pose should be the same
            H_ee = exponential_map_se3(pose_ee)
            rotated_meas_force = torch.einsum("ij,jk,k->i", H_ee[:3, :3], R_sensor, ee_force_measured[:3])
            external_force_meas = rotated_meas_force - expected_force
            return ee_force_external - external_force_meas

        return connection_func


class JointLimitLikelihoodConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        interval_in=None,
        interval_out=None,
    ):
        self._interval_in = interval_in
        self._interval_out = interval_out
        super().__init__(
            name,
            {"robot_joint_states": (7,), "likelihood_joint_at_limit": (7,), "joint_close_to_limit": (7,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        joint_limits_low = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=self.dtype, device=self.device
        )
        joint_limits_high = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=self.dtype, device=self.device
        )
        interval_in = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15], dtype=self.dtype, device=self.device)
        interval_out = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.25, 0.3, 0.3], dtype=self.dtype, device=self.device)
        if self._interval_in is not None:
            interval_in = torch.as_tensor(self._interval_in, dtype=self.dtype, device=self.device)
        if self._interval_out is not None:
            interval_out = torch.as_tensor(self._interval_out, dtype=self.dtype, device=self.device)
        if interval_in.shape != (7,) or interval_out.shape != (7,):
            raise ValueError(
                "interval_in/out must have 7 entries, got %s / %s"
                % (tuple(interval_in.shape), tuple(interval_out.shape))
            )
        if bool((interval_out <= interval_in).any()):
            raise ValueError(
                "interval_out must exceed interval_in on every joint: %s vs %s"
                % (interval_out.tolist(), interval_in.tolist())
            )

        def connection_func(robot_joint_states, likelihood_joint_at_limit, joint_close_to_limit):
            joint_limits_low_cost = joint_limits_low - robot_joint_states
            joint_limits_high_cost = robot_joint_states - joint_limits_high
            in_interval_low = joint_limits_low_cost > -interval_in
            in_interval_high = joint_limits_high_cost > -interval_in
            out_interval_low = joint_limits_low_cost > -interval_out
            out_interval_high = joint_limits_high_cost > -interval_out
            likelihood = gradient_preserving_clipping(
                torch.abs(
                    (joint_limits_low_cost / interval_in) * in_interval_low.detach()
                    + (joint_limits_high_cost / interval_in) * in_interval_high.detach()
                    + (joint_limits_low_cost / interval_out) * out_interval_low.detach() * joint_close_to_limit
                    + (joint_limits_high_cost / interval_out) * out_interval_high.detach() * joint_close_to_limit
                ),
                0.0,
                1.0,
            )
            return likelihood_joint_at_limit - likelihood

        return connection_func


class JointRangeUtilizationConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        nominal=None,
    ):
        self._nominal = nominal
        super().__init__(
            name,
            {"robot_joint_states": (7,), "joint_range_utilization": (7,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        joint_limits_low = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=self.dtype, device=self.device
        )
        joint_limits_high = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=self.dtype, device=self.device
        )
        if self._nominal is None:
            nominal = 0.5 * (joint_limits_low + joint_limits_high)
        else:
            nominal = torch.as_tensor(self._nominal, dtype=self.dtype, device=self.device)
            if nominal.shape != joint_limits_low.shape:
                raise ValueError("nominal must have 7 entries, got %s" % (tuple(nominal.shape),))
            if bool(((nominal < joint_limits_low) | (nominal > joint_limits_high)).any()):
                raise ValueError("nominal outside the joint limits: %s" % (nominal.tolist(),))
        half_range = 0.5 * (joint_limits_high - joint_limits_low)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.3, 1.0, 0.0], dtype=self.dtype, device=self.device)

        def connection_func(joint_range_utilization, robot_joint_states):
            utilization = weights * ((robot_joint_states - nominal) / half_range) ** 2
            return joint_range_utilization - utilization

        return connection_func
