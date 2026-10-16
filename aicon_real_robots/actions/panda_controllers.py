"""
Real-World Panda Controller Action Component

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Union, Callable, Tuple

import torch
import rclpy
from sensor_msgs.msg import JointState

from aicon_base.components import ActionComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.math.se3_and_SE3 import exponential_map_se3
from aicon_real_robots.sensors.ros_tf_sensor import ROSTFSensor
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace


from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile_panda_cmd = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10
)


class JointVeloPandaControlViaROSCtrl(ActionComponent):
    state_dim = 7
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"action_velo_joints": (7,)}

    def _start(self) -> None:
        self.pose_sensor.update(self.timestamp)
        self.start_pose = exponential_map_se3(self.pose_sensor.quantities["pose_measured_ee_internal"])
        new_action = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["action_velo_joints"] = new_action

    def _stop(self) -> None:
        msg = JointState()
        for i in range(self.state_dim):
            msg.position.append(0.0)
        self.joint_velo_publisher.publish(msg)

    @staticmethod
    def _scale_into_limits(action: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
        """Scale a group of joint velocities into their limits without changing the direction."""
        violation = torch.max(torch.abs(action) / limits)
        if violation > 1.0:
            action = action / violation
        return action

    def determine_new_actions(self) -> None:
        last_action = self.quantities["action_velo_joints"]
        steepest_grad = self.get_steepest_plus_projected(
            "action_velo_joints", time_threshold=self.timestamp - torch.ones_like(self.timestamp) * 1.0
        )
        gain = 0.2
        velo_gain = 0.05
        v_max = 0.1
        # enforced velocity limits: the Panda's joint velocity limits times the per-joint safety factor vmax_scale
        joint_vmax = (
            torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=self.dtype, device=self.device)
            * self.vmax_scale
        )

        grad_unit = steepest_grad / (torch.norm(steepest_grad) + 1e-9)

        # Leaky integration of the normalised gradient; the leak keeps directions the gradient does not constrain
        # from drifting.
        proposed_action = last_action - gain * v_max * grad_unit * self.contribution_factors - velo_gain * last_action

        alpha = 0.99
        new_action = torch.nan_to_num((1 - alpha) * last_action + alpha * proposed_action)

        # Scale into the limits uniformly (keeps the direction). The last joint (camera roll on the pushing setup) is
        # scaled separately, so it does not take budget from the other joints.
        new_action_limited = torch.cat(
            [
                self._scale_into_limits(new_action[:6], joint_vmax[:6]),
                self._scale_into_limits(new_action[6:], joint_vmax[6:]),
            ]
        )
        self.quantities["action_velo_joints"] = new_action_limited

    def send_action_values(self) -> None:
        msg = JointState()
        for i in range(self.state_dim):
            msg.position.append(self.quantities["action_velo_joints"][i])
        self.joint_velo_publisher.publish(msg)

    def initialize_quantities(self) -> bool:
        new_action = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["action_velo_joints"] = new_action
        return True

    def __init__(
        self,
        name: str,
        connections: Dict[str, ActiveInterconnection],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.get_default_device(),
        mockbuild: bool = False,
        gain: float = 0.1,
        v_max: float = 0.125,
        contribution_factors=None,
        vmax_scale=0.03,
    ):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return

        self.v_max = v_max
        self.gain = gain
        # per-joint gain factors and velocity limit safety factors (7 entries; vmax_scale may be a single float)
        if contribution_factors is None:
            contribution_factors = [1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.7]
        self.contribution_factors = torch.as_tensor(contribution_factors, dtype=self.dtype, device=self.device)
        self.vmax_scale = torch.as_tensor(vmax_scale, dtype=self.dtype, device=self.device).expand(7).clone()
        if self.contribution_factors.shape != (7,):
            raise ValueError(
                "contribution_factors must have 7 entries, got %s" % (tuple(self.contribution_factors.shape),)
            )
        self.pose_sensor = ROSTFSensor(
            "InternalEEPoseSensor",
            connections={},
            source_frame="panda_link0",
            target_frame="panda_link8",
            device=device,
            quantity_name="pose_measured_ee_internal",
            dtype=dtype,
            mockbuild=mockbuild,
        )
        self.node = rclpy.create_node(name + "_action_publisher", namespace=default_ros_namespace)
        self.joint_velo_publisher = self.node.create_publisher(
            JointState, "/aicon/panda_command", qos_profile_panda_cmd
        )
