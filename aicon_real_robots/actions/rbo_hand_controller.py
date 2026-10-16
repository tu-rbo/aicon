"""
Real-World RH3 Action Component

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from abc import ABCMeta
from typing import Dict, Callable, Union, Tuple, Type

import rclpy
import torch
import time
from aicon_base.components import ActionComponent
from aicon_base.connections import ActiveInterconnection
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace
from std_msgs.msg import String

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile_hand_cmd = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10
)


class RBOHandSynergyAction(ActionComponent, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device = torch.get_default_device(),
        mockbuild: bool = False,
        initially_grasped: bool = False,
    ):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return
        self.node = rclpy.create_node(name + "_action_publisher", namespace=default_ros_namespace)
        self.hand_state_publisher = self.node.create_publisher(String, "/aicon/rh3_command", qos_profile_hand_cmd)
        self.next_controller_cmd = None
        if initially_grasped:
            self.next_controller_cmd = "close"

    def _start(self, reset=True):
        if reset:
            self.hand_state_publisher.publish(String(data="reset"))
            time.sleep(1.5)
        self.hand_state_publisher.publish(String(data="open"))

    def _stop(self, reset=True):
        if reset:
            self.hand_state_publisher.publish(String(data="reset"))

    def send_action_values(self) -> None:
        if self.next_controller_cmd is not None:
            self.hand_state_publisher.publish(String(data=self.next_controller_cmd))
            self.next_controller_cmd = None


class SimpleHandSynergyAction(RBOHandSynergyAction):
    state_dim = 1
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "hand_synergy_activation": (1,),
    }

    def determine_new_actions(self):
        steepest_grad, timestamps, trace = self.get_steepest_gradient("hand_synergy_activation")
        print("Steepest hand grad", steepest_grad, self.quantities["hand_synergy_activation"])
        if steepest_grad < -0.00000001:
            if self.quantities["hand_synergy_activation"] != 0.99:
                print("closing hand")
                self.next_controller_cmd = "close"
                self.quantities["hand_synergy_activation"] = 0.99
        elif steepest_grad > 0.000000000000000001:
            if self.quantities["hand_synergy_activation"] != 0.01:
                print("opening hand")
                self.next_controller_cmd = "open"
                self.quantities["hand_synergy_activation"] = 0.01

    def initialize_quantities(self) -> bool:
        self.quantities["hand_synergy_activation"] = (
            torch.ones(self.state_dim, dtype=self.dtype, device=self.device) * 0.01
        )
        return True
