from abc import ABCMeta
from typing import Dict, Callable, Union

import rospy
import torch
from hidden_hand_interface import syn_execution

from aicon.base_classes.components import ActionComponent
from aicon.base_classes.connections import ActiveInterconnection


class RBOHandSynergyAction(ActionComponent, metaclass=ABCMeta):

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals : Union[None,Dict[str, Callable]] = None,
                 dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device(),
                 mockbuild : bool = False, initially_grasped : bool = False):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return
        self.hand_controller = rospy.ServiceProxy('synergy_execution_srv', syn_execution)
        self.next_controller_cmd = None
        if initially_grasped:
            self.next_controller_cmd = "close_base_tip_sequentially_movement_fast_plus_thumb"

    def _start(self, reset=True):
        if reset:
            self.hand_controller("reset")

    def _stop(self, reset=True):
        if reset:
            self.hand_controller("reset")

    def send_action_values(self) -> None:
        if self.next_controller_cmd is not None:
            self.hand_controller(self.next_controller_cmd)
            self.next_controller_cmd = None


