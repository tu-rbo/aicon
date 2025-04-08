from typing import Union, Dict

import rospy
import torch
from geometry_msgs.msg import WrenchStamped

from aicon.base_classes.components import SensorComponent
from aicon.base_classes.connections import ActiveInterconnection


class ForceTorqueSensor(SensorComponent):
    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], quantity_name: str, ros_topic:str, bias : torch.Tensor,
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False):
        self.quantity_name = quantity_name
        super().__init__(name, connections, dtype, device, mockbuild)
        if mockbuild:
            return
        self.bias = bias

        self.subscriber = rospy.Subscriber(ros_topic, WrenchStamped, self._callback, queue_size=1)
        self._internal_timestep = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_last_image = torch.tensor(-1, dtype=self.dtype, device=self.device)

    def obtain_measurements(self) -> bool:
        if self._internal_timestep < 0:
            return False
        self.timestamp = self._internal_timestep
        self.quantities[self.quantity_name] = self._internal_last_ft - self.bias
        return True

    def initial_definitions(self) -> None:
        self.quantities[self.quantity_name] = torch.empty(6, dtype=self.dtype, device=self.device)

    def _callback(self, msg):
        self._internal_last_ft = torch.tensor([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                               msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                                              dtype=self.dtype, device=self.device)
        self._internal_timestep = torch.tensor(msg.header.stamp.to_sec(), dtype=self.dtype, device=self.device)