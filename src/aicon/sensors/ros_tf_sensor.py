from typing import Dict, Union

import rospy
import tf2_ros
import torch

from aicon.math.util_3d import log_map_se3
from aicon.middleware.util_ros import get_tf_frame
from aicon.base_classes.components import SensorComponent
from aicon.base_classes.connections import ActiveInterconnection


class ROSTFSensor(SensorComponent):

    def obtain_measurements(self) -> bool:
        transform, transform_time = get_tf_frame(buffer=self._tfBuffer, source_frame=self.source_frame, target_frame=self.target_frame)
        transform_time = torch.tensor(transform_time.to_sec(), dtype=self.dtype, device=self.device)
        if self.timestamp >= transform_time > 1.0:
            return False
        self.timestamp = transform_time
        self.quantities[self.quantity_name] = log_map_se3(transform)
        return True

    def initial_definitions(self):
        self.quantities[self.quantity_name] = torch.empty(6, dtype=self.dtype, device=self.device)

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], source_frame:str, target_frame:str, quantity_name:str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        self.quantity_name = quantity_name
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)
        if mockbuild:
            return 

        self._tfBuffer = tf2_ros.Buffer()
        self._tflistener = tf2_ros.TransformListener(self._tfBuffer)
        rospy.sleep(1)
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.frame_id = self.source_frame
