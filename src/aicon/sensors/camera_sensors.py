import traceback
from typing import Dict, Union

import message_filters
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from aicon.base_classes.components import SensorComponent
from aicon.base_classes.connections import ActiveInterconnection


def preprocess_img_depth(depth_img, depth_divisor, dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device()):
    ## Process Depth
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(depth_img, desired_encoding='passthrough')
    img_torch = torch.tensor(img.astype(float) / depth_divisor, dtype=dtype, device=device)
    return img_torch


def preprocess_img_color(color_img, dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device()):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(color_img, desired_encoding='passthrough')
    return torch.tensor(img.copy(), dtype=dtype, device=device)


def get_camtopic_name(camera_type:str, output_type:str="rgb"):
    match camera_type, output_type:
        case "realsense", "rgb":
            return "/camera/color/"
        case "realsense", "depth":
            return "/camera/depth/"
        case "realsense", "depth_aligned":
            return "/camera/aligned_depth_to_color/"
        case "asus", "rgb":
            return "/camera/rgb/"
        case "asus", "depth_aligned":
            return "/camera/depth_registered/"
        case _:
            raise LookupError("No known camera topics for ("+camera_type + ", " +output_type, ")")


class ROSCameraRGBSensor(SensorComponent):

    def obtain_measurements(self) -> bool:
        if self._internal_timestep <= 0:
            return False
        self.quantities["rgb_image_" + self.name] = torch.clone(self._internal_last_image)
        self.quantities["cam_D_" + self.name] = torch.clone(self._internal_D)
        self.quantities["cam_K_" + self.name] = torch.clone(self._internal_K)
        self.quantities["cam_R_" + self.name] = torch.clone(self._internal_R)
        self.quantities["cam_P_" + self.name] = torch.clone(self._internal_P)
        self.timestamp = self._internal_timestep
        return True

    def initial_definitions(self):
        self.quantities["rgb_image_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_D_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_K_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_R_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)
        self.quantities["cam_P_" + self.name] = torch.empty(dtype=self.dtype, device=self.device)

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], ros_topic_base:str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)
        if mockbuild:
            return 

        self.frame_id = "unknown"

        color_image_sub = message_filters.Subscriber(ros_topic_base+"image_raw", Image)
        color_info_sub = message_filters.Subscriber(ros_topic_base+"camera_info", CameraInfo)
        self._internal_timestep = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_last_image = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_D = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_K = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_R = torch.tensor(-1, dtype=self.dtype, device=self.device)
        self._internal_P = torch.tensor(-1, dtype=self.dtype, device=self.device)

        # ts = message_filters.TimeSynchronizer([color_image_sub, color_info_sub, depth_image_sub, depth_info_sub], 10)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_image_sub, color_info_sub], 100, slop=0.15)
        self.ts.registerCallback(self.callback_synchronized)

    def callback_synchronized(self, color_img, color_info):
        try:
            self._internal_D = torch.tensor(color_info.D, dtype=self.dtype, device=self.device).reshape(5)
            self._internal_K = torch.tensor(color_info.K, dtype=self.dtype, device=self.device).reshape(3, 3)
            self._internal_R = torch.tensor(color_info.R, dtype=self.dtype, device=self.device).reshape(3, 3)
            self._internal_P = torch.tensor(color_info.P, dtype=self.dtype, device=self.device).reshape(3, 4)
            self._internal_last_image = preprocess_img_color(color_img, dtype=self.dtype, device=self.device)
            self._internal_timestep = torch.tensor(color_img.header.stamp.to_sec(), dtype=self.dtype, device=self.device)
            self.frame_id = color_img.header.frame_id
        except Exception as e:
            print("Exception during Camera Callback:")
            traceback.print_exc()
