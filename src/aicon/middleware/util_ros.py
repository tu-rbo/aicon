import os
import subprocess
import time
from typing import List

import numpy as np
import torch

import aicon
from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.derivatives import QuantityRelatedDerivativeBlock, DerivativeDict
from aicon.math.util_3d import rotation_matrix_from_quaternion

"""
This was based on ROS1 code.
The public version currently does not support ROS2 yet. 
"""

def publish_feature_points_with_covariance_pointcloud(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    pass


def get_ros_point(position):
    pass

def get_ros_vector(vector):
    pass

def get_ros_quaternion(ros_ordered_quat):
    pass


def rgbd_from_rosbag(bagfilename, index=0):
    pass

def publish_numpy_as_image(publisher, img, encoding="8UC4"):
    pass

def remove_ros_node_args_from_list(l):
    pass


def torch_tensor_to_ros_multiarray(t, msg):
    pass


def ros_multiarray_to_torch_tensor(msg, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
    pass


def derivative_msg_to_derivative_dict(msg, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
    pass


def derivative_block_to_derivative_msg(derivative_block: QuantityRelatedDerivativeBlock, msg):
    pass


def ros_transform_to_torch_Htransform(transform):
    pass


def get_tf_frame(buffer, source_frame:str, target_frame:str, timeout=0.2):
    pass


def wait_for_tf_frame(source_frame:str, target_frame:str, timeout:float):
    pass


def start_using_ros_sim_time():
    pass


class ROSWrappingConnection(ActiveInterconnection):

    def define_implicit_connection_function(self):
        pass


class ROSBagRecorder:

    def __init__(self, topic_name_space : str = None, topics : List[str] = None, prefix : str = None):
        pass

    def close(self):
        pass

    def terminate(self):
        pass
