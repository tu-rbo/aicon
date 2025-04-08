import os
import subprocess
import time
from typing import List

import numpy as np
import rosbag
import rospy
import tf2_ros
import torch
from cv_bridge import CvBridge
from aicon_msgs.msg import DerivativesMsg, StringListMsg, TensorListMsg
from geometry_msgs.msg import Point, Vector3, Quaternion
from sensor_msgs.msg import PointField
from sensor_msgs.point_cloud2 import create_cloud
from std_msgs.msg import MultiArrayDimension, Float64MultiArray

import aicon
from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.derivatives import QuantityRelatedDerivativeBlock, DerivativeDict
from aicon.math.util_3d import rotation_matrix_from_quaternion

util_tf_buffer = None
util_tf_listener = None

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


    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('label', 16, PointField.UINT32, 1),
          PointField('covariance', 20, PointField.FLOAT32, 9), ]
    return create_cloud(header, fields, points)


def get_ros_point(position):
    pt = Point()
    pt.x = position[0].item()
    pt.y = position[1].item()
    pt.z = position[2].item()
    return pt

def get_ros_vector(vector):
    vec = Vector3()
    vec.x = vector[0].item()
    vec.y = vector[1].item()
    vec.z = vector[2].item()
    return vec

def get_ros_quaternion(ros_ordered_quat):
    q = Quaternion()
    q.x = ros_ordered_quat[0].item()
    q.y = ros_ordered_quat[1].item()
    q.z = ros_ordered_quat[2].item()
    q.w = ros_ordered_quat[3].item()
    return q


def rgbd_from_rosbag(bagfilename, index=0):
    print("Reading rgbd from:", bagfilename)
    bag = rosbag.Bag(bagfilename)
    index_of_target_img = 0

    for i_img,(topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/color/image_raw'])):
        if i_img == index_of_target_img:
            img_color = aicon.sensors.camera_sensors.preprocess_img_color(msg)

    for i_img,(topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw'])):
        if i_img == index_of_target_img:
            img_depth = aicon.sensors.camera_sensors.preprocess_img_depth(msg)

    for i_img,(topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/color/camera_info'])):
        if i_img == index_of_target_img:
            caminfo_color = msg

    for i_img,(topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/aligned_depth_to_color/camera_info'])):
        if i_img == index_of_target_img:
            caminfo_depth = msg

    return img_color, img_depth, caminfo_color, caminfo_depth

def publish_numpy_as_image(publisher, img, encoding="8UC4"):
    assert type(img) == np.ndarray
    assert img.dtype == np.uint8

    bridge = CvBridge()

    image_message = bridge.cv2_to_imgmsg(img, encoding=encoding)
    publisher.publish(image_message)

def remove_ros_node_args_from_list(l):
    new_list = []
    for s in l:
        if s.startswith("__name:="):
            continue
        if s.startswith("__log:="):
            continue
        new_list.append(s)
    return new_list


def torch_tensor_to_ros_multiarray(t, msg):
    msg.data = t.view(-1).detach().cpu().tolist()

    msg.layout.data_offset = 0
    msg.layout.dim = []
    for i in range(len(t.shape)):
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[i].label = str(i)
        msg.layout.dim[i].size = t.shape[i]
        msg.layout.dim[i].stride = int(np.prod(t.shape[(i+1):]))
    return msg


def ros_multiarray_to_torch_tensor(msg, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
    t = torch.tensor(msg.data, dtype=torch.double, device=device).type(dtype)
    shape = [msg.layout.dim[i].size for i in range(len(msg.layout.dim))]
    return t.reshape(shape)


def derivative_msg_to_derivative_dict(msg : DerivativesMsg, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
    if len(msg.quantity_traces) == 0:
        return DerivativeDict()
    new_block = QuantityRelatedDerivativeBlock(msg.name)
    new_block.derivatives_tensor = ros_multiarray_to_torch_tensor(msg.derivatives_tensor, dtype=dtype, device=device)
    for i, m in enumerate(msg.quantity_traces):
        new_block.quantity_traces.append([str(s) for s in m.string_list])
        new_block.timestamps.append([ros_multiarray_to_torch_tensor(stamp, dtype=dtype, device=device) for stamp in msg.timestamps[i].tensor_list])
    return DerivativeDict({msg.name: new_block})


def derivative_block_to_derivative_msg(derivative_block: QuantityRelatedDerivativeBlock, msg : DerivativesMsg):
    msg.derivatives_tensor = torch_tensor_to_ros_multiarray(derivative_block.derivatives_tensor, msg.derivatives_tensor)
    msg.name = derivative_block.related_quantity
    for i, qt in enumerate(derivative_block.quantity_traces):
        qt_msg = StringListMsg()
        ts_msg = TensorListMsg()
        for s in qt:
            qt_msg.string_list.append(s)
        for t in derivative_block.timestamps[i]:
            new_t = Float64MultiArray()
            new_t = torch_tensor_to_ros_multiarray(t, new_t)
            ts_msg.tensor_list.append(new_t)
        msg.quantity_traces.append(qt_msg)
        msg.timestamps.append(ts_msg)
    return msg


def ros_transform_to_torch_Htransform(transform):
    translation = torch.tensor(
        [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    rotation = rotation_matrix_from_quaternion(torch.tensor([transform.transform.rotation.w,
                                                             transform.transform.rotation.x,
                                                             transform.transform.rotation.y,
                                                             transform.transform.rotation.z]),
                                               is_batch=False)
    H = torch.eye(4)
    H[:3, :3] = rotation
    H[:3, 3] = translation
    return H


def get_tf_frame(buffer, source_frame:str, target_frame:str, timeout=0.2):
    try:
        transform_time = buffer.get_latest_common_time(source_frame, target_frame)
        transform = buffer.lookup_transform(source_frame, target_frame,
                                                    transform_time, timeout=rospy.Duration.from_sec(timeout))
    except (tf2_ros.ExtrapolationException, tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.TransformException) as e:
        print("TF Frames for Camera Pose couldn't be looked up... " + str(e))
        raise AssertionError("No Cam Pose could be looked up right now!")
    return ros_transform_to_torch_Htransform(transform), transform_time


def wait_for_tf_frame(source_frame:str, target_frame:str, timeout:float):
    global util_tf_buffer, util_tf_listener
    if util_tf_buffer is None:
        util_tf_buffer = tf2_ros.Buffer()
        util_tf_listener = tf2_ros.TransformListener(util_tf_buffer)
        rospy.sleep(0.5)
    transform, _ = get_tf_frame(util_tf_buffer, source_frame, target_frame, timeout=timeout)
    return transform


def start_using_ros_sim_time():
    rospy.set_param("use_sim_time", True)
    time.sleep(0.5)


class ROSWrappingConnection(ActiveInterconnection):

    def define_implicit_connection_function(self):

        def empty_c_func():
            return torch.zeros(1, dtype=self.dtype, device=self.device)

        return empty_c_func


class ROSBagRecorder:

    def __init__(self, topic_name_space : str = None, topics : List[str] = None, prefix : str = None):
        call_list = ["rosbag", "record"]
        if topics is not None:
            call_list.extend(topics)
        if topic_name_space is not None:
            call_list.extend(["--regex", topic_name_space + "/\"(.*)\""])
        if prefix is not None:
            call_list.extend(["--output-prefix", prefix])
        call_list.append("--lz4")
        call_str = ""
        for s in call_list:
            call_str += str(s) + " "
        actual_call_list = ['/bin/bash', '-c', 'source /home/vito/.rosrc && '+call_str]
        self.rosbag_process = subprocess.Popen(actual_call_list, env=os.environ.copy(), stdin=subprocess.PIPE)
        rospy.on_shutdown(self.close)

    def close(self):
        self.rosbag_process.send_signal(subprocess.signal.SIGINT)

    def terminate(self):
        self.rosbag_process.send_signal(subprocess.signal.SIGINT)
