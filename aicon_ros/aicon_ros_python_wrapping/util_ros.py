"""
Utility functions to convert things from AICON/torch to ROS and back

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import atexit
import datetime
import os
import random
import shutil
import signal
import subprocess
import time
from typing import List

import numpy as np
import rclpy
import tf2_ros
import torch
from tensordict import TensorDict
from cv_bridge import CvBridge
from aicon_ros.msg import DerivativesMsg, StringListMsg, TensorListMsg, ListOfStringListsMsg, ListOfListOfStringListsMsg
from geometry_msgs.msg import Point, Vector3, Quaternion
from std_msgs.msg import MultiArrayDimension, Float64MultiArray

from aicon_base.connections import ActiveInterconnection
from aicon_base.derivatives import QuantityRelatedDerivativeBlock, DerivativeDict
from aicon_base.math.se3_and_SE3 import rotation_matrix_from_quaternion
from aicon_base.timestamps import TIME_DTYPE, as_timestamp

util_tf_buffer = None
util_tf_listener = None
util_tf_node = None


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


def publish_numpy_as_image(publisher, img, encoding="8UC4"):
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8

    bridge = CvBridge()

    image_message = bridge.cv2_to_imgmsg(img, encoding=encoding)
    publisher.publish(image_message)


def remove_ros_node_args_from_list(args):
    new_list = []
    for s in args:
        if s.startswith("__name:="):
            continue
        if s.startswith("__log:="):
            continue
        new_list.append(s)
    return new_list


def torch_tensor_to_ros_multiarray(t, msg):
    msg.data = torch.reshape(t.detach().cpu().to(torch.double), (-1,)).tolist()

    msg.layout.data_offset = 0
    msg.layout.dim = []
    for i in range(len(t.shape)):
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[i].label = str(i)
        msg.layout.dim[i].size = t.shape[i]
        msg.layout.dim[i].stride = int(np.prod(t.shape[(i + 1) :]))
    return msg


def ros_multiarray_to_torch_tensor(msg, dtype=torch.get_default_dtype(), device=torch.get_default_device()):
    t = torch.tensor(msg.data, dtype=torch.double, device=device).type(dtype)
    shape = [msg.layout.dim[i].size for i in range(len(msg.layout.dim))]
    return t.reshape(shape)


def derivative_msg_to_derivative_dict(
    msg: DerivativesMsg, dtype=torch.get_default_dtype(), device=torch.get_default_device()
):
    if len(msg.quantity_traces) == 0:
        return DerivativeDict()
    new_block = QuantityRelatedDerivativeBlock(msg.name)
    new_block.derivatives_tensor = ros_multiarray_to_torch_tensor(msg.derivatives_tensor, dtype=dtype, device=device)
    new_block.last_partial_derivatives = TensorDict({}, device=device, batch_size=())
    for k, v in zip(msg.partial_derivative_keys, msg.partial_derivative_values):
        new_block.last_partial_derivatives[k] = ros_multiarray_to_torch_tensor(v, dtype=dtype, device=device)
    for i, m in enumerate(msg.quantity_traces):
        new_block.quantity_traces.append([str(s) for s in m.string_list])
        new_block.mixed_in_quantity_traces.append(
            [
                [[str(s) for s in trace.string_list] for trace in l_quantity_traces.mixed_in_at_node]
                for l_quantity_traces in msg.mixed_in_quantity_traces[i].mixed_in_quantity_traces_for_gradient
            ]
        )
        # timestamps come packed into one array per path; one array per trace entry is accepted too
        stamp_arrays = msg.timestamps[i].tensor_list
        if len(stamp_arrays) == 1:
            flat = ros_multiarray_to_torch_tensor(stamp_arrays[0], dtype=TIME_DTYPE, device="cpu")
            new_block.timestamps.append(list(flat.reshape(-1).split(1)))
        else:
            new_block.timestamps.append(
                [ros_multiarray_to_torch_tensor(stamp, dtype=TIME_DTYPE, device="cpu") for stamp in stamp_arrays]
            )
    return DerivativeDict({msg.name: new_block})


def derivative_block_to_derivative_msg(derivative_block: QuantityRelatedDerivativeBlock, msg: DerivativesMsg):
    msg.derivatives_tensor = torch_tensor_to_ros_multiarray(derivative_block.derivatives_tensor, msg.derivatives_tensor)
    msg.name = derivative_block.related_quantity
    for k, v in derivative_block.last_partial_derivatives.items():
        msg.partial_derivative_keys.append(k)
        new_t = Float64MultiArray()
        new_t = torch_tensor_to_ros_multiarray(v, new_t)
        msg.partial_derivative_values.append(new_t)
    for i, qt in enumerate(derivative_block.quantity_traces):
        qt_msg = StringListMsg()
        ts_msg = TensorListMsg()
        mixed_in_msg = ListOfListOfStringListsMsg()
        for s in qt:
            qt_msg.string_list.append(s)
        # all timestamps of a path packed into one array (much faster to encode than one array per entry)
        stamps = derivative_block.timestamps[i]
        if stamps:
            packed = Float64MultiArray()
            ts_msg.tensor_list.append(
                torch_tensor_to_ros_multiarray(torch.cat([s if s.dim() else s.unsqueeze(0) for s in stamps]), packed)
            )
        for m_ij in derivative_block.mixed_in_quantity_traces[i]:
            q_mixins_msg = ListOfStringListsMsg()
            for trace in m_ij:
                single_trace_msg = StringListMsg()
                for s in trace:
                    single_trace_msg.string_list.append(s)
                q_mixins_msg.mixed_in_at_node.append(single_trace_msg)
            mixed_in_msg.mixed_in_quantity_traces_for_gradient.append(q_mixins_msg)
        msg.quantity_traces.append(qt_msg)
        msg.timestamps.append(ts_msg)
        msg.mixed_in_quantity_traces.append(mixed_in_msg)
    return msg


def ros_transform_to_torch_Htransform(transform):
    translation = torch.tensor(
        [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
    )
    rotation = rotation_matrix_from_quaternion(
        torch.tensor(
            [
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
            ]
        ),
        is_batch=False,
    )
    H = torch.eye(4)
    H[:3, :3] = rotation
    H[:3, 3] = translation
    return H


def get_tf_frame(buffer, node, source_frame: str, target_frame: str, timeout=0.2):
    try:
        if node is not None:
            # need to handle spinning myself
            t_start = time.time()
            while not buffer.can_transform(source_frame, target_frame, rclpy.time.Time()):
                try:
                    rclpy.spin_once(node, timeout_sec=0.1)
                except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
                    raise AssertionError("TF Frames couldn't be looked up...")
                if time.time() - t_start > timeout:
                    break
        transform_time = buffer.get_latest_common_time(source_frame, target_frame)
        transform = buffer.lookup_transform(
            source_frame, target_frame, transform_time, timeout=rclpy.duration.Duration(seconds=timeout)
        )
    except (
        tf2_ros.ExtrapolationException,
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.TransformException,
    ) as e:
        print("TF Frames couldn't be looked up... " + str(e))
        raise AssertionError("TF Frames couldn't be looked up...")
    return ros_transform_to_torch_Htransform(transform), transform_time


def wait_for_tf_frame(source_frame: str, target_frame: str, timeout: float):
    global util_tf_buffer, util_tf_listener, util_tf_node
    if util_tf_buffer is None:
        util_tf_node = rclpy.create_node("util_tf_listener_" + str(random.randint(0, 100000)))
        util_tf_buffer = tf2_ros.Buffer()
        util_tf_listener = tf2_ros.TransformListener(util_tf_buffer, util_tf_node)
    transform, _ = get_tf_frame(util_tf_buffer, util_tf_node, source_frame, target_frame, timeout=timeout)
    return transform


def start_using_ros_sim_time():
    raise NotImplementedError()


class ROSWrappingConnection(ActiveInterconnection):
    def define_implicit_connection_function(self):
        def empty_c_func():
            return torch.zeros(1, dtype=self.dtype, device=self.device)

        return empty_c_func


def time_float_to_ros(t):
    t_sec = int(t)
    t_nsec = int((t - t_sec) * 1e9)
    ros_time = rclpy.time.Time(seconds=t_sec, nanoseconds=t_nsec)
    return ros_time


def time_ros_to_float(ros_time: rclpy.time.Time):
    t_sec, t_nsec = ros_time.seconds_nanoseconds()
    t = float(t_sec) + float(t_nsec) / 1e9
    return t


def time_ros_to_torch(ros_time: rclpy.time.Time) -> torch.Tensor:
    """A ROS time as an AICON timestamp (float64 CPU tensor, see aicon_base.timestamps)."""
    return as_timestamp(time_ros_to_float(ros_time))


class ROSBagRecorder:
    """
    Records a rosbag2 in a background 'ros2 bag record' process.

    The recorder is started in its own session, so hitting ctrl+C in the terminal does not reach
    it directly. Instead close() (also registered via atexit) sends a single SIGINT, which makes
    rosbag2 flush and close the bag properly. That way the bag stays open until everything else
    has shut down, and it is closed exactly once.
    """

    def __init__(
        self,
        topic_name_space: str = None,
        topics: List[str] = None,
        prefix: str = None,
        exclude_regex: str = None,
        record_all: bool = False,
        timestamp_suffix: bool = True,
    ):
        if shutil.which("ros2") is None:
            raise RuntimeError("Cannot record a rosbag: 'ros2' is not on the PATH, source your ROS 2 workspace first.")

        call_list = ["ros2", "bag", "record"]
        if topics is not None:
            call_list.extend(topics)
        if topic_name_space is not None:
            call_list.extend(["--regex", "^" + topic_name_space + "/.*"])
        if record_all or (topics is None and topic_name_space is None):
            call_list.append("--all")
        if exclude_regex is not None:
            call_list.extend(["--exclude", exclude_regex])
        self.output = None
        if prefix is not None:
            self.output = prefix
            if timestamp_suffix:
                self.output += "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(os.path.dirname(os.path.abspath(self.output)), exist_ok=True)
            call_list.extend(["--output", self.output])

        # own session, so that ctrl+C in the terminal does not kill the recorder before we closed it
        self.rosbag_process = subprocess.Popen(call_list, env=os.environ.copy(), start_new_session=True)
        self._closed = False
        # components are forked, so they inherit this object and its atexit hook, only the owner may close the bag
        self._owner_pid = os.getpid()
        atexit.register(self.close)
        print("Started rosbag recording" + ("" if self.output is None else " into " + str(self.output)))

    def close(self, timeout: float = 20.0):
        if self._closed or os.getpid() != self._owner_pid:
            return
        if self.rosbag_process.poll() is None:
            self.rosbag_process.send_signal(signal.SIGINT)
            # only now, so that another ctrl+C while waiting below does not leave the bag unsignalled
            self._closed = True
            try:
                self.rosbag_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print("Rosbag did not close within " + str(timeout) + "s, terminating it.")
                self.rosbag_process.terminate()
                try:
                    self.rosbag_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.rosbag_process.kill()
        self._closed = True
        print("Closed rosbag recording" + ("" if self.output is None else " in " + str(self.output)))

    def terminate(self):
        self.close()

    def join(self, timeout: float = None):
        # so that the recorder can live in the same list as the multiprocessing components
        self.close()
