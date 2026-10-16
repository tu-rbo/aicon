"""
AICON Visualization as RViz Markers -- Currently not ported to ROS2 yet

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import threading
from threading import Lock

import torch
import rclpy
import scipy.spatial.transform
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker

from aicon_base.math.se3_and_SE3 import H_transform_to_trans_quat, rotation_matrix_axis_angle
from aicon_ros_python_wrapping.util_ros import get_ros_point, time_float_to_ros, derivative_msg_to_derivative_dict
from aicon_ros_python_wrapping.ros_wrapping import default_ros_namespace, spin_node_in_thread
from aicon_ros.msg import ListOfStringListsMsg, DerivativesMsg

tf_broadcatser = None

JOINT_AXIS_AND_VARIABLE_MARKER_RADIUS = 0.005
JOINT_VALUE_TEXT_SIZE = 0.025
JOINT_AXIS_MARKER_RADIUS = 0.0025
JOINT_VALUE_TEXT_DIST = 0.2
JOINT_AXIS_LENGTH_MULTIPLIER = 1.0

default_qos_settings = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=2,
)

# Global broadcaster holder
tf_broadcaster = None
ros_node = None


def visualize_as_broadcasted_tf(pose: torch.Tensor, time: torch.Tensor, parent_frame: str, visualized_frame: str):
    global tf_broadcaster, ros_node
    if tf_broadcaster is None:
        if ros_node is None:
            ros_node = rclpy.create_node("visualization_publisher", namespace=default_ros_namespace)
        tf_broadcaster = TransformBroadcaster(ros_node)

    translation, quat = H_transform_to_trans_quat(pose, is_batch=False)
    quat_ros = quat[[1, 2, 3, 0]].cpu().numpy()

    t = TransformStamped()
    t.header.stamp = time_float_to_ros(time).to_msg()
    t.header.frame_id = parent_frame
    t.child_frame_id = visualized_frame

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])
    t.transform.rotation.x = float(quat_ros[0])
    t.transform.rotation.y = float(quat_ros[1])
    t.transform.rotation.z = float(quat_ros[2])
    t.transform.rotation.w = float(quat_ros[3])

    tf_broadcaster.sendTransform(t)


def get_markerarray_publisher(topic: str = "aicon/visualizations") -> rclpy.publisher.Publisher:
    global ros_node
    if ros_node is None:
        ros_node = rclpy.create_node("visualization_publisher", namespace=default_ros_namespace)
    return ros_node.create_publisher(MarkerArray, topic, 10)


class UsedGradientViaROS(object):
    def __init__(self, quantity_name: str, ns: str = "/aicon/used_gradients/"):
        global ros_node
        if ros_node is None:
            ros_node = rclpy.create_node("visualization_publisher", namespace=default_ros_namespace)
        self.subscriber = ros_node.create_subscription(
            ListOfStringListsMsg, ns + quantity_name, self.callback, qos_profile=default_qos_settings
        )
        self.traces = []
        self.lock = Lock()
        spin_thread = threading.Thread(target=spin_node_in_thread, args=(ros_node,), daemon=True)
        spin_thread.start()

    def callback(self, msg: ListOfStringListsMsg):
        with self.lock:
            self.traces = [x.string_list for x in msg.mixed_in_at_node]

    def get_traces(self):
        with self.lock:
            t = self.traces
        return t


class FullGradientViaROS(object):
    def __init__(self, quantity_name: str, ns: str = "/aicon/pursued_gradient/"):
        global ros_node
        if ros_node is None:
            ros_node = rclpy.create_node("visualization_publisher", namespace=default_ros_namespace)
        self.subscriber = ros_node.create_subscription(
            DerivativesMsg, ns + quantity_name, self.callback, qos_profile=default_qos_settings
        )
        self.derivatives = dict()
        self.lock = Lock()
        spin_thread = threading.Thread(target=spin_node_in_thread, args=(ros_node,), daemon=True)
        spin_thread.start()

    def callback(self, msg: DerivativesMsg):
        with self.lock:
            self.derivatives = derivative_msg_to_derivative_dict(msg)

    def get_derivatives(self):
        with self.lock:
            d = self.derivatives
        return d


def construct_cov_markerarray(frame_id, ids, mu, sigma, timestamp, color, scale=1.0, ns="visualizations/"):
    ids = ids.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach()
    timestamp = timestamp.detach().cpu().numpy()
    marker_array = MarkerArray()
    m = get_delete_all_marker(frame_id, ns, timestamp)
    marker_array.markers.append(m)
    # getting the distribution eigen vectors and values
    L, V = torch.linalg.eigh(sigma.detach()[:, :3, :3])
    L = L.cpu().numpy()
    if not torch.is_tensor(scale):
        scale = [scale for _ in range(mu.shape[0])]
    else:
        scale = scale.cpu().numpy()
    for i in range(sigma.shape[0]):
        if scale[i] == 0:
            continue
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = time_float_to_ros(timestamp).to_msg()
        m.ns = ns
        m.id = int(ids[i])
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = mu[i, 0]
        m.pose.position.y = mu[i, 1]
        m.pose.position.z = mu[i, 2]

        R = V[i]
        if torch.det(R) < 0:
            R = torch.cat([R[0:1], R[1:2], -R[2:3]])
        quat = scipy.spatial.transform.Rotation.from_matrix(R.cpu().numpy()).as_quat(canonical=False)
        # orientation of the ellipsoid marker
        m.pose.orientation.w = quat[3]
        m.pose.orientation.x = quat[0]
        m.pose.orientation.y = quat[1]
        m.pose.orientation.z = quat[2]
        m.scale.x = L[i, 0] * scale[i]
        m.scale.y = L[i, 1] * scale[i]
        m.scale.z = L[i, 2] * scale[i]

        m.color.a = 0.75
        if isinstance(color, list):
            m.color.r = float(color[i][0])
            m.color.g = float(color[i][1])
            m.color.b = float(color[i][2])
        else:
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
        marker_array.markers.append(m)
    return marker_array


def construct_point_markerarray(frame_id, ids, points, timestamp, color, scale=0.1, ns="visualizations/"):
    ids = ids.detach().cpu().numpy()
    points = points.detach().cpu().numpy()
    timestamp = timestamp.detach().cpu().numpy()
    marker_array = MarkerArray()
    m = get_delete_all_marker(frame_id, ns, timestamp)
    marker_array.markers.append(m)
    if not torch.is_tensor(scale):
        scale = [scale for _ in range(points.shape[0])]
    else:
        scale = scale.cpu().numpy()
    for i in range(points.shape[0]):
        if scale[i] == 0:
            continue
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = time_float_to_ros(timestamp).to_msg()
        m.ns = ns
        m.id = int(ids[i])
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = points[i, 0]
        m.pose.position.y = points[i, 1]
        m.pose.position.z = points[i, 2]

        m.scale.x = scale[i]
        m.scale.y = scale[i]
        m.scale.z = scale[i]

        m.color.a = 0.75
        if isinstance(color, list):
            m.color.r = float(color[i][0])
            m.color.g = float(color[i][1])
            m.color.b = float(color[i][2])
        else:
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
        marker_array.markers.append(m)
    return marker_array


def construct_vector_markers(frame_id, ids, pos, vec, timestamp, color, scale=1.0, ns="feature_cov"):
    pos = pos.detach()
    vec = vec.detach()
    marker_array = MarkerArray()
    m = get_delete_all_marker(frame_id, ns, timestamp)
    marker_array.markers.append(m)
    if not torch.is_tensor(scale):
        scale = [scale for _ in range(vec.shape[0])]
    else:
        scale = scale.cpu().numpy()
    for i in range(vec.shape[0]):
        if scale[i] == 0:
            continue
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = time_float_to_ros(timestamp).to_msg()
        m.ns = ns
        m.id = int(ids[i].cpu().numpy())
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.points.extend(
            [get_ros_point(pos[i].cpu().numpy()), get_ros_point((pos[i] + vec[i] * scale[i]).cpu().numpy())]
        )

        m.color.a = 0.75
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        marker_array.markers.append(m)
    return marker_array


def get_delete_all_marker(frame_id, ns, timestamp):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = time_float_to_ros(timestamp).to_msg()
    m.ns = ns
    m.id = 0
    m.action = Marker.DELETEALL
    return m


def make_marker_array(markers):
    marker_array = MarkerArray()
    marker_array.markers = markers
    return marker_array


def get_pris_markers(frame_id, timestamp, ns, id_start, kinematic_joint_mu, kinematic_joint_sigma):
    azimuth, elevation, p_state = kinematic_joint_mu[0:1], kinematic_joint_mu[1:2], kinematic_joint_mu[2]
    orientation_vector = torch.cat(
        [torch.cos(azimuth) * torch.sin(elevation), torch.sin(azimuth) * torch.sin(elevation), torch.cos(elevation)],
        dim=0,
    )
    start_point = kinematic_joint_mu[3:]

    # get points
    position_points = []
    position_points.append(start_point)
    position_points.append(start_point - p_state * orientation_vector)
    position_points.append(start_point + p_state * orientation_vector)
    position_points.append(start_point - JOINT_AXIS_LENGTH_MULTIPLIER * orientation_vector)
    position_points.append(start_point + JOINT_AXIS_LENGTH_MULTIPLIER * orientation_vector)

    # get uncertainty ellipses
    angle_Sigma = kinematic_joint_sigma[:2, :2]
    L, V = torch.linalg.eigh(angle_Sigma)
    L = L.cpu().numpy()
    x_major_length, y_minor_length = L[1], L[0]
    uncertainty_quaternions = []
    alpha = torch.atan2(V[1, 1], V[1, 0])
    ori_vec = orientation_vector / torch.norm(orientation_vector)

    unit_z = torch.tensor([0.0, 0.0, 1.0], dtype=kinematic_joint_mu.dtype, device=kinematic_joint_mu.device)
    x_axis_unit = torch.cross(ori_vec, unit_z)
    x_axis_unit = x_axis_unit / torch.norm(x_axis_unit)
    y_axis_unit = torch.cross(ori_vec, x_axis_unit)
    y_axis_unit = y_axis_unit / torch.norm(y_axis_unit)
    rot_ellipse = torch.cat([x_axis_unit.unsqueeze(-1), y_axis_unit.unsqueeze(-1), ori_vec.unsqueeze(-1)], dim=-1)
    rot_init = rotation_matrix_axis_angle(unit_z.unsqueeze(0), -alpha.unsqueeze(0))[0]
    rot1 = torch.mm(rot_ellipse, rot_init)
    quat_cone1 = torch.tensor(
        scipy.spatial.transform.Rotation.from_matrix(rot1.cpu().numpy()).as_quat(canonical=True),
        dtype=rot1.dtype,
        device=rot1.device,
    )
    uncertainty_quaternions.append(quat_cone1)

    unit_x = torch.tensor([1.0, 0.0, 0.0], dtype=kinematic_joint_mu.dtype, device=kinematic_joint_mu.device)
    pi = torch.tensor([torch.pi], dtype=kinematic_joint_mu.dtype, device=kinematic_joint_mu.device)
    rot_circ = rotation_matrix_axis_angle(unit_x.unsqueeze(0), pi.unsqueeze(0))[0]

    rot2 = torch.mm(rot1, rot_circ)
    quat_cone2 = torch.tensor(
        scipy.spatial.transform.Rotation.from_matrix(rot2.cpu().numpy()).as_quat(canonical=True),
        dtype=rot1.dtype,
        device=rot1.device,
    )
    uncertainty_quaternions.append(quat_cone2)

    position_points = [p.cpu().numpy() for p in position_points]
    uncertainty_quaternions = [q.cpu().numpy() for q in uncertainty_quaternions]

    prismatic_markers = []
    # axis orientation + joint state marker
    axis_orientation_marker = Marker()
    axis_orientation_marker.header.frame_id = frame_id
    axis_orientation_marker.header.stamp = time_float_to_ros(timestamp).to_msg()
    axis_orientation_marker.ns = ns
    axis_orientation_marker.id = id_start
    axis_orientation_marker.scale.x = JOINT_AXIS_AND_VARIABLE_MARKER_RADIUS
    axis_orientation_marker.scale.y = 0.00001
    axis_orientation_marker.scale.z = 0.00001
    axis_orientation_marker.color.r = 0.0
    axis_orientation_marker.color.g = 1.0
    axis_orientation_marker.color.b = 0.0
    axis_orientation_marker.color.a = 1.0
    pt1 = get_ros_point(position_points[1])
    axis_orientation_marker.points.append(pt1)
    pt2 = get_ros_point(position_points[2])
    axis_orientation_marker.points.append(pt2)
    prismatic_markers.append(axis_orientation_marker)
    # joint orientation marker
    axis_orientation_marker = Marker()
    axis_orientation_marker.header.frame_id = frame_id
    axis_orientation_marker.header.stamp = time_float_to_ros(timestamp).to_msg()
    axis_orientation_marker.ns = ns
    axis_orientation_marker.id = id_start + 1
    axis_orientation_marker.scale.x = JOINT_AXIS_MARKER_RADIUS
    axis_orientation_marker.scale.y = 0.00001
    axis_orientation_marker.scale.z = 0.00001
    axis_orientation_marker.color.r = 0.0
    axis_orientation_marker.color.g = 1.0
    axis_orientation_marker.color.b = 0.0
    axis_orientation_marker.color.a = 1.0
    pt1 = get_ros_point(position_points[3])
    axis_orientation_marker.points.append(pt1)
    pt2 = get_ros_point(position_points[4])
    axis_orientation_marker.points.append(pt2)
    prismatic_markers.append(axis_orientation_marker)
    # joint state text
    axis_orientation_marker = Marker()
    axis_orientation_marker.header.frame_id = frame_id
    axis_orientation_marker.header.stamp = time_float_to_ros(timestamp).to_msg()
    axis_orientation_marker.ns = ns
    axis_orientation_marker.id = id_start + 2
    axis_orientation_marker.type = axis_orientation_marker.TEXT_VIEW_FACING
    axis_orientation_marker.scale.x = 0.0
    axis_orientation_marker.scale.y = 0.0
    axis_orientation_marker.scale.z = JOINT_VALUE_TEXT_SIZE
    axis_orientation_marker.text = "{:.2f}".format(p_state.item() * 100.0) + "cm"
    axis_orientation_marker.pose.position = get_ros_point(position_points[0])
    axis_orientation_marker.pose.position.z -= JOINT_VALUE_TEXT_DIST
    axis_orientation_marker.pose.orientation.x = 0.00001
    axis_orientation_marker.pose.orientation.y = 0.00001
    axis_orientation_marker.pose.orientation.z = 0.0
    axis_orientation_marker.pose.orientation.w = 1.0
    axis_orientation_marker.color.r = 0.0
    axis_orientation_marker.color.g = 1.0
    axis_orientation_marker.color.b = 0.0
    axis_orientation_marker.color.a = 1.0
    prismatic_markers.append(axis_orientation_marker)
    # uncertainty markers
    pris_axis_unc_cone1 = Marker()
    pris_axis_unc_cone1.header.frame_id = frame_id
    pris_axis_unc_cone1.header.stamp = time_float_to_ros(timestamp).to_msg()
    pris_axis_unc_cone1.ns = ns
    pris_axis_unc_cone1.id = id_start + 3
    pris_axis_unc_cone1.type = pris_axis_unc_cone1.MESH_RESOURCE
    pris_axis_unc_cone1.mesh_resource = "package://aicon_ros/meshes/cone.stl"
    pris_axis_unc_cone1.pose.position.x = position_points[0][0].item()
    pris_axis_unc_cone1.pose.position.y = position_points[0][1].item()
    pris_axis_unc_cone1.pose.position.z = position_points[0][2].item()
    pris_axis_unc_cone1.pose.orientation.w = uncertainty_quaternions[0][3].item()
    pris_axis_unc_cone1.pose.orientation.x = uncertainty_quaternions[0][0].item()
    pris_axis_unc_cone1.pose.orientation.y = uncertainty_quaternions[0][1].item()
    pris_axis_unc_cone1.pose.orientation.z = uncertainty_quaternions[0][2].item()
    pris_axis_unc_cone1.color.a = 0.4
    pris_axis_unc_cone1.color.r = 0.0
    pris_axis_unc_cone1.color.g = 1.0
    pris_axis_unc_cone1.color.b = 0.0

    pris_axis_unc_cone1.scale.x = x_major_length.item()  # TODO extra scaling
    pris_axis_unc_cone1.scale.y = y_minor_length.item()
    pris_axis_unc_cone1.scale.z = 1.0
    prismatic_markers.append(pris_axis_unc_cone1)

    pris_axis_unc_cone2 = Marker()
    pris_axis_unc_cone2.header.frame_id = frame_id
    pris_axis_unc_cone2.header.stamp = time_float_to_ros(timestamp).to_msg()
    pris_axis_unc_cone2.ns = ns
    pris_axis_unc_cone2.id = id_start + 4
    pris_axis_unc_cone2.type = pris_axis_unc_cone2.MESH_RESOURCE
    pris_axis_unc_cone2.mesh_resource = "package://aicon_ros/meshes/cone.stl"
    pris_axis_unc_cone2.pose.position.x = position_points[0][0].item()
    pris_axis_unc_cone2.pose.position.y = position_points[0][1].item()
    pris_axis_unc_cone2.pose.position.z = position_points[0][2].item()
    pris_axis_unc_cone2.pose.orientation.w = uncertainty_quaternions[1][3].item()
    pris_axis_unc_cone2.pose.orientation.x = uncertainty_quaternions[1][0].item()
    pris_axis_unc_cone2.pose.orientation.y = uncertainty_quaternions[1][1].item()
    pris_axis_unc_cone2.pose.orientation.z = uncertainty_quaternions[1][2].item()
    pris_axis_unc_cone2.color.a = 0.4
    pris_axis_unc_cone2.color.r = 0.0
    pris_axis_unc_cone2.color.g = 1.0
    pris_axis_unc_cone2.color.b = 0.0

    pris_axis_unc_cone2.scale.x = x_major_length.item()  # TODO extra scaling
    pris_axis_unc_cone2.scale.y = y_minor_length.item()
    pris_axis_unc_cone2.scale.z = 1.0
    prismatic_markers.append(pris_axis_unc_cone2)
    return prismatic_markers
