"""
Active Interconnections for Real-Robot Object Pushing

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import collections
import torch

from aicon_base.connections import ActiveInterconnection
from aicon_base.inference.util import gradient_preserving_clipping, order_and_gradient_preserving_clipping
from aicon_base.math.se3_and_SE3 import exponential_map_se3, homogeneous_transform_inverse
from aicon_real_robots.camera_visibility import visible_likelihood_given_relative_pos
from .params import NUM_KEYPOINTS, UNCERTAINTY_SCALE
from .util import polygon_centroid, make_signed_distance_transform, fit_se2_to_sdf, sample_sdf, point_in_polygon
from aicon_real_robots.connections.connections import (
    EEFTWithPusherConnection,
    JointsVeloConnection,
    JointsProprioConnection,
    PandaJoint2EEConnection,
    BodyPoints2SelfCollisionConnection,
    PandaWithPusherJoint2BodyPointsConnection,
    JointLimitLikelihoodConnection,
    PandaJoint2SingularityConnection,
    JointRangeUtilizationConnection,
)

# Nominal posture for JointRangeUtilizationConnection: q4 folds the elbow into the arm's working range, and q6 is
# chosen so that the pushing stick is vertical in this posture.
POSTURE_NOMINAL = [0.0, 0.0, 0.0, -2.3, 0.0, 2.3, 0.0]

# Joint limit bands for JointLimitLikelihoodConnection. Joint 7 rolls the camera quickly, so its band is wider to
# leave enough time to stop before the limit.
J7_LIMIT_INTERVAL_IN = [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.35]
J7_LIMIT_INTERVAL_OUT = [0.15, 0.15, 0.15, 0.15, 0.25, 0.3, 0.55]

# Length of the pushing stick from the flange [m]
PUSHER_LENGTH = 0.33

# Calibrated panda_link8 -> camera2_color_optical_frame transform, used when it is not available on tf.
H_EE_TO_CAM_FALLBACK = [
    [0.24731536209583282471, -0.96893119812011718750, 0.00273264548741281033, -0.11312059313058853149],
    [0.96893090009689331055, 0.24732157588005065918, 0.00223034899681806564, -0.00336945243179798126],
    [-0.00283689680509269238, 0.00209614518098533154, 0.99999380111694335938, -0.00931791588664054871],
    [0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000],
]


def ee_to_camera_transform(dtype, device):
    """panda_link8 -> camera transform from tf, or the stored calibration if tf does not provide it within 2 s."""
    from aicon_ros_python_wrapping.util_ros import wait_for_tf_frame

    try:
        H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera2_color_optical_frame", timeout=2.0).to(
            dtype=dtype, device=device
        )
        print("No fallback:", H_ee_to_cam)
    except AssertionError:
        print("Fallback on saved transform for cam")
        H_ee_to_cam = torch.tensor(H_EE_TO_CAM_FALLBACK, dtype=dtype, device=device)
    return H_ee_to_cam


def build_connections():
    connections = {
        "KeyPoints2ContactPointsReal": lambda device, dtype, mockbuild: KeyPoints2ContactPointsReal(
            "KeyPoints2ContactPoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ContactPoints2ContactLikelihoodReal": lambda device, dtype, mockbuild: ContactPoints2ContactLikelihoodReal(
            "ContactPoints2ContactLikelihoodReal", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ContactPoints2PushVelocityDirectionsReal": lambda device, dtype, mockbuild: (
            ContactPoints2PushVelocityDirectionsReal(
                "ContactPoints2PushVelocityDirectionsReal", device=device, dtype=dtype, mockbuild=mockbuild
            )
        ),
        "Contacts2ObjectPosition": lambda device, dtype, mockbuild: Contacts2ObjectPosition(
            "Contacts2ObjectPosition", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "PusherNonPenetration2ObjectPosition": lambda device, dtype, mockbuild: PusherNonPenetration2ObjectPosition(
            "PusherNonPenetration2ObjectPosition", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "OwnWeightCompensation": lambda device, dtype, mockbuild: EEFTWithPusherConnection(
            "OwnWeightCompensation", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "LinearVelocityJoints": lambda device, dtype, mockbuild: JointsVeloConnection(
            "LinearVelocityJoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "DirectMeasurementJoints": lambda device, dtype, mockbuild: JointsProprioConnection(
            "DirectMeasurementJoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "RobotForwardKinematics": lambda device, dtype, mockbuild: PandaJoint2EEConnection(
            "RobotForwardKinematics", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "PandaJoint2BodyPoints": lambda device, dtype, mockbuild: PandaWithPusherJoint2BodyPointsConnection(
            "PandaJoint2BodyPoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "BodyPoints2SelfCollision": lambda device, dtype, mockbuild: BodyPoints2SelfCollisionConnection(
            "BodyPoints2SelfCollision", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "EE2PusherHeight": lambda device, dtype, mockbuild: EE2PusherHeight(
            "EE2PusherHeight", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "EE2Pusher": lambda device, dtype, mockbuild: EE2Pusher(
            "EE2Pusher", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "TwoDimKeypointsFrom3D": lambda device, dtype, mockbuild: TwoDimKeypointsFrom3D(
            "TwoDimKeypointsFrom3D", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "Keypoints2ObjectHeight": lambda device, dtype, mockbuild: Keypoints2ObjectHeight(
            "Keypoints2ObjectHeight", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "KeyPoint2ObjectUncertainty": lambda device, dtype, mockbuild: KeyPoint2ObjectUncertainty(
            "KeyPoint2ObjectUncertainty", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "KeyPoints2PositionReal": lambda device, dtype, mockbuild: KeyPoints2PositionReal(
            "KeyPoints2PositionReal", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "HeightDifference": lambda device, dtype, mockbuild: HeightDifference(
            "HeightDifference", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "JointLimitLikelihoodConnection": lambda device, dtype, mockbuild: JointLimitLikelihoodConnection(
            "JointLimitLikelihoodConnection",
            device=device,
            dtype=dtype,
            mockbuild=mockbuild,
            interval_in=J7_LIMIT_INTERVAL_IN,
            interval_out=J7_LIMIT_INTERVAL_OUT,
        ),
        "PandaJoint2SingularityConnection": lambda device, dtype, mockbuild: PandaJoint2SingularityConnection(
            "PandaJoint2SingularityConnection", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "JointRangeUtilization": lambda device, dtype, mockbuild: JointRangeUtilizationConnection(
            "JointRangeUtilization", device=device, dtype=dtype, mockbuild=mockbuild, nominal=POSTURE_NOMINAL
        ),
        "ObjectvsPush2Interactionlikelihood": lambda device, dtype, mockbuild: ObjectvsPush2Interactionlikelihood(
            "ObjectvsPush2Interactionlikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
    return connections


class KeyPoints2PositionReal(ActiveInterconnection):
    """Object pose measurement: the centroid of the keypoints, and the orientation from the object detector's fit
    (`object_angle_meas`, NaN when it was not measured this frame)."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "object_position": (3,),
                "object_angle_meas": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def initial_definitions(self, quantity_names_and_shapes, required_signs_dict=None):
        # NaN ("no angle measured") rather than the default zero, which would be a valid angle
        super().initial_definitions(quantity_names_and_shapes, required_signs_dict)
        self.connected_quantities["object_angle_meas"] = torch.ones(1, dtype=self.dtype, device=self.device) * torch.nan

    def define_implicit_connection_function(self):
        def connection_func(keypoint_locations, object_position, object_angle_meas):
            object_position_center = polygon_centroid(keypoint_locations)
            object_angle = object_angle_meas.reshape(())
            if torch.isnan(object_angle):
                angle_update = torch.zeros_like(object_position[2])
            else:
                angle_update = object_angle - object_position[2]
            object_angle_aligned = object_position[2] + angle_update
            object_position_meas = torch.cat([object_position_center, object_angle_aligned.unsqueeze(0)])
            innovation = object_position_meas - object_position
            return innovation

        return connection_func


class KeyPoints2ContactPointsReal(ActiveInterconnection):
    """Candidate contact point on each edge of the object polygon: the point closest to the pusher, kept
    `offset_length` away from the corners."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "contactpoint_locations": (NUM_KEYPOINTS, 2),
                "pusher_point_batched": (NUM_KEYPOINTS, 2),
                "pusher_point": (2,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        offset_length = 0.025

        def connection_func(keypoint_locations, contactpoint_locations, pusher_point_batched):
            next_keypoint = torch.cat([keypoint_locations[-1:], keypoint_locations[:-1]])
            r = torch.einsum(
                "bi,bi->b", next_keypoint - keypoint_locations, pusher_point_batched - keypoint_locations
            ) / torch.pow(torch.norm(next_keypoint - keypoint_locations, dim=1), 2)
            added_vector = torch.einsum("b,bi->bi", r, (next_keypoint - keypoint_locations))
            inbetween_point = keypoint_locations + added_vector
            offset_length_relative = offset_length / torch.norm(next_keypoint - keypoint_locations, dim=1)
            options = torch.cat(
                [
                    (
                        keypoint_locations + offset_length_relative.unsqueeze(1) * (next_keypoint - keypoint_locations)
                    ).unsqueeze(0),
                    (
                        keypoint_locations
                        + (1.0 - offset_length_relative.unsqueeze(1)) * (next_keypoint - keypoint_locations)
                    ).unsqueeze(0),
                    inbetween_point.unsqueeze(0),
                ]
            )
            idxs = [
                0 if r[i] <= offset_length_relative[i] else 1 if r[i] >= 1 - offset_length_relative[i] else 2
                for i in range(NUM_KEYPOINTS)
            ]
            innovation = (
                torch.cat([options[idx, i : i + 1] for i, idx in enumerate(idxs)], dim=0) - contactpoint_locations
            )
            return innovation

        return connection_func


class ContactPoints2ContactLikelihoodReal(ActiveInterconnection):
    """Likelihood of contact at each candidate contact point, from the pusher's distance to it and the angle of
    approach relative to the edge normal."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "contactpoint_locations": (NUM_KEYPOINTS, 2),
                "pusher_point_batched": (NUM_KEYPOINTS, 2),
                "contact_likelihoods": (NUM_KEYPOINTS,),
                "pusher_point": (2,),
                "keypoint_locations": (NUM_KEYPOINTS, 2),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        dist_min = 0.005
        dist_max = 0.07

        def connection_func(keypoint_locations, contactpoint_locations, pusher_point_batched, contact_likelihoods):
            next_keypoint = torch.cat([keypoint_locations[-1:], keypoint_locations[:-1]])
            contact_tangents = next_keypoint - keypoint_locations
            contact_normals = torch.cat([contact_tangents[:, 1:2], -contact_tangents[:, 0:1]], dim=1)
            contact_normals = contact_normals / torch.norm(contact_normals, dim=1, keepdim=True)
            pusher2contact = contactpoint_locations - pusher_point_batched
            dist = torch.norm(pusher2contact, dim=1)
            dist = gradient_preserving_clipping(dist, dist_min, dist_max)
            angle = gradient_preserving_clipping(
                torch.acos(
                    gradient_preserving_clipping(
                        torch.einsum("bi,bi->b", pusher2contact, contact_normals) / torch.norm(pusher2contact, dim=1),
                        -0.9999,
                        0.9999,
                    )
                ),
                0.0,
                torch.pi,
            ).detach()
            contact_likelihoods_meas = (2 * torch.pow(1 + (dist - dist_min) / 100 * 500, -5)) * (
                torch.erf((torch.pi / 2 - angle)) / 4.0 + 0.2501
            )
            contact_likelihoods_meas = order_and_gradient_preserving_clipping(
                contact_likelihoods_meas, 0.0001, 1.0, scale=1.0
            )
            innovation = contact_likelihoods_meas - contact_likelihoods
            return innovation

        return connection_func


class ObjectvsPush2Interactionlikelihood(ActiveInterconnection):
    """Likelihood that the pusher interacts with the object at all: 1 when close, falling off with distance."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "object_position": (NUM_KEYPOINTS, 2),
                "pusher_point": (2,),
                "interaction_likelihood": (NUM_KEYPOINTS, 2),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        steepness = 1.8
        limit = 0.35
        normalizer = torch.exp(torch.tensor(steepness * limit, device=self.device, dtype=self.dtype))
        k = 50.0  # sharpness of the near/far transition
        mid_tip = 0.25
        lower_bound = 0.1

        def connection_func(object_position, pusher_point, interaction_likelihood):
            dist = gradient_preserving_clipping(torch.norm(object_position[:2] - pusher_point), 0.0, limit)
            dist = torch.clip(dist, lower_bound, limit * 1.1)
            transition = torch.sigmoid(k * (dist - mid_tip))

            far_part = gradient_preserving_clipping(10 * (normalizer - torch.exp(steepness * dist)), 0.01, 1.0)

            likelihood = (1 - transition) + transition * far_part
            return interaction_likelihood - likelihood

        return connection_func


# Gain on the rotation each contact induces relative to its translation.
CONTACT_ROT_GAIN: float = 3.0


class ContactPoints2PushVelocityDirectionsReal(ActiveInterconnection):
    """Direction of object motion when pushed at each candidate contact: translation along the edge normal, and a
    rotation proportional to the lever arm about the object's centre."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "contactpoint_locations": (NUM_KEYPOINTS, 2),
                "object_COM": (2,),
                "push_velocity_directions": (NUM_KEYPOINTS, 3),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(keypoint_locations, contactpoint_locations, object_COM, push_velocity_directions):
            next_keypoint = torch.cat([keypoint_locations[-1:], keypoint_locations[:-1]])
            contact_tangents = next_keypoint - keypoint_locations
            contact_normals = torch.cat([contact_tangents[:, 1:2], -contact_tangents[:, 0:1]], dim=1)
            contact_normals = contact_normals / torch.norm(contact_normals, dim=1, keepdim=True)
            contact2COM = object_COM.unsqueeze(0) - contactpoint_locations
            cross2d = contact_normals[:, 0:1] * contact2COM[:, 1:2] - contact_normals[:, 1:2] * contact2COM[:, 0:1]
            # Normalised by the norm over all contacts (not per row), so the rotation is proportional to the lever
            # arm. The overall factor is a rough approximation; the true ratio depends on friction.
            rotVStrans = torch.arcsin(cross2d / torch.norm(contact2COM)) * 5.0 / torch.pi
            rot_direction = CONTACT_ROT_GAIN * rotVStrans
            lin_direction = contact_normals
            push_velocity_directions_curr = torch.cat([lin_direction, rot_direction], dim=1)
            innovation = push_velocity_directions_curr - push_velocity_directions
            return innovation

        return connection_func


class Contacts2ObjectPosition(ActiveInterconnection):
    """Contact-based motion model: the object moves along the push directions of the likely contacts. Used by the
    ObjectEstimator for its gradient only (the predicted value is discarded)."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "object_position": (3,),
                "contact_likelihoods": (NUM_KEYPOINTS,),
                "push_velocity_directions": (NUM_KEYPOINTS, 3),
                "interaction_likelihood": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            required_signs_dict={
                "object_position": {"interaction_likelihood": 1},
            },
        )

    def define_implicit_connection_function(self):
        def connection_func(object_position, contact_likelihoods, push_velocity_directions, interaction_likelihood):
            directions = torch.cat(
                [
                    push_velocity_directions[:, :2]
                    / torch.norm(push_velocity_directions[:, :2], dim=-1, keepdim=True).detach(),
                    gradient_preserving_clipping(push_velocity_directions[:, 2:], -0.1, 0.1),
                ],
                dim=1,
            )
            object_position_new = object_position + interaction_likelihood * torch.sum(
                torch.pow(contact_likelihoods.unsqueeze(1), 2) * directions, dim=0
            )
            return object_position_new

        return connection_func


class PusherNonPenetration2ObjectPosition(ActiveInterconnection):
    """The pusher cannot be inside the object: a one-sided positional constraint.

    Returns `object_position` shifted along the outward normal of the closest contour edge by exactly the amount
    that brings the pusher's surface back onto the contour; unchanged if the pusher is outside the contour or at a
    different height. Uses the absolute pusher position from the joint encoders, so it does not drift while the
    object is not observed.

    Both TwoDimKeypointEstimator (on the contour) and ObjectEstimator (on the pose) apply the same shift, as a
    value only (`x + (out - x).detach()`), so the hard geometric selections below never enter a gradient.
    Orientation is not changed.
    """

    pusher_radius = 0.015  # radius of the pushing stick [m]
    max_height_difference = 0.06  # only constrain when pusher and object are at the same height [m]

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "object_position": (3,),
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "pusher_point": (2,),
                "pusher_height": (1,),
                "object_height": (1,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        pusher_radius = self.pusher_radius
        max_height_difference = self.max_height_difference

        def connection_func(object_position, keypoint_locations, pusher_point, pusher_height, object_height):
            if torch.abs(pusher_height[0] - object_height[0]) > max_height_difference:
                return object_position

            polygon = keypoint_locations
            next_point = torch.cat([polygon[-1:], polygon[:-1]])
            tangents = next_point - polygon
            lengths_sq = torch.sum(torch.pow(tangents, 2), dim=1)
            if torch.any(lengths_sq < 1e-12):  # degenerate contour, nothing to constrain against
                return object_position

            # closest point on each edge segment (not line: the contour is non-convex)
            r = torch.clamp(
                torch.einsum("bi,bi->b", pusher_point.unsqueeze(0) - polygon, tangents) / lengths_sq, 0.0, 1.0
            )
            closest = polygon + r.unsqueeze(1) * tangents
            pusher2closest = closest - pusher_point.unsqueeze(0)
            distances = torch.norm(pusher2closest, dim=1)
            idx = int(torch.argmin(distances))

            # outward normal of the closest edge; the winding of the contour is taken from its signed area
            signed_area = 0.5 * torch.sum(polygon[:, 0] * next_point[:, 1] - next_point[:, 0] * polygon[:, 1])
            normal = torch.stack([tangents[idx, 1], -tangents[idx, 0]]) / torch.sqrt(lengths_sq[idx])
            if signed_area < 0.0:  # clockwise: [t_y, -t_x] points inward
                normal = -normal

            inside = point_in_polygon(pusher_point, polygon)
            signed_distance = -distances[idx] if inside else distances[idx]
            # distances are to the pusher's centre, so the surfaces touch at signed_distance == pusher_radius
            violation = pusher_radius - signed_distance
            if violation <= 0.0:
                return object_position

            return object_position + torch.cat([-violation * normal, torch.zeros_like(object_position[2:])])

        return connection_func


class EE2Pusher(ActiveInterconnection):
    """2D position of the pusher tip from the end-effector pose."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pusher_point": (2,), "pose_ee": (6,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee2pusher = torch.eye(4, dtype=self.dtype, device=self.device)
        H_ee2pusher[2, 3] = PUSHER_LENGTH

        def connection_func(pusher_point, pose_ee):
            H_ee = exponential_map_se3(pose_ee)
            H_pusher = torch.einsum("ij,jk->ik", H_ee, H_ee2pusher)
            pusher_point_pred = H_pusher[:2, 3]
            return pusher_point - pusher_point_pred

        return connection_func


class EE2PusherHeight(ActiveInterconnection):
    """Height of the pusher tip from the end-effector pose."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pusher_height": (1,), "pose_ee": (6,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee2pusher = torch.eye(4, dtype=self.dtype, device=self.device)
        H_ee2pusher[2, 3] = PUSHER_LENGTH

        def connection_func(pusher_height, pose_ee):
            H_ee = exponential_map_se3(pose_ee)
            H_pusher = torch.einsum("ij,jk->ik", H_ee, H_ee2pusher)
            pusher_height_pred = H_pusher[2:3, 3]
            return pusher_height - pusher_height_pred

        return connection_func


class Keypoints2ObjectHeight(ActiveInterconnection):
    """Object height as the median base-frame height of the measured keypoints."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"keypoints_in_CF": (NUM_KEYPOINTS, 3), "pose_ee": (6,), "object_height": (1,), "object_position": (2,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee_to_cam = ee_to_camera_transform(self.dtype, self.device)
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)

        def connection_func(object_height, keypoints_in_CF, pose_ee):
            H_base_to_ee = exponential_map_se3(pose_ee)
            keypoint_heights = torch.einsum(
                "ij,jk,bk->bi",
                H_base_to_ee,
                H_ee_to_cam,
                torch.cat(
                    [
                        keypoints_in_CF,
                        torch.ones(NUM_KEYPOINTS, 1, dtype=keypoints_in_CF.dtype, device=keypoints_in_CF.device),
                    ],
                    dim=1,
                ),
            )[:, 2]
            object_height_pred = torch.nanmedian(keypoint_heights).unsqueeze(0)
            return object_height - object_height_pred

        return connection_func


# --- Keypoint fit (TwoDimKeypointsFrom3D) ---

# Orientation margin [m of mean contour distance] for truncated contours: the sweep's minimum is only taken if it
# beats the predicted angle by this much (see fit_se2_to_sdf's hold_margin).
ORIENTATION_MARGIN = 0.010
# The centroid of a truncated contour is not the object's centre, so its translation is searched instead, provided at
# least JOINT_SEARCH_MIN_VALID keypoints are off the image border.
JOINT_SEARCH_MIN_VALID = 3
# A contour whose non-border points fit the template better than VETO_BYPASS_RESIDUAL [m] (with at least
# VETO_BYPASS_MIN_VALID such points) is not rejected by the visibility check.
VETO_BYPASS_RESIDUAL = 0.008
VETO_BYPASS_MIN_VALID = 6
# A contour the sensor rejected is still used if its non-border points (at least PARTIAL_CONTOUR_MIN_POINTS) fit the
# template better than PARTIAL_CONTOUR_GATE [m] and it lies within PARTIAL_CONTOUR_MAX_DIST_FROM_OBS [m] of one of
# the last PARTIAL_CONTOUR_OBS_MEMORY observed contour centroids.
PARTIAL_CONTOUR_GATE = 0.020
PARTIAL_CONTOUR_MIN_POINTS = 2
PARTIAL_CONTOUR_MAX_DIST_FROM_OBS = 0.20
PARTIAL_CONTOUR_OBS_MEMORY = 30


class TwoDimKeypointsFrom3D(ActiveInterconnection):
    """Measurement of the 2D object keypoints: fits the registered template to the measured contour.

    The contour (lifted into the base frame) is de-rotated by the predicted pose and matched against the template's
    distance transform by an exhaustive angle sweep, with the predicted orientation as a soft prior. The resulting
    correction to the predicted pose is gated on fit quality and smoothed; corrections from truncated contours
    (keypoints on the image border) are damped and capped in translation.
    """

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "keypoints_in_CF": (NUM_KEYPOINTS, 3),
                "pose_ee": (6,),
                "keypoints_registered": (NUM_KEYPOINTS, 2),
                "object_position": (3,),
                # the contour the sensor rejected, for TwoDimKeypointEstimator's salvage_partial
                "keypoints_partial_in_CF": (NUM_KEYPOINTS, 3),
                "keypoints_valid": (NUM_KEYPOINTS,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee_to_cam = ee_to_camera_transform(self.dtype, self.device)
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)
        alpha = 0.99  # translation gain for complete contours
        alpha_rot = 0.5  # orientation gain for small corrections
        # Large orientation corrections from clean fits are applied almost fully (up to 0.9), so the estimate snaps
        # to the object after fast rotations or re-acquisition.
        theta_snap = torch.pi / 4
        snap_residual = 0.004  # [m] only fits cleaner than this may snap
        # Fits with a larger mean contour residual [m] are rejected (the prediction is kept). Truncated contours
        # include the frame's cut, so they get a looser limit.
        max_fit_residual = 0.008
        max_fit_residual_truncated = 0.015
        # after this many consecutive rejections the measurement is let through again
        max_consecutive_rejects = 15
        self.consecutive_rejects = 0
        # weight of the predicted orientation as a prior in the sweep
        prior_weight = 0.5
        # Translation from truncated contours: damped, and capped per frame [m].
        alpha_truncated = 0.6
        max_t_step_truncated = 0.008
        # Translation jumps from complete contours larger than this [m] are rejected as well; when the rejections run
        # out, the correction is capped to this instead.
        max_t_step_full = 0.05
        self.sdf_ref = None
        self.sdf_scale = None
        self.sdf_min_xy = None
        self.fixed_t_offset = None

        def connection_func(
            keypoint_locations,
            keypoints_in_CF,
            pose_ee,
            keypoints_registered,
            object_position,
            keypoints_valid,
            keypoints_partial_in_CF=None,
        ):
            if self.sdf_ref is None:
                self.fixed_t_offset = torch.mean(keypoints_registered, dim=0, keepdim=True)
                self.sdf_ref, self.sdf_scale, self.sdf_min_xy = make_signed_distance_transform(
                    keypoints_registered - self.fixed_t_offset, res=256, device=keypoints_registered.device
                )
            t_pred = object_position[:2]
            theta_pred = object_position[2]
            c, s = torch.cos(-theta_pred), torch.sin(-theta_pred)
            R_pred = torch.cat(
                [
                    torch.cat([c.unsqueeze(0), (-s).unsqueeze(0)], dim=0).unsqueeze(0),
                    torch.cat([s.unsqueeze(0), c.unsqueeze(0)], dim=0).unsqueeze(0),
                ],
                dim=0,
            )
            H_base_to_ee = exponential_map_se3(pose_ee)

            # the measured contour keypoints in the base frame
            keypoints_in_frame = torch.einsum(
                "ij,jk,bk->bi",
                H_base_to_ee,
                H_ee_to_cam,
                torch.cat(
                    [
                        keypoints_in_CF,
                        torch.ones(NUM_KEYPOINTS, 1, dtype=keypoints_in_CF.dtype, device=keypoints_in_CF.device),
                    ],
                    dim=1,
                ),
            )

            # --- distance transform alignment ---
            valid_mask = keypoints_valid > 0.5
            n_valid = int(torch.sum(valid_mask))
            truncated = n_valid < NUM_KEYPOINTS

            keypoints_pred = keypoints_in_frame[:, :2] - t_pred
            centroid = polygon_centroid(keypoints_pred)
            keypoints_pred = keypoints_pred - centroid
            pred_from_guess = (R_pred @ keypoints_pred.T).T
            solve_translation = "search" if truncated and n_valid >= JOINT_SEARCH_MIN_VALID else "centroid"
            hold_margin = ORIENTATION_MARGIN if truncated else None
            theta_align, t_align, fit_residual = fit_se2_to_sdf(
                pred_from_guess,
                self.sdf_ref,
                self.sdf_scale,
                self.sdf_min_xy,
                theta_prior=0.0,
                prior_weight=prior_weight,
                solve_translation=solve_translation,
                hold_margin=hold_margin,
            )

            # residual of the non-border points only, which decides whether the visibility check is bypassed
            shape_ok = False
            if int(n_valid) >= VETO_BYPASS_MIN_VALID:
                keep = valid_mask.reshape(-1)[: pred_from_guess.shape[0]].to(torch.bool)
                if int(keep.sum()) >= VETO_BYPASS_MIN_VALID:
                    c_a, s_a = torch.cos(theta_align), torch.sin(theta_align)
                    R_a = torch.stack([torch.stack([c_a, -s_a]), torch.stack([s_a, c_a])])
                    in_tpl = (R_a @ pred_from_guess.T).T + t_align
                    r_m = sample_sdf(self.sdf_ref, in_tpl[keep], self.sdf_scale, self.sdf_min_xy)
                    masked_resid = float(torch.sqrt(r_m**2 + 1e-6).mean() / self.sdf_scale)
                    shape_ok = masked_resid < VETO_BYPASS_RESIDUAL

            # Measured pose. The fit maps R(theta_align) @ pred_from_guess + t_align ~= registered - offset, with
            # pred_from_guess = R(-theta_pred) @ (m - t_pred - centroid).
            theta_meas = theta_pred - theta_align
            c_m, s_m = torch.cos(theta_meas), torch.sin(theta_meas)
            R_meas = torch.stack([torch.stack([c_m, -s_m]), torch.stack([s_m, c_m])]).to(
                dtype=keypoints_pred.dtype, device=keypoints_pred.device
            )
            t_meas = t_pred + centroid - R_meas @ (t_align + self.fixed_t_offset.squeeze(0))

            delta_theta = torch.atan2(torch.sin(theta_meas - theta_pred), torch.cos(theta_meas - theta_pred))
            delta_t = t_meas - t_pred

            # --- outlier gating on fit quality, and on physically impossible jumps of complete contours ---
            residual_limit = max_fit_residual_truncated if truncated else max_fit_residual
            reject = fit_residual > residual_limit
            if (not reject) and (not truncated) and float(torch.norm(delta_t)) > max_t_step_full:
                reject = True
            if reject:
                self.consecutive_rejects += 1
                print(
                    f"Reject keypoint fit (residual={fit_residual.item():.4f}m > "
                    f"{residual_limit}m{', truncated' if truncated else ''}; "
                    f"dtheta={delta_theta.item():.3f}rad, "
                    f"dt={torch.norm(delta_t).item():.3f}m, {self.consecutive_rejects} in a row)"
                )
                if self.consecutive_rejects < max_consecutive_rejects:
                    return torch.zeros_like(keypoint_locations)
                # persistent disagreement: the prediction is probably wrong, re-acquire (at a bounded rate)
                print("Too many consecutive rejects, re-acquiring from measurement")
                dt_norm = torch.norm(delta_t)
                if float(dt_norm) > max_t_step_full:
                    delta_t = delta_t * (max_t_step_full / dt_norm)
            else:
                self.consecutive_rejects = 0

            # --- smoothing ---
            allow_snap = fit_residual < snap_residual
            if allow_snap:
                gain_rot = alpha_rot + (1.0 - alpha_rot) * torch.clamp(torch.abs(delta_theta) / theta_snap, max=1.0)
                gain_rot = torch.clamp(gain_rot, max=0.9)
            else:
                gain_rot = alpha_rot
            theta_new = theta_pred + gain_rot * delta_theta

            if truncated:
                gain_t = alpha_truncated
                delta_t_norm = torch.norm(delta_t)
                if delta_t_norm > max_t_step_truncated:
                    delta_t = delta_t * (max_t_step_truncated / delta_t_norm)
            else:
                gain_t = alpha
            t_new = t_pred + gain_t * delta_t

            c, s = torch.cos(theta_new), torch.sin(theta_new)
            R_new = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(
                dtype=keypoints_pred.dtype, device=keypoints_pred.device
            )

            keypoint_locations_pred = torch.einsum("ij,kj->ki", R_new, keypoints_registered) + t_new

            # reject the update if the new contour would not even be visible to the camera
            keypoints_h = torch.cat(
                [
                    keypoint_locations_pred,
                    torch.mean(keypoints_in_frame[:, 2]).unsqueeze(0).expand(keypoints_in_frame.shape[0], -1),
                    torch.ones(NUM_KEYPOINTS, 1, dtype=keypoint_locations.dtype, device=keypoint_locations.device),
                ],
                dim=1,
            )
            relative_pos = torch.einsum(
                "ki,ij,bj->bk",
                homogeneous_transform_inverse(H_ee_to_cam),
                homogeneous_transform_inverse(H_base_to_ee),
                keypoints_h,
            )[:, :3]
            likelihood_visible = visible_likelihood_given_relative_pos(
                relative_pos,
                steepness=1000.0,
                steepness_depth=None,
                depth_midpoint=None,
                fov_max_x=0.6,
                fov_max_y=0.37,
                fov_adder=0.1,
            )
            salvaged_frame = getattr(self, "_salvage_pending", False)
            self._salvage_pending = False
            # remember where the object was actually observed (not salvaged), for salvage_partial
            if not getattr(self, "_is_salvaged_tick", False):
                if not hasattr(self, "_recent_obs"):
                    self._recent_obs = collections.deque(maxlen=PARTIAL_CONTOUR_OBS_MEMORY)
                self._recent_obs.append(polygon_centroid(keypoints_in_frame[:, :2]).detach())
            self._is_salvaged_tick = False
            # contours confirmed by their shape (salvaged, or shape_ok) bypass this check
            if torch.sum(likelihood_visible < 0.5) > 1 and not salvaged_frame and not shape_ok:
                print("Reject keypoints not visible enough!", likelihood_visible)
                return torch.zeros_like(keypoint_locations_pred)
            return keypoint_locations - keypoint_locations_pred

        return connection_func

    def salvage_partial(self, keypoints_partial_in_CF, keypoints_valid, pose_ee, keypoints_registered, object_position):
        """Return a contour the sensor rejected if its shape still identifies it as the object, else None.

        Accepted if the non-border points lie on the template outline at the best fitting pose, and the contour is
        close to where the object was recently observed. Such a contour then also bypasses the visibility check in
        connection_func (called next in the same tick).
        """
        if self.sdf_ref is None:
            return None
        if keypoints_partial_in_CF is None or torch.all(torch.isnan(keypoints_partial_in_CF)):
            return None
        fin = torch.isfinite(keypoints_partial_in_CF[:, 0])
        kv = keypoints_valid.reshape(-1)
        if kv.numel() < NUM_KEYPOINTS:  # short contours come with a short mask
            kv = torch.cat([kv, torch.zeros(NUM_KEYPOINTS - kv.numel(), dtype=kv.dtype, device=kv.device)])
        val = (kv[:NUM_KEYPOINTS] > 0.5) & fin
        if int(fin.sum()) < 3 or int(val.sum()) < PARTIAL_CONTOUR_MIN_POINTS:
            return None
        if (
            torch.any(torch.isnan(pose_ee))
            or torch.any(torch.isnan(object_position))
            or not torch.is_nonzero(torch.sum(object_position))
        ):
            return None
        H_base_to_ee = exponential_map_se3(pose_ee).detach()
        n_fin = int(fin.sum())
        w = torch.einsum(
            "ij,jk,bk->bi",
            H_base_to_ee,
            self.H_ee_to_cam,
            torch.cat(
                [
                    keypoints_partial_in_CF[fin],
                    torch.ones(n_fin, 1, dtype=keypoints_partial_in_CF.dtype, device=keypoints_partial_in_CF.device),
                ],
                dim=1,
            ),
        )[:, :2]
        theta_pred = object_position[2].detach()
        c, s_ = torch.cos(-theta_pred), torch.sin(-theta_pred)
        R = torch.stack([torch.stack([c, -s_]), torch.stack([s_, c])])
        cen = polygon_centroid(w)
        pts = (R @ (w - cen).T).T
        try:
            theta_align, t_align, _ = fit_se2_to_sdf(
                pts,
                self.sdf_ref,
                self.sdf_scale,
                self.sdf_min_xy,
                theta_prior=0.0,
                prior_weight=0.5,
                hold_margin=ORIENTATION_MARGIN,
            )
        except RuntimeError:
            return None
        cf, sf = torch.cos(theta_align), torch.sin(theta_align)
        Rf = torch.stack([torch.stack([cf, -sf]), torch.stack([sf, cf])])
        in_template = (Rf @ pts.T).T + t_align
        resid = float(
            torch.sqrt(
                sample_sdf(self.sdf_ref, in_template[val[fin]], self.sdf_scale, self.sdf_min_xy) ** 2 + 1e-6
            ).mean()
            / self.sdf_scale
        )
        if resid >= PARTIAL_CONTOUR_GATE:
            return None
        # must be near a recent observation (not near the belief, which may be what is wrong)
        obs = getattr(self, "_recent_obs", None)
        if not obs:
            return None
        here = polygon_centroid(w)
        near = min(float(torch.norm(here - o)) for o in obs)
        if near >= PARTIAL_CONTOUR_MAX_DIST_FROM_OBS:
            return None
        self._salvage_pending = True
        self._is_salvaged_tick = True
        return keypoints_partial_in_CF


class KeyPoint2ObjectUncertainty(ActiveInterconnection):
    """Uncertainty about the object pose: how badly the camera currently sees the keypoints, plus a term that grows
    with the tilt of the pushing stick (a poor viewing pose). Normalised by UNCERTAINTY_SCALE to roughly [0, 1]."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "keypoint_locations": (NUM_KEYPOINTS, 2),
                "object_uncertainty": (NUM_KEYPOINTS),
                "object_height": (1,),
                "pose_ee": (6,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        H_ee_to_cam = ee_to_camera_transform(self.dtype, self.device)
        self.H_ee_to_cam = torch.clone(H_ee_to_cam)

        def connection_func(keypoint_locations, object_uncertainty, object_height, pose_ee):
            H_base_to_ee = exponential_map_se3(torch.cat([pose_ee[:3].detach(), pose_ee[3:]]))

            keypoints_h = torch.cat(
                [
                    keypoint_locations,
                    object_height.expand(NUM_KEYPOINTS).unsqueeze(-1),
                    torch.ones(NUM_KEYPOINTS, 1, dtype=keypoint_locations.dtype, device=keypoint_locations.device),
                ],
                dim=1,
            )
            relative_pos = torch.einsum(
                "ki,ij,bj->bk",
                homogeneous_transform_inverse(H_ee_to_cam),
                homogeneous_transform_inverse(H_base_to_ee),
                keypoints_h,
            )[:, :3]
            # fov_adder > 0 keeps a graded likelihood outside the field of view, and "min" passes the gradient of the
            # worse image axis at full strength.
            likelihood_visible = visible_likelihood_given_relative_pos(
                relative_pos,
                steepness=10.0,
                steepness_depth=None,
                depth_midpoint=None,
                fov_max_x=0.6,
                fov_max_y=0.37,
                fov_adder=0.1,
                combine="min",
            )
            xy_magnitude = torch.norm(H_base_to_ee[:2, 2])
            angle_to_good_viewing_pose = torch.arcsin(xy_magnitude)
            mean_likelihood_visible = torch.mean(likelihood_visible)

            # The clipping preserves gradients, so UNCERTAINTY_SCALE (divided out here and multiplied back in by
            # Push2GoalReal) does not change any gradient.
            object_uncertainty_pred = gradient_preserving_clipping(
                (
                    torch.abs(30 * (1.0 - mean_likelihood_visible.unsqueeze(0)))
                    + 1 * torch.pow(angle_to_good_viewing_pose * 3.0, 2)
                )
                / UNCERTAINTY_SCALE,
                0,
                1,
            )
            return object_uncertainty - object_uncertainty_pred

        return connection_func


PROBE_DEPTH = 0.03
RELEASE_LIFT = 0.015
CONTACT_DEPTH = 0.026
FORCE_KNEE_SHARPNESS = 2.0
PUSH_KNEE_FORCE = 0.5
RELEASE_KNEE_FORCE = 2.0


class HeightDifference(ActiveInterconnection):
    """Signed error between the pusher height and the force-modulated height setpoint."""

    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"pusher_height": (1,), "object_height": (1,), "height_difference": (1,), "ee_force_external": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(pusher_height, object_height, height_difference, ee_force_external):
            target_height = object_height - CONTACT_DEPTH
            force = ee_force_external[2]

            push_weight = torch.sigmoid((PUSH_KNEE_FORCE - force) * FORCE_KNEE_SHARPNESS)
            release_weight = torch.sigmoid((force - RELEASE_KNEE_FORCE) * FORCE_KNEE_SHARPNESS)
            equilibrium_offset = (RELEASE_LIFT * release_weight - PROBE_DEPTH * push_weight) / (
                1.0 + push_weight + release_weight
            )
            height_error = pusher_height - (target_height + equilibrium_offset)

            return height_difference - height_error

        return connection_func
