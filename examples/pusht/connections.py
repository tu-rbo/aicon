"""
Active Interconnections for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection
from aicon_base.inference.util import gradient_preserving_clipping
from .params import NUM_KEYPOINTS


def build_connections():
    connections = {
        "KeyPoints2Position": lambda device, dtype, mockbuild: KeyPoints2Position(
            "KeyPoints2Position", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "PusherPointMeasurement": lambda device, dtype, mockbuild: PusherPointMeasurement(
            "PusherPointMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "PusherPointAction": lambda device, dtype, mockbuild: PusherPointAction(
            "PusherPointAction", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "KeyPoints2ContactPoints": lambda device, dtype, mockbuild: KeyPoints2ContactPoints(
            "KeyPoints2ContactPoints", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ContactPoints2ContactLikelihood": lambda device, dtype, mockbuild: ContactPoints2ContactLikelihood(
            "ContactPoints2ContactLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ContactPoints2PushVelocityDirections": lambda device, dtype, mockbuild: ContactPoints2PushVelocityDirections(
            "ContactPoints2PushVelocityDirections", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "Contacts2ObjectPosition": lambda device, dtype, mockbuild: Contacts2ObjectPosition(
            "Contacts2ObjectPosition", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ObjectvsPush2InteractionlikelihoodSim": lambda device, dtype, mockbuild: ObjectvsPush2InteractionlikelihoodSim(
            "ObjectvsPush2InteractionlikelihoodSim", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
    return connections


class KeyPoints2Position(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"keypoint_locations": (NUM_KEYPOINTS, 2), "object_position": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(keypoint_locations, object_position):
            object_position_center = torch.mean(keypoint_locations, dim=0)
            ref_point = keypoint_locations[0] + 0.5 * (keypoint_locations[1] - keypoint_locations[0])
            vec = object_position_center - ref_point
            object_angle = torch.atan2(vec[1:2], vec[0:1]) - torch.pi / 2.0
            object_position_meas = torch.cat([ref_point, object_angle])
            innovation = object_position_meas - object_position
            return innovation

        return connection_func


class PusherPointMeasurement(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name, {"pusher_point_meas": (2,), "pusher_point": (2,)}, dtype=dtype, device=device, mockbuild=mockbuild
        )

    def define_implicit_connection_function(self):
        def connection_func(pusher_point_meas, pusher_point):
            innovation = pusher_point_meas - pusher_point
            return innovation

        return connection_func


class PusherPointAction(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name, {"pusher_point_action": (2,), "pusher_point": (2,)}, dtype=dtype, device=device, mockbuild=mockbuild
        )

    def define_implicit_connection_function(self):
        def connection_func(pusher_point_action, pusher_point):
            pusher_point_new = pusher_point + pusher_point_action
            return pusher_point_new

        return connection_func


class KeyPoints2ContactPoints(ActiveInterconnection):
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
        offset_length = 10.0

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


class ContactPoints2ContactLikelihood(ActiveInterconnection):
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
        def connection_func(keypoint_locations, contactpoint_locations, pusher_point_batched, contact_likelihoods):
            next_keypoint = torch.cat([keypoint_locations[-1:], keypoint_locations[:-1]])
            contact_tangents = next_keypoint - keypoint_locations
            contact_normals = torch.cat([contact_tangents[:, 1:2], -contact_tangents[:, 0:1]], dim=1)
            contact_normals = contact_normals / torch.norm(contact_normals, dim=1, keepdim=True)
            pusher2contact = contactpoint_locations - pusher_point_batched
            dist = torch.norm(pusher2contact, dim=1)
            dist = gradient_preserving_clipping(dist, 14.1, 50.0)
            angle = torch.acos(
                gradient_preserving_clipping(
                    torch.einsum("bi,bi->b", pusher2contact, contact_normals) / torch.norm(pusher2contact, dim=1),
                    -0.9999,
                    0.9999,
                )
            ).detach()
            contact_likelihoods_meas = (2 * torch.pow(1 + (dist - 14) / 100, -5)) * (
                torch.erf((torch.pi / 2 - angle)) / 4.0 + 0.7501
            )
            contact_likelihoods_meas = gradient_preserving_clipping(contact_likelihoods_meas, 0.0001, 1.0)
            innovation = contact_likelihoods_meas - contact_likelihoods
            return innovation

        return connection_func


class ContactPoints2PushVelocityDirections(ActiveInterconnection):
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
            # Rough rotation-vs-translation ratio of a push; the true factor depends on friction.
            rotVStrans = torch.arcsin(cross2d / torch.norm(contact2COM)) * 5.0 / torch.pi
            rot_direction = 3.0 * rotVStrans
            lin_direction = contact_normals
            push_velocity_directions_curr = torch.cat([lin_direction, rot_direction], dim=1)
            innovation = push_velocity_directions_curr - push_velocity_directions
            return innovation

        return connection_func


class Contacts2ObjectPosition(ActiveInterconnection):
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


class ObjectvsPush2InteractionlikelihoodSim(ActiveInterconnection):
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
        steepness = 0.05
        limit = 250
        magnitude = 45.0
        normalizer = torch.exp(torch.tensor(steepness * limit, device=self.device, dtype=self.dtype))
        k = 0.2  # sharpness of the near/far transition

        def connection_func(object_position, pusher_point, interaction_likelihood):
            theta = object_position[2:3] + torch.pi / 2.0
            vec_x = magnitude * torch.cos(theta)
            vec_y = magnitude * torch.sin(theta)
            object_center = object_position[:2] + torch.cat([vec_x, vec_y], dim=0)
            dist = gradient_preserving_clipping(torch.norm(object_center - pusher_point), 0.0, limit)
            transition = torch.sigmoid(k * (dist - 180))

            near_part = 1.0 - 1e-20 * dist
            far_part = gradient_preserving_clipping(normalizer - torch.exp(steepness * dist), 0.05, 1.0)

            likelihood = (1 - transition) * near_part + transition * far_part
            return interaction_likelihood - likelihood

        return connection_func
