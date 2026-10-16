"""
Active Interconnections for the 2D Navigation Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from .constants import obs_img_size
from aicon_base.connections import ActiveInterconnection


def build_connections():
    connections = {
        "NavMeasurement": lambda device, dtype, mockbuild: NavMeasurement(
            "NavMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "NavAction": lambda device, dtype, mockbuild: NavAction(
            "NavAction", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "NavTargetMeasurement": lambda device, dtype, mockbuild: NavTargetMeasurement(
            "NavTargetMeasurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ObstacleDistConnection": lambda device, dtype, mockbuild: ObstacleDistConnection(
            "ObstacleDistConnection", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "TargetDistConnection": lambda device, dtype, mockbuild: TargetDistConnection(
            "TargetDistConnection", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
    return connections


class NavAction(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"agent_pos": (2,), "action": (2,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(agent_pos, action):
            agent_pos_new = agent_pos + action
            return agent_pos_new

        return connection_func


class NavMeasurement(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"agent_pos": (2,), "agent_meas": (2,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(agent_pos, agent_meas):
            innovation = agent_meas - agent_pos
            return innovation

        return connection_func


class NavTargetMeasurement(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"agent_pos": (2,), "target_meas": (2,), "target_pos": (2,), "target_uncertainty": (2, 2)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(agent_pos, target_meas, target_pos):
            relative_vector = target_pos - agent_pos
            cos_angle = relative_vector[0:1] / torch.norm(relative_vector)
            sin_angle = relative_vector[1:2] / torch.norm(relative_vector)
            innovation = target_meas - torch.cat([cos_angle, sin_angle])
            return innovation

        return connection_func


class TargetDistConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"agent_pos": (2,), "target_pos": (2,), "target_dist": (1,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(agent_pos, target_pos, target_dist):
            relative_vector = target_pos - agent_pos
            target_dist_new = torch.norm(relative_vector)
            innovation = target_dist_new - target_dist
            return innovation

        return connection_func


class ObstacleDistConnection(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"obstacle_image": (obs_img_size, obs_img_size), "agent_pos": (2,), "current_dist2obs": (1,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        # pixel [i, j] of the obstacle image is the cell at position (i, j), as in scenarios.py
        ind_x, ind_y = torch.meshgrid(torch.arange(obs_img_size), torch.arange(obs_img_size), indexing="ij")
        locations = torch.stack((ind_x, ind_y), dim=2).float()
        cov = torch.eye(2) * 0.5

        def connection_func(obstacle_image, agent_pos, current_dist2obs):
            dist_trans = obstacle_image  # need to do it in the sensor because not diffable
            log_probs = torch.distributions.MultivariateNormal(agent_pos, covariance_matrix=cov).log_prob(locations)
            probs = torch.exp(log_probs)
            probs = probs / torch.max(probs).detach()
            probs[probs < 0.05] = 0.0
            probs = probs / (torch.sum(probs).detach() + 0.00000001)
            expected_dist = torch.sum(probs * dist_trans)
            innovation = expected_dist - current_dist2obs
            return innovation

        return connection_func
