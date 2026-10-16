"""
Component Graph for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

from .actions import PusherAction
from .connections import build_connections
from .estimators import ContactsEstimatorSim, PusherEstimatorSim, ObjectEstimatorSim, InteractionEstimatorSim
from .goals import Push2GoalSim
from .sensors import KeypointSensor, PusherSensor, COMSensor


def get_building_functions(debug_mode: bool = False):
    connection_builders = build_connections()
    cpu = torch.device("cpu")
    component_builders = {
        "KeypointSensor": lambda mockbuild: KeypointSensor(
            "KeypointSensor",
            connections={
                k: connection_builders[k]
                for k in (
                    "KeyPoints2Position",
                    "KeyPoints2ContactPoints",
                    "ContactPoints2PushVelocityDirections",
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "PusherSensor": lambda mockbuild: PusherSensor(
            "PusherSensor",
            connections={k: connection_builders[k] for k in ("PusherPointMeasurement",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "COMSensor": lambda mockbuild: COMSensor(
            "COMSensor",
            connections={k: connection_builders[k] for k in ("ContactPoints2PushVelocityDirections",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "PusherAction": lambda mockbuild: PusherAction(
            "PusherAction",
            connections={k: connection_builders[k] for k in ("PusherPointAction",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "ContactsEstimator": lambda mockbuild: ContactsEstimatorSim(
            "ContactsEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "KeyPoints2ContactPoints",
                    "ContactPoints2ContactLikelihood",
                    "ContactPoints2PushVelocityDirections",
                    "Contacts2ObjectPosition",
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "PusherEstimator": lambda mockbuild: PusherEstimatorSim(
            "PusherEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "PusherPointAction",
                    "PusherPointMeasurement",
                    "KeyPoints2ContactPoints",
                    "ContactPoints2ContactLikelihood",
                    "ObjectvsPush2InteractionlikelihoodSim",
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "ObjectEstimator": lambda mockbuild: ObjectEstimatorSim(
            "ObjectEstimator",
            connections={
                k: connection_builders[k]
                for k in ("KeyPoints2Position", "Contacts2ObjectPosition", "ObjectvsPush2InteractionlikelihoodSim")
            },
            goals={
                "Push2Goal": lambda d, t, mockbuild=False: Push2GoalSim(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "InteractionEstimatorSim": lambda mockbuild: InteractionEstimatorSim(
            "InteractionEstimator",
            connections={k: connection_builders[k] for k in ("ObjectvsPush2InteractionlikelihoodSim",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
    }
    frame_rates = {k: 10 for k in component_builders}
    return component_builders, connection_builders, frame_rates
