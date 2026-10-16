"""
Python Sequential Middleware for AICON (Not Asynchronous)

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Callable

import torch

from aicon_base.components.base_components import Component
from aicon_base.components.action_components import ActionComponent
from aicon_base.derivatives.derivative_maintenance import DerivativeDict
from aicon_base.visualization.visualization_component import VisualizationComponent


def build_components(component_building_functions: Dict[str, Callable]):
    """Build all components and link the instances of each connection between them, for running in one process."""
    components = {k: v(mockbuild=False) for k, v in component_building_functions.items()}
    connection_instances = dict()
    for component in components.values():
        for name, inst in component.connections.items():
            if name in connection_instances:
                connection_instances[name].append(inst)
            else:
                connection_instances[name] = [inst]
    for component in components.values():
        component.other_synchronization_functions.append(
            lambda c: sync_other_connection_instances(c, connection_instances=connection_instances)
        )
    return components


def sync_other_connection_instances(component, connection_instances):
    for name, own_inst in component.connections.items():
        all_instances = connection_instances[name]
        for inst in all_instances:
            if inst == own_inst:
                continue
            with inst.lock:
                for quantity_name, q_value in component.quantities.items():
                    inst.connected_quantities[quantity_name] = q_value
                    inst.connected_quantities_initialized[quantity_name] = True
                    inst.connected_quantities_timestamps[quantity_name] = component.timestamp
                    if quantity_name in component.derivatives_forward_mode:
                        inst.derivatives_forward_mode.update_(
                            DerivativeDict({quantity_name: component.derivatives_forward_mode[quantity_name]})
                        )
                    if quantity_name in inst.derivatives_backward_mode:
                        with own_inst.lock:
                            own_inst.derivatives_backward_mode.update_(
                                DerivativeDict({quantity_name: inst.derivatives_backward_mode[quantity_name]})
                            )
                if isinstance(component, VisualizationComponent):
                    own_inst.derivatives_backward_mode.update_(inst.derivatives_backward_mode)


def run_component_sequence(components: Dict[str, Component], timestamp: torch.tensor, passive_mode: bool = False):
    """Update every component once (starting action components first unless `passive_mode`)."""
    for c in components.values():
        if isinstance(c, ActionComponent):
            if not c.active and not passive_mode:
                c.start()
        c.update(timestamp, zero_derivatives_for_this_estimator_after_update=True)
    return components
