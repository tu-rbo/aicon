from typing import Dict, Callable

import torch

from aicon.base_classes.components import Component, ActionComponent
from aicon.base_classes.derivatives import DerivativeDict


def build_components(component_building_functions:Dict[str, Callable]):
    components = {k: v(mockbuild=False) for k, v in component_building_functions.items()}
    connection_instances = dict()
    for component in components.values():
        for name, inst in component.connections.items():
            if name in connection_instances:
                connection_instances[name].append(inst)
            else:
                connection_instances[name] = [inst]
    for component in components.values():
        component.other_synchronization_functions.append(lambda c:
                                                         sync_other_connection_instances(c, connection_instances=connection_instances))
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
                        inst.derivatives_forward_mode.update_(DerivativeDict({quantity_name:component.derivatives_forward_mode[quantity_name]}))
                    if quantity_name in inst.derivatives_backward_mode:
                        with own_inst.lock:
                            own_inst.derivatives_backward_mode.update_(DerivativeDict({quantity_name: inst.derivatives_backward_mode[quantity_name]}))


def run_component_sequence(components: Dict[str, Component], timestamp:torch.tensor, passive_mode:bool=False):
    for c in components.values():
        if isinstance(c, ActionComponent):
            if not c.active and not passive_mode:
                c.start()
        c.update(timestamp)
    return components
