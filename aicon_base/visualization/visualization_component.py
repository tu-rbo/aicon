"""
Base for Visualization in AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Callable, Dict, Tuple, Union, List

import torch
import traceback
from aicon_base.components.base_components import Component
from aicon_base.components.util import collect_inputs, collect_derivatives
from aicon_base.visualization.network_graphs import (
    get_initial_animatable_figure_for_graph,
    update_fig_with_gradients,
    get_fancy_network_graph,
)


class VisualizationComponent(Component):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        quantity_viz_funcs: Dict[Tuple[str, ...], Callable],
        derivative_viz_funcs_forward: Dict[Tuple[str, ...], Callable],
        derivative_viz_funcs_backward: Dict[Tuple[str, ...], Callable],
        generic_viz_funcs: List[Callable],
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name, connections, dtype=dtype, device=device, mockbuild=True
        )  # we do not need fully constructed connection functions or goal functions
        self.mocked = mockbuild  # but only want fully mocked state if truly commanded to
        self.quantity_viz_funcs = quantity_viz_funcs
        self.derivative_viz_funcs_forward = derivative_viz_funcs_forward
        self.derivative_viz_funcs_backward = derivative_viz_funcs_backward
        self.obtain_gradient_funcs = []
        self.generic_viz_funcs = generic_viz_funcs
        self.name_remappings = {}
        self.simple_combinations = {}
        self.recursive_estimators = {}
        self.type_combinations = {}
        self.summarized_recursive_estimators = []
        self.cut_edges = []
        self.cut_nodes = []
        self.node_name_to_layer = None

    def initialize_quantities(self) -> bool:
        return True  # has nothing to init

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            all_inputs = collect_inputs(self.quantities, self.connections, self.get_dt(new_time))
            all_inputs["timestamp"] = new_time
            all_forward_derivatives, all_backward_derivatives = collect_derivatives(self.connections)
            for quantity_names, f in self.quantity_viz_funcs.items():
                try:
                    inputs = [all_inputs[q] for q in quantity_names]
                except KeyError:
                    continue
                f(*inputs)
            for quantity_names, f in self.derivative_viz_funcs_forward.items():
                try:
                    inputs = [all_forward_derivatives[q] for q in quantity_names]
                except KeyError:
                    continue
                f(*inputs)
            for quantity_names, f in self.derivative_viz_funcs_backward.items():
                inputs = []
                for q in quantity_names:
                    try:
                        inputs.append(all_backward_derivatives[q])
                    except KeyError:
                        continue
                f(*inputs)
            for f in self.generic_viz_funcs:
                f(all_inputs, all_forward_derivatives, all_backward_derivatives)
        except Exception:
            print("Error during visualization:")
            traceback.print_exc()
            return

    def start_network_plotting(
        self,
        component_builders: Dict[str, Callable],
        connection_builders: Dict[str, Callable],
        grad_treshold: Dict[str, float] = None,
    ):
        try:
            G, pos, action_quantity_names, remappings = get_fancy_network_graph(
                component_builders,
                connection_builders,
                name_remapping=self.name_remappings,
                simple_combinations=self.simple_combinations,
                recursive_estimators=self.recursive_estimators,
                type_combinations=self.type_combinations,
                summarized_recursive_estimators=self.summarized_recursive_estimators,
                cut_edges=self.cut_edges,
                cut_nodes=self.cut_nodes,
                node_name_to_layer=self.node_name_to_layer,
            )
            fig, ax, background = get_initial_animatable_figure_for_graph(G, pos)
        except Exception:
            print("Could not start network plotting:")
            traceback.print_exc()
            return

        def update_network_fig(*args):
            local_traces = []
            for func in self.obtain_gradient_funcs:
                local_traces = func()
            update_fig_with_gradients(local_traces, G, pos, fig, ax, background, remappings=remappings)

        with self.lock:
            self.generic_viz_funcs.append(update_network_fig)
