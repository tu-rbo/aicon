# -*- coding: utf-8 -*-
"""
Base for all Components in AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import traceback
from abc import abstractmethod
from threading import Lock
from typing import Dict, Type, List, Callable, Tuple, Union

import torch
from tensordict import TensorDict

from aicon_base.connections.connections import ActiveInterconnection
from aicon_base.derivatives.derivative_maintenance import DerivativeDict
from aicon_base.components.goals import Goal
from aicon_base.timestamps import TIME_DTYPE, as_timestamp


class Component:
    """Base class of all AICON components.

    A component owns a set of quantities (`cls_quantity_names_and_shapes`, stored in `self.quantities`) and is linked
    to other components through active interconnections (`connections`). Components only exchange information through
    these connections: after each update a component writes its quantities and derivatives into all of its
    connections, and reads the quantities of the others from them.

    Components are created by builder functions `lambda mockbuild: Component(...)` (see the experiment specifications
    of the examples). `connections` and `goals` map names to builders `lambda device, dtype, mockbuild: ...`. With
    `mockbuild=True` a component is only built for its structure (e.g. a component that runs on another machine).
    """

    name: str
    lock: Lock
    dtype: torch.dtype
    device: torch.device
    connections: Dict[str, Type[ActiveInterconnection]]
    quantities: TensorDict
    timestamp: torch.Tensor
    goals: Dict[str, Type[Goal]]
    partial_derivatives: TensorDict
    derivatives_forward_mode: DerivativeDict
    derivatives_backward_mode: DerivativeDict
    finished_forward_derivatives: DerivativeDict
    other_synchronization_functions: List[Callable]
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = dict()
    local_quantity_names_and_shapes: Dict[str, Tuple[int]]

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None,
        debug_mode: bool = False,
    ):
        self.name = name
        self.lock = Lock()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.connections = {k: connections[k](device, dtype, mockbuild) for k in connections.keys()}
        self.debug_mode = debug_mode
        if goals is None:
            self.goals = dict()
        else:
            self.goals = {k: goals[k](device, dtype, mockbuild) for k in goals.keys()}
        self.quantities = TensorDict(dict(), device=device, batch_size=())
        self.timestamp = torch.zeros(1, dtype=TIME_DTYPE)
        self.partial_derivatives = TensorDict(dict(), device=device, batch_size=())
        self.derivatives_forward_mode = DerivativeDict()
        self.derivatives_backward_mode = DerivativeDict()
        self.finished_forward_derivatives = DerivativeDict()
        self.other_synchronization_functions = list()
        self.initial_definitions(quantity_names_and_shapes)
        self.initialized_all_quantities = False
        self.mocked = mockbuild

    def get_dt(self, new_time: torch.tensor) -> torch.tensor:
        return (new_time - self.timestamp).to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def initialize_quantities(self) -> bool:
        """Set initial values for the quantities, typically from connected quantities. Return False to try again on the
        next update."""
        return False

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None) -> None:
        if quantity_names_and_shapes is None:
            quantity_names_and_shapes = self.cls_quantity_names_and_shapes
        for n, s in quantity_names_and_shapes.items():
            self.quantities[n] = torch.zeros(s, dtype=self.dtype, device=self.device)
        self.local_quantity_names_and_shapes = quantity_names_and_shapes

    def update(self, new_time: torch.tensor, zero_derivatives_for_this_estimator_after_update: bool = False) -> None:
        if self.mocked:
            return
        new_time = as_timestamp(new_time)
        if self.debug_mode:
            print(str(self.name))
            print("Derivatives: (Backward, Pre-Update)", self.derivatives_backward_mode.full_derivative_string())
        if not self.initialized_all_quantities:
            self._attempt_init(new_time)
        else:
            self._attempt_update(new_time)
        if self.debug_mode:
            print(
                str(self.name),
                ":\n\tState:\n",
                *["\t\t" + str(k) + ":" + str(v) + "\n" for k, v in self.quantities.items()],
            )
            if len(self.partial_derivatives.keys()) > 0:
                print(
                    "Partials: ",
                    [(k, v) for k, v in self.partial_derivatives.items(leaves_only=True, include_nested=True)],
                )
            if len(self.derivatives_forward_mode.keys()) > 0:
                print("Derivatives (Forward): ", self.derivatives_forward_mode.full_derivative_string())
            if len(self.derivatives_backward_mode.keys()) > 0:
                print("Derivatives: (Backward)", self.derivatives_backward_mode.full_derivative_string())
        if zero_derivatives_for_this_estimator_after_update:
            self.zero_derivatives_for_this_estimator()
        self._attempt_synch(new_time)
        self.remove_local_input_grad_copies()

    def _attempt_synch(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.timestamp = new_time
                self.synchronize()
        except Exception:
            print("Could not synchronize " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _attempt_update(self, new_time: torch.tensor) -> None:
        raise NotImplementedError()

    def zero_derivatives_for_this_estimator(self):
        my_quantities = list(self.quantities.keys())
        for c in self.connections.values():
            with c.lock:
                c.derivatives_forward_mode.zero_derivatives(my_quantities)
                c.derivatives_backward_mode.zero_derivatives(my_quantities)
        self.derivatives_forward_mode.zero_derivatives(my_quantities)
        self.derivatives_backward_mode.zero_derivatives(my_quantities)

    def remove_local_input_grad_copies(self) -> None:
        pass

    def synchronize(self) -> None:
        for c in self.connections.values():
            try:
                with c.lock:
                    c.derivatives_forward_mode.update_(self.derivatives_forward_mode)
                    c.derivatives_backward_mode.update_(self.derivatives_backward_mode)
                    for k, v in self.quantities.items():
                        c.connected_quantities[k] = v
                        c.connected_quantities_timestamps[k] = self.timestamp
                        c.connected_quantities_initialized[k] = self.initialized_all_quantities
            except (ValueError, KeyError):
                print("During Synchronization with Connection " + str(c.name) + " encountered following Issue:")
                traceback.print_exc()
                print("New Updated Values from Component:")
                print(self.quantities)
                print("Connection Previous Value:")
                with c.lock:
                    print(c.connected_quantities)
        for sync_func in self.other_synchronization_functions:
            sync_func(self)

    def _attempt_init(self, new_time: torch.tensor) -> None:
        print(self.name + " not initialized yet")
        try:
            with self.lock:
                self.initialized_all_quantities = self.initialize_quantities()
        except Exception:
            print("Could not initialize " + str(self.name) + ":")
            traceback.print_exc()
