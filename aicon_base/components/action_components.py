# -*- coding: utf-8 -*-
"""
Action Components in AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"


import traceback
from abc import abstractmethod
from typing import Dict, List, Callable, Tuple, Union

import torch

from aicon_base.components.util import collect_derivatives
from aicon_base.components.base_components import Component
from aicon_base.derivatives.projections import get_main_grads_and_projected_to, combine_backwards_derivatives
from aicon_base.timestamps import TIME_DTYPE, as_timestamp, stamps_tensor


class ActionComponent(Component):
    """A component that commands actions. Implement `initialize_quantities`, `determine_new_actions` (set the action
    quantities, typically by descending `get_steepest_plus_projected(quantity)`, the combined gradient of all goals),
    `send_action_values` and `_start` / `_stop`."""

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        goals: Union[None, Dict[str, Callable]] = None,
        forward_mode: bool = False,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__(
            name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild, debug_mode=debug_mode
        )
        self.active = False
        self.nullspace_escape_trigger_length = 2.0  # only starts to trigger with values < 1
        self.nullspace_escape_trigger_similarity = -2  # only starts to trigger with values  between -1 and 1
        self.nullspace_escape_end_similarity = -0.55
        self.nullspace_escape_end_length = 0.3
        self.nullspace_escape_phase_out_ticks = 10
        self.forward_mode = forward_mode
        self.used_gradients = ([], [])
        # per action quantity: "missing" once reported to have no gradients yet, "available" once it had some
        self._gradient_reports = {}

    def start(self) -> None:
        try:
            self._start()
            self.active = True
        except Exception:
            print("Could not start " + str(self.name) + ":")
            traceback.print_exc()

    def stop(self) -> None:
        try:
            self._stop()
            self.active = False
        except Exception:
            print("Could not stop " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError

    def _report_gradients(self, quantity: str, available: bool) -> None:
        """Says once per quantity that it has no gradients yet and, if it said so, once when they have arrived."""
        status = self._gradient_reports.get(quantity)
        if available and status != "available":
            if status == "missing":
                print(f"{self.name}: gradients for {quantity} available now (t = {float(self.timestamp):.1f})")
            self._gradient_reports[quantity] = "available"
        elif not available and status is None:
            print(f"{self.name}: no gradients for {quantity} yet")
            self._gradient_reports[quantity] = "missing"

    def get_steepest_gradient(
        self, quantity: str, time_threshold: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            self._report_gradients(quantity, available=False)
            return torch.zeros_like(self.quantities[quantity]), [], []
        self._report_gradients(quantity, available=True)

        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=TIME_DTYPE)
        from aicon_base.derivatives.derivative_processing import get_steepest_gradient_from_derivative_block

        steepest_grad, timestamps, trace = get_steepest_gradient_from_derivative_block(my_derivatives, time_threshold)
        self.used_gradients = ([trace], [])
        return steepest_grad, timestamps, trace

    def get_sum_of_gradients(self, quantity: str, time_threshold: Union[torch.Tensor, None] = None) -> torch.Tensor:
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            self._report_gradients(quantity, available=False)
            return torch.zeros_like(self.quantities[quantity])
        self._report_gradients(quantity, available=True)

        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=TIME_DTYPE)
        oldest_stamp_for_each_grad = stamps_tensor(my_derivatives.oldest_stamps())
        recent = (oldest_stamp_for_each_grad > as_timestamp(time_threshold)).to(
            my_derivatives.derivatives_tensor.device
        )
        grad_sum = torch.sum(my_derivatives.derivatives_tensor[recent], dim=0)
        return grad_sum.squeeze()

    def get_sum_of_max_goal_gradients(
        self, quantity: str, time_threshold: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            self._report_gradients(quantity, available=False)
            return torch.zeros_like(self.quantities[quantity])
        self._report_gradients(quantity, available=True)
        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=TIME_DTYPE)
        too_old = (stamps_tensor(my_derivatives.oldest_stamps()) < as_timestamp(time_threshold)).to(self.device)
        goal_options = set()
        goal_list_for_grads = []
        for t in my_derivatives.quantity_traces:
            goal_options.add(t[0])
            goal_list_for_grads.append(t[0])
        steepest_grads = []
        grad_norm = torch.norm(my_derivatives.derivatives_tensor.view(len(my_derivatives.quantity_traces), -1), dim=1)
        for goal in goal_options:
            mask_g = torch.tensor([g != goal for g in goal_list_for_grads], device=self.device)
            mask_combined = torch.logical_or(mask_g, too_old)
            grad_norm_g = torch.clone(grad_norm)
            grad_norm_g[mask_combined] = 0
            if torch.sum(grad_norm_g) == 0:
                steepest_grads.append(torch.zeros_like(self.quantities[quantity]).unsqueeze(0))
            else:
                steepest_grad_idx = torch.argmax(grad_norm_g)
                steepest_grads.append(my_derivatives.derivatives_tensor[steepest_grad_idx].squeeze().unsqueeze(0))
        grad_sum = torch.sum(torch.cat(steepest_grads, dim=0), dim=0)
        return grad_sum

    def get_specific_gradient(
        self, quantity: str, trace: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
        steepest_grad, timestamps = torch.zeros_like(self.quantities[quantity]), []
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            self._report_gradients(quantity, available=False)
            return steepest_grad, timestamps
        self._report_gradients(quantity, available=True)
        for i in range(len(my_derivatives.quantity_traces)):
            if len(my_derivatives.quantity_traces[i]) == len(trace):
                if all(x == y for x, y in zip(trace, my_derivatives.quantity_traces[i])):  # if traces are the same
                    steepest_grad = my_derivatives.derivatives_tensor[i].squeeze()
                    timestamps = my_derivatives.timestamps[i]
                    trace = my_derivatives.quantity_traces[i]
                    return steepest_grad, timestamps
        print("The Gradient with this trace is not available yet:", trace)
        return steepest_grad, timestamps

    def determine_projected(self, quantity: str):
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            self._report_gradients(quantity, available=False)
            return []
        self._report_gradients(quantity, available=True)
        main_grad_traces, projection_to = get_main_grads_and_projected_to(my_derivatives)
        return projection_to, main_grad_traces

    def get_steepest_plus_projected(self, quantity: str, time_threshold: Union[torch.Tensor, None] = None):
        """The combined gradient on `quantity`: the steepest path, with the other paths projected into its nullspace.
        Paths older than `time_threshold` are ignored."""
        try:
            if self.forward_mode:
                # Not supported yet: the combination below only uses backward derivatives, so this path ends in a
                # NameError on relevant_backward_derivatives.
                my_derivatives = self.finished_forward_derivatives[quantity]  # noqa: F841
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
        except KeyError:
            self._report_gradients(quantity, available=False)
            return torch.zeros_like(self.quantities[quantity])
        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=TIME_DTYPE)

        combined_backward_derivatives, _, _, _, _ = combine_backwards_derivatives(
            relevant_backward_derivatives,
            [
                quantity,
            ],
            time_threshold,
            dict(),
            dict(),
            dict(),
            dict(),
            dict(),
            nullspace_escape_trigger_length=self.nullspace_escape_trigger_length,
            nullspace_escape_trigger_similarity=self.nullspace_escape_trigger_similarity,
            nullspace_escape_end_similarity=self.nullspace_escape_end_similarity,
            nullspace_escape_end_length=self.nullspace_escape_end_length,
            nullspace_escape_phase_out_ticks=self.nullspace_escape_phase_out_ticks,
        )

        try:
            combined_grads = combined_backward_derivatives[quantity].derivatives_tensor[0].squeeze()
            self.used_gradients = ([combined_backward_derivatives[quantity].quantity_traces[0]], [])
        except KeyError:
            self._report_gradients(quantity, available=False)
            return torch.zeros_like(self.quantities[quantity])
        self._report_gradients(quantity, available=True)

        return combined_grads

    @abstractmethod
    def determine_new_actions(self) -> None:
        raise NotImplementedError()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.determine_new_actions()
                if self.active:
                    self.send_action_values()
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def send_action_values(self) -> None:
        raise NotImplementedError()
