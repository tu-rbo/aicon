# -*- coding: utf-8 -*-
"""
Estimation Components in AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import copy
import inspect
import traceback
from abc import abstractmethod
from collections import deque
from typing import Dict, List, Callable, Tuple, Union

import torch
from tensordict import TensorDict

from aicon_base.components.base_components import Component
from aicon_base.components.util import (
    collect_args,
    collect_inputs,
    collect_derivatives,
    gather_partial_derivatives,
    generate_outputdict,
    collect_shapes,
    collect_required_signs,
)
from aicon_base.derivatives.derivative_maintenance import DerivativeDict, QuantityRelatedDerivativeBlock
from aicon_base.derivatives.derivative_processing import (
    compute_derivatives,
    get_only_max_gradient,
    project_gradients_with_conflict_resolution,
)
from aicon_base.components.goals import evaluate_goals
from aicon_base.derivatives.projections import combine_backwards_derivatives


class EstimationComponent(Component):
    """A recursive estimator: its quantities are updated by a differentiable function f each tick.

    Implement `initialize_quantities` and `define_estimation_function_f`. The derivatives of f are propagated through
    the graph: goal gradients on this component's quantities (and gradients arriving through its connections) are
    combined and passed on to its inputs, until they reach the action components.

    Optional attributes set in `define_estimation_function_f`: `differentiate_against` (names of the f arguments to
    differentiate against; default all except own outputs and "dt"), `alpha` (per-quantity smoothing of the combined
    gradient) and the nullspace escape parameters (see combine_backwards_derivatives).
    """

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        no_backwards_combinations: bool = False,
    ):
        self.no_differentiation = no_differentiation
        self.prevent_loops_in_differentiation = prevent_loops_in_differentiation
        self.max_length_differentiation_trace = max_length_differentiation_trace
        self.no_backwards_combinations = no_backwards_combinations
        self.output_buffer_length = 30
        self.nullspace_escape_trigger_length = 2.0  # only triggers with values < 1
        self.nullspace_escape_trigger_similarity = -2  # only triggers with values between -1 and 1
        self.nullspace_escape_end_similarity = -0.55
        self.nullspace_escape_end_length = 0.3
        self.nullspace_escape_phase_out_ticks = 10
        self.nullspace_escape_short_window = 5
        self.nullspace_escape_medium_window = 25
        self.nullspace_escape_long_window = 100
        self.alpha = dict()
        self.last_combined_backward = dict()
        super().__init__(
            name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild, debug_mode=debug_mode
        )
        if not mockbuild:
            self.differentiate_against = None  # can be overwritten in define_estimation_function_f
            f_func, output_signature_diff, output_signature_non_diff = self.define_estimation_function_f()
            self.f_input_signature = inspect.getfullargspec(f_func).args
            self.f_output_signature_diff = output_signature_diff
            self.f_output_signature_non_diff = output_signature_non_diff
            self.f_func = f_func
            self.goal_values = None
            self.output_buffer = dict()
            self.reduction_factors = dict()
            self.dominant_idxs = dict()
            for k in self.f_output_signature_diff:
                self.output_buffer[k] = deque(maxlen=self.output_buffer_length)
            self.current_conflicts = dict()
            self.conflict_entry_points = dict()

    @abstractmethod
    def define_estimation_function_f(self) -> Tuple[Callable, List[str], List[str]]:
        """Return (f_func, output names to differentiate, all output names).

        f_func's argument names are quantity names: this component's own quantities, the quantities of its connections,
        and "dt". It returns two tuples: the differentiated outputs and all outputs. Typically f uses the connections'
        `c_func`s, e.g. to predict with one connection and correct with another.
        """
        pass

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            self.update_state_and_derivatives(new_time)
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    def remove_local_input_grad_copies(self) -> None:
        my_quantities = list(self.f_input_signature)
        self.derivatives_forward_mode.zero_derivatives(my_quantities)
        self.derivatives_backward_mode.zero_derivatives(my_quantities)

    def update_state_and_derivatives(self, new_time: torch.tensor) -> None:
        with self.lock:
            all_inputs = collect_inputs(self.quantities, self.connections, self.get_dt(new_time))
            relevant_forward_derivatives, relevant_backward_derivatives = collect_derivatives(self.connections)
            required_signs_dict = collect_required_signs(self.connections)
        all_inputs = self.preprocess_inputs(all_inputs)
        args, diff_argnums, input_names_diff = collect_args(
            all_inputs, self.f_input_signature, self.differentiate_against, self.f_output_signature_diff
        )
        outputs, partial_jacs = self.inference_and_differentiation(args, diff_argnums)
        output_dict = generate_outputdict(outputs, self.f_output_signature_non_diff)
        for o_name in self.f_output_signature_diff:
            self.output_buffer[o_name].append(output_dict[o_name])
        backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives = (
            self.derivative_processing(
                all_inputs,
                input_names_diff,
                new_time,
                output_dict,
                partial_jacs,
                relevant_backward_derivatives,
                relevant_forward_derivatives,
                required_signs_dict,
            )
        )
        with self.lock:
            self.quantities = self.quantities.update(output_dict)
            self.partial_derivatives = self.partial_derivatives.update(partial_derivatives)
            self.derivatives_forward_mode = self.derivatives_forward_mode.update(forward_derivatives)
            self.derivatives_backward_mode = self.derivatives_backward_mode.update(backward_derivatives)
            self.finished_forward_derivatives = self.finished_forward_derivatives.update(finished_forward_derivatives)

    def derivative_processing(
        self,
        all_inputs,
        input_names_diff,
        new_time,
        output_dict,
        partial_jacs,
        relevant_backward_derivatives,
        relevant_forward_derivatives,
        required_signs_dict,
    ):
        if self.no_differentiation:
            return DerivativeDict(), DerivativeDict(), DerivativeDict(), DerivativeDict()
        in_out_shapes = collect_shapes(all_inputs, output_dict)
        partial_derivatives = gather_partial_derivatives(
            partial_jacs, input_names_diff, self.f_output_signature_diff, in_out_shapes
        )
        goal_derivatives, self.goal_values = evaluate_goals(self.goals.values(), output_dict)
        goal_derivative_dict = DerivativeDict.from_tensor_dict(goal_derivatives, timestamp=new_time)
        relevant_backward_derivatives = relevant_backward_derivatives.update(goal_derivative_dict)
        for k in goal_derivatives.keys(include_nested=True, leaves_only=True):
            relevant_backward_derivatives[k[-1]].last_partial_derivatives[k[0]] = goal_derivatives[k]
        for k, v in self.goal_values.items():
            if self.goals[k].fulfilled_value is not None:
                if v <= self.goals[k].fulfilled_value:
                    print("Finished Goal " + str(k))
                    self.goals[k].is_active = False
        relevant_backward_derivatives = self.preprocess_derivatives(relevant_backward_derivatives)
        if self.no_backwards_combinations:
            combined_backward_derivatives = copy.copy(relevant_backward_derivatives)
        else:
            (
                combined_backward_derivatives,
                self.current_conflicts,
                self.reduction_factors,
                self.dominant_idxs,
                self.conflict_entry_points,
            ) = combine_backwards_derivatives(
                relevant_backward_derivatives,
                self.f_output_signature_diff,
                new_time - 1.0,
                self.output_buffer,
                self.current_conflicts,
                self.conflict_entry_points,
                self.reduction_factors,
                self.dominant_idxs,
                nullspace_escape_trigger_length=self.nullspace_escape_trigger_length,
                nullspace_escape_trigger_similarity=self.nullspace_escape_trigger_similarity,
                nullspace_escape_end_similarity=self.nullspace_escape_end_similarity,
                nullspace_escape_end_length=self.nullspace_escape_end_length,
                nullspace_escape_phase_out_ticks=self.nullspace_escape_phase_out_ticks,
                nullspace_escape_short_window=self.nullspace_escape_short_window,
                nullspace_escape_medium_window=self.nullspace_escape_medium_window,
                nullspace_escape_long_window=self.nullspace_escape_long_window,
            )
            for q_name in combined_backward_derivatives.keys():
                if q_name in self.last_combined_backward.keys():
                    if q_name in self.alpha:
                        alpha = self.alpha[q_name]
                    else:
                        alpha = 1.0
                    combined_backward_derivatives[q_name].derivatives_tensor = (
                        1 - alpha
                    ) * self.last_combined_backward[q_name].derivatives_tensor + alpha * combined_backward_derivatives[
                        q_name
                    ].derivatives_tensor
            self.last_combined_backward = combined_backward_derivatives
        forward_derivatives, backward_derivatives, finished_forward_derivatives = compute_derivatives(
            partial_derivatives,
            relevant_forward_derivatives,
            combined_backward_derivatives,
            goal_derivatives,
            input_names_diff,
            self.f_output_signature_diff,
            new_time,
            required_signs_dict,
            prevent_repeats=self.prevent_loops_in_differentiation,
            max_length_trace=self.max_length_differentiation_trace,
        )
        forward_derivatives, backward_derivatives, finished_forward_derivatives = self.postprocess_derivatives(
            forward_derivatives,
            backward_derivatives,
            finished_forward_derivatives,
            all_inputs,
            output_dict,
            combined_backward_derivatives,
        )
        return backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives

    def inference_and_differentiation(self, args, diff_argnums):
        if self.no_differentiation:
            _, outputs = self.f_func(*args)
            partial_jacs = tuple()
        else:
            # Reverse mode: estimators typically have many differentiated inputs and a small state.
            partial_jacs, outputs = torch.func.jacrev(self.f_func, argnums=diff_argnums, has_aux=True)(*args)
        return outputs, partial_jacs

    def preprocess_inputs(self, all_inputs: TensorDict) -> TensorDict:
        return all_inputs

    def preprocess_derivatives(self, relevant_backward_derivatives: TensorDict) -> TensorDict:
        return relevant_backward_derivatives

    def postprocess_derivatives(
        self,
        forward_derivatives,
        backward_derivatives,
        finished_forward_derivatives,
        inputs,
        outputs,
        relevant_backward_derivatives,
    ):
        return forward_derivatives, backward_derivatives, finished_forward_derivatives


class BatchedEstimationComponent(EstimationComponent):
    """An estimator whose quantities have a batch dimension (e.g. one entry per candidate contact). The inputs named in
    `to_be_batched_inputs` are repeated along `batch_dimensions` (available to f as "<name>_batched"), and the gradients
    on them are reduced over the batch with `batch_backward_derivative_combination_strategy` ("mean", "sum", "max" or
    "projection")."""

    to_be_batched_inputs: List[str]
    batch_dimensions: Tuple[int]
    batch_backward_derivative_combination_strategy: str

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        no_backwards_combinations: bool = False,
    ):
        self.trigger_length = 0.5
        self.trigger_similarity = -0.7
        self.end_similarity = -0.55
        self.end_length = 0.3
        self.out_of_conflict_counter = 10
        self.input_buffer_length = 1200
        super().__init__(
            name,
            connections,
            goals=goals,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
            no_differentiation=no_differentiation,
            prevent_loops_in_differentiation=prevent_loops_in_differentiation,
            max_length_differentiation_trace=max_length_differentiation_trace,
            no_backwards_combinations=no_backwards_combinations,
        )
        self.input_buffer = dict()
        self.current_batched_conflicts = dict()
        self.current_batched_conflict_entry_points = dict()
        for k in self.to_be_batched_inputs:
            self.input_buffer[k] = deque(maxlen=self.input_buffer_length)
            self.current_batched_conflicts[k] = dict()
            self.current_batched_conflict_entry_points[k] = dict()

    def postprocess_derivatives(
        self,
        forward_derivatives,
        backward_derivatives,
        finished_forward_derivatives,
        inputs,
        outputs,
        relevant_backward_derivatives,
    ):
        for k in self.to_be_batched_inputs:
            self.input_buffer[k].append(inputs[k])
            if k + "_batched" in backward_derivatives:
                t = backward_derivatives[k + "_batched"].derivatives_tensor
                dimensions_to_eliminate = [2 + i for i in range(len(self.batch_dimensions))]
                resolved_conflict_idxs = []
                if self.batch_backward_derivative_combination_strategy == "mean":
                    new_t = torch.nanmean(t, dim=dimensions_to_eliminate)
                elif self.batch_backward_derivative_combination_strategy == "sum":
                    new_t = torch.nansum(t, dim=dimensions_to_eliminate)
                elif self.batch_backward_derivative_combination_strategy == "max":
                    new_t = get_only_max_gradient(t, select_along_dimensions=self.batch_dimensions)
                elif self.batch_backward_derivative_combination_strategy == "projection":
                    if len(self.input_buffer[k]) > 5:
                        last_inputs = torch.cat([i.unsqueeze(0) for i in self.input_buffer[k]], dim=0)
                        last_input_diffs = last_inputs[-1].unsqueeze(0) - last_inputs
                        max_idx = torch.argmax(torch.norm(last_input_diffs, dim=-1))
                        curr_diff_vector = last_input_diffs[max_idx]
                        curr_conflict_dims_with_counters = self.current_batched_conflicts[k]

                        output_buffer = [i.unsqueeze(0) for i in reversed(self.input_buffer[k])]

                        new_t, resolved_conflicts, updated_entry_points = project_gradients_with_conflict_resolution(
                            t,
                            select_along_dimensions=self.batch_dimensions,
                            curr_diff_vector=curr_diff_vector,
                            curr_conflict=curr_conflict_dims_with_counters,
                            conflict_entry_points=self.current_batched_conflict_entry_points[k],
                            output_buffer=output_buffer,
                            trigger_length=self.trigger_length,
                            trigger_similarity=self.trigger_similarity,
                            end_similarity=self.end_similarity,
                            end_length=self.end_length,
                            out_of_conflict_counter=self.out_of_conflict_counter,
                        )
                        self.current_batched_conflicts[k] = resolved_conflicts
                        self.current_batched_conflict_entry_points[k] = updated_entry_points
                        # counter 0: the conflict condition fired this tick, >= 1: phase-out after it stopped firing
                        resolved_conflict_idxs = dict(resolved_conflicts)
                    else:
                        new_t = get_only_max_gradient(t, select_along_dimensions=self.batch_dimensions)
                else:
                    raise ValueError(
                        'Combination strategy "' + self.batch_backward_derivative_combination_strategy + '" is unknown'
                    )
                block = QuantityRelatedDerivativeBlock(k)
                block.derivatives_tensor = new_t
                block.timestamps = backward_derivatives[k + "_batched"].timestamps
                block.quantity_traces = [
                    [k if x == k + "_batched" else x for x in trace]
                    for trace in backward_derivatives[k + "_batched"].quantity_traces
                ]
                block.mixed_in_quantity_traces = [
                    [[k if x == k + "_batched" else x for x in trace] for trace in trace_list]
                    for trace_list in backward_derivatives[k + "_batched"].mixed_in_quantity_traces
                ]
                if len(resolved_conflict_idxs) > 0:
                    for i in resolved_conflict_idxs:
                        block.quantity_traces[i].insert(-2, "ConflictResolution")
                block.last_partial_derivatives = TensorDict({}, batch_size=(), device=self.device)
                backward_derivatives[k] = block
        return forward_derivatives, backward_derivatives, finished_forward_derivatives

    def preprocess_inputs(self, all_inputs: TensorDict) -> TensorDict:
        self.determine_current_batch_dimensions(all_inputs)
        for k in self.to_be_batched_inputs:
            if k in all_inputs.keys():
                t = all_inputs[k]
                all_inputs[k + "_batched"] = t.repeat(*self.batch_dimensions, *[1 for _ in t.shape])
            else:
                raise Warning("Should be batching an input that is currently not in all collected inputs!")
        return all_inputs

    def determine_current_batch_dimensions(self, all_inputs: TensorDict):
        pass

    @abstractmethod
    def define_estimation_function_f(self) -> Tuple[Callable, List[str], List[str]]:
        pass
