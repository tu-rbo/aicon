# -*- coding: utf-8 -*-
"""
Processing, Updating, and Computing Derivatives in AICON

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from itertools import accumulate
from typing import Union, Iterable, Dict

import torch
from tensordict import TensorDict

from aicon_base.derivatives.derivative_maintenance import DerivativeDict, QuantityRelatedDerivativeBlock
from aicon_base.derivatives.projections import get_appropriate_projector, compute_multiscale_progress, adapt_thresholds
from aicon_base.timestamps import as_timestamp, stamps_tensor


def compute_derivatives(
    partial_derivatives: TensorDict,
    relevant_forward_derivatives: DerivativeDict,
    relevant_backward_derivatives: DerivativeDict,
    goal_derivatives: TensorDict,
    input_names_diff: Iterable[str],
    output_names_diff: Iterable[str],
    timestamp: torch.Tensor,
    required_signs_dict: Dict[str, Dict[str, int]],
    prevent_repeats: bool = True,
    max_length_trace: Union[int, None] = None,
):
    new_forward_derivatives = DerivativeDict()
    for i in input_names_diff:
        if i not in relevant_forward_derivatives:
            continue
        this_input_derivatives = relevant_forward_derivatives[i]
        for o in output_names_diff:
            this_partial_derivative = partial_derivatives[o, i]  # TODO transpose?
            required_sign = get_required_sign_if_applicable(i, o, required_signs_dict)
            if torch.norm(this_partial_derivative) <= 0:
                continue
            try:
                new_derivative = this_input_derivatives.apply_new_partial_derivative(
                    this_partial_derivative,
                    related_quantity=o,
                    timestamp=timestamp,
                    required_sign=required_sign,
                    prevent_repeats=prevent_repeats,
                    max_length=max_length_trace,
                )
            except AssertionError:
                continue
            if o in new_forward_derivatives:
                new_forward_derivatives[o].update_(new_derivative)
            else:
                new_forward_derivatives[o] = new_derivative

    finished_forward_derivatives = DerivativeDict()
    for o in output_names_diff:
        if o not in new_forward_derivatives:
            continue
        this_forward_deriavtives = new_forward_derivatives[o]
        for k in goal_derivatives.keys():
            this_partial_derivative = torch.transpose(goal_derivatives[k, o], 0, 1)  # TODO transpose?
            if torch.norm(this_partial_derivative) <= 0:
                continue
            try:
                new_derivative = this_forward_deriavtives.apply_new_partial_derivative(
                    this_partial_derivative,
                    related_quantity=k,
                    timestamp=timestamp,
                    prevent_repeats=prevent_repeats,
                    max_length=max_length_trace,
                )
            except AssertionError:
                continue
            if k in finished_forward_derivatives:
                finished_forward_derivatives[k].update_(new_derivative)
            else:
                finished_forward_derivatives[k] = new_derivative

    new_backward_derivatives = DerivativeDict()
    for o in output_names_diff:
        if o not in relevant_backward_derivatives:
            continue
        this_output_derivatives = relevant_backward_derivatives[o]
        for i in input_names_diff:
            this_partial_derivative = partial_derivatives[o, i]
            if torch.sum(torch.abs((this_partial_derivative))) == 0:
                continue
            required_sign = get_required_sign_if_applicable(i, o, required_signs_dict)
            try:
                new_derivative = this_output_derivatives.apply_new_partial_derivative(
                    this_partial_derivative,
                    related_quantity=i,
                    timestamp=timestamp,
                    required_sign=required_sign,
                    prevent_repeats=prevent_repeats,
                    max_length=max_length_trace,
                )
            except AssertionError:
                continue
            if i in new_backward_derivatives:
                new_backward_derivatives[i].update_(new_derivative)
            else:
                new_backward_derivatives[i] = new_derivative
    return new_forward_derivatives, new_backward_derivatives, finished_forward_derivatives


def get_steepest_gradient_from_derivative_block(
    derivative_block: QuantityRelatedDerivativeBlock, time_threshold: torch.Tensor
):
    grad_norm = torch.norm(derivative_block.derivatives_tensor.view(len(derivative_block.quantity_traces), -1), dim=1)
    too_old = stamps_tensor(derivative_block.oldest_stamps()) < as_timestamp(time_threshold)
    grad_norm[too_old.to(grad_norm.device)] = 0.0
    steepest_grad_idx = torch.argmax(grad_norm)
    steepest_grad = derivative_block.derivatives_tensor[steepest_grad_idx].squeeze()
    timestamps = derivative_block.timestamps[steepest_grad_idx]
    trace = derivative_block.quantity_traces[steepest_grad_idx]
    return steepest_grad, timestamps, trace


def get_required_sign_if_applicable(
    input_name: str, output_name: str, required_signs_dict: Dict[str, Dict[str, int]]
) -> Union[None, int]:
    required_sign = None
    if output_name in required_signs_dict:
        if input_name in required_signs_dict[output_name]:
            required_sign = required_signs_dict[output_name][input_name]
    return required_sign


def gradient_descent(gain, last_action, steepest_grad):
    if torch.norm(steepest_grad) > 0 and torch.norm(last_action) > 0:
        # gradient descent
        new_action = last_action / torch.norm(last_action) - gain * steepest_grad / torch.norm(steepest_grad)
    elif torch.norm(steepest_grad) > 0:
        new_action = -steepest_grad / torch.norm(steepest_grad)
    else:
        # no action
        new_action = torch.zeros_like(last_action)
    return new_action


def get_only_max_gradient(t, select_along_dimensions):
    in_dims = tuple(t.shape[:2])
    out_dims = tuple(t.shape[(2 + len(select_along_dimensions)) :])
    combined_in = [*accumulate(in_dims, lambda a, b: a * b)][-1]
    combined_out = 1 if len(out_dims) == 0 else [*accumulate(out_dims, lambda a, b: a * b)][-1]
    combined_selected = (
        1 if len(select_along_dimensions) == 0 else [*accumulate(select_along_dimensions, lambda a, b: a * b)][-1]
    )
    simple_t = t.view(combined_in, combined_selected, combined_out)
    norms = torch.sum(torch.abs(simple_t), dim=-1)
    max_norm_idx = torch.argmax(norms, dim=-1)
    selected = torch.gather(simple_t, dim=1, index=max_norm_idx.view(-1, 1, 1).expand(-1, 1, simple_t.size(2)))
    new_t = selected.view(*in_dims, *out_dims)
    return new_t


def project_gradients_with_conflict_resolution(
    t,
    select_along_dimensions,
    curr_diff_vector,
    curr_conflict,
    output_buffer,  # dict: i -> deque of tensors
    conflict_entry_points,  # dict: i -> tensor
    trigger_length=0.5,
    trigger_similarity=-0.7,
    end_similarity=-0.55,
    end_length=0.3,
    out_of_conflict_counter=10,
    short_window=300,
    medium_window=600,
    long_window=1000,
    sensitivity=0.0,
):
    in_dims = tuple(t.shape[:2])
    out_dims = tuple(t.shape[(2 + len(select_along_dimensions)) :])
    combined_in = [*accumulate(in_dims, lambda a, b: a * b)][-1]
    combined_out = 1 if len(out_dims) == 0 else [*accumulate(out_dims, lambda a, b: a * b)][-1]
    combined_selected = (
        1 if len(select_along_dimensions) == 0 else [*accumulate(select_along_dimensions, lambda a, b: a * b)][-1]
    )
    simple_t = t.view(combined_in, combined_selected, combined_out)
    norms = torch.sum(torch.abs(simple_t), dim=-1)
    max_norm_idx = torch.argmax(norms, dim=-1)
    selected = torch.gather(simple_t, dim=1, index=max_norm_idx.view(-1, 1, 1).expand(-1, 1, simple_t.size(2)))
    new_t_basic_max = selected.view(*in_dims, *out_dims)
    new_t_resolved = torch.clone(new_t_basic_max)
    conflict_resolved_idxs = dict()

    # one progress signal for this quantity, shared across all i
    short_p, medium_p, long_p = compute_multiscale_progress(
        {"placeholder": output_buffer}, "placeholder", short_window, medium_window, long_window
    )

    current_q = output_buffer[0]

    for i in range(combined_in):
        local_t = simple_t[i]
        local_norms = norms[i]
        idx_sorted = torch.argsort(local_norms, descending=True, dim=0)
        if i == 1:
            continue

        eff_trigger_similarity, eff_trigger_length, eff_end_similarity, eff_end_length = adapt_thresholds(
            short_p,
            medium_p,
            long_p,
            in_conflict=(i in curr_conflict),
            base_trigger_similarity=trigger_similarity,
            base_trigger_length=trigger_length,
            base_end_similarity=end_similarity,
            base_end_length=end_length,
            sensitivity=sensitivity,
        )

        for j in idx_sorted[1:]:
            similarity = torch.nn.functional.cosine_similarity(local_t[idx_sorted[0]], local_t[j], dim=-1)
            relative_length = local_norms[j] / local_norms[idx_sorted[0]]

            if (i in curr_conflict and similarity < eff_end_similarity and relative_length > eff_end_length) or (
                similarity < eff_trigger_similarity and relative_length > eff_trigger_length
            ):
                new_grad = local_t[idx_sorted[0]]
                projector = get_appropriate_projector(None, new_grad, combined_out, out_dims, None)
                projected_diff = (
                    torch.einsum("ij,j->i", projector, curr_diff_vector.reshape(-1)).view(1, *out_dims) + 0.0000000001
                )
                new_t_resolved[i] = -projected_diff / torch.norm(projected_diff) * local_norms[idx_sorted[0]]
                conflict_resolved_idxs[i] = 0
                # record entry point on fresh conflict entry
                if i not in curr_conflict and conflict_entry_points is not None and current_q is not None:
                    conflict_entry_points[i] = current_q.clone()
                break

        if i in curr_conflict and i not in conflict_resolved_idxs:
            if curr_conflict[i] < out_of_conflict_counter:
                conflict_resolved_idxs[i] = curr_conflict[i] + 1
                new_grad = local_t[idx_sorted[0]]
                projector = get_appropriate_projector(None, new_grad, combined_out, out_dims, None)
                projected_diff = torch.einsum("ij,j->i", projector, curr_diff_vector.reshape(-1)).view(1, *out_dims)
                diff_norm = torch.norm(projected_diff)
                # without sideways motion there is no escape direction; the unresolved gradient is kept
                if diff_norm > 0:
                    new_t_resolved[i] = -projected_diff / diff_norm * local_norms[idx_sorted[0]]
            else:
                # fully exited — clear entry point
                if conflict_entry_points is not None and i in conflict_entry_points:
                    del conflict_entry_points[i]

    return new_t_resolved, conflict_resolved_idxs, conflict_entry_points


def orthogonalize(a, b):
    b_norm_sq = torch.dot(b, b)
    if b_norm_sq == 0:
        raise ValueError("Vector b must not be zero.")

    projection = (torch.dot(a, b) / b_norm_sq) * b
    a_orth = a - projection
    return a_orth
