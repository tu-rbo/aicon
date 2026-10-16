# -*- coding: utf-8 -*-
"""
Backwards projections

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

from copy import copy, deepcopy
from itertools import accumulate
from typing import List

import torch

from aicon_base.derivatives.derivative_maintenance import DerivativeDict, QuantityRelatedDerivativeBlock
from aicon_base.timestamps import as_timestamp, stamps_tensor


def apply_reduction_factors(reduction_factors, q_name, traces, grad_norm, reducer, recovery_alpha, recovery_exponent):
    """
    Per-path conflict inhibition factors, keyed by gradient trace so that the state follows a path even while it
    is temporarily stale. Factors below 1 recover towards 1 on every call and are dropped once they reach it.
    """
    stored = reduction_factors.setdefault(q_name, {})
    if stored:
        keys = list(stored.keys())
        v = torch.tensor([stored[k] for k in keys], dtype=grad_norm.dtype, device=grad_norm.device)
        mask = v < 1.0
        v[mask] = v[mask] + reducer / recovery_alpha * torch.pow(v[mask], recovery_exponent) * (1 - v[mask])
        v[torch.logical_not(mask)] = 1.0
        stored.clear()
        for k, val in zip(keys, v.tolist()):
            if val < 1.0:
                stored[k] = val
    return torch.tensor([stored.get(tr, 1.0) for tr in traces], dtype=grad_norm.dtype, device=grad_norm.device)


def combine_backwards_derivatives(
    backward_derivatives: DerivativeDict,
    relevant_quantities: List[str],
    time_threshold: torch.Tensor,
    output_buffer,
    current_conflicts,
    conflict_entry_points,
    reduction_factors,
    dominant_idxs,
    nullspace_escape_trigger_length=0.5,
    nullspace_escape_trigger_similarity=-0.7,
    nullspace_escape_end_similarity=-0.55,
    nullspace_escape_end_length=0.3,
    nullspace_escape_phase_out_ticks=10,
    reducer=0.2,
    recovery_alpha=1,
    recovery_exponent=1,
    nullspace_escape_short_window=5,
    nullspace_escape_medium_window=25,
    nullspace_escape_long_window=100,
):
    new_dict = DerivativeDict()
    updated_conflicts = dict()
    for q_name, original_derivative_block in backward_derivatives.items():
        # setup new derivative block with zero grads for all
        new_derivative_block = QuantityRelatedDerivativeBlock(q_name)
        new_derivative_block.derivatives_tensor = torch.clone(original_derivative_block.derivatives_tensor)
        new_derivative_block.quantity_traces = deepcopy(original_derivative_block.quantity_traces)
        new_derivative_block.timestamps = copy(original_derivative_block.timestamps)
        new_derivative_block.mixed_in_quantity_traces = copy(original_derivative_block.mixed_in_quantity_traces)
        new_derivative_block.last_partial_derivatives = original_derivative_block.last_partial_derivatives.clone()

        if q_name not in relevant_quantities:
            continue
        num_paths = len(original_derivative_block.quantity_traces)
        if num_paths == 1:
            new_dict[q_name] = new_derivative_block
            continue

        already_projected_at_all = [
            sum([x[0] == "[" for x in y]) > 0 for y in original_derivative_block.quantity_traces
        ]
        already_projected_at_all = torch.tensor(
            already_projected_at_all, device=original_derivative_block.derivatives_tensor.device, dtype=torch.bool
        )
        already_projected_at_this_level = [y[-2][0] == "[" for y in original_derivative_block.quantity_traces]
        already_projected_at_this_level = torch.tensor(
            already_projected_at_this_level,
            device=original_derivative_block.derivatives_tensor.device,
            dtype=torch.bool,
        )
        grad_norm = torch.norm(original_derivative_block.derivatives_tensor.view(num_paths, -1), dim=1)
        traces = [tuple(qt) for qt in original_derivative_block.quantity_traces]
        local_factors = apply_reduction_factors(
            reduction_factors, q_name, traces, grad_norm, reducer, recovery_alpha, recovery_exponent
        )

        grad_norm = grad_norm * local_factors
        too_old = (stamps_tensor(original_derivative_block.oldest_stamps()) < as_timestamp(time_threshold)).to(
            grad_norm.device
        )
        grad_norm[too_old] = 0.0

        if torch.sum(grad_norm) == 0.0:
            continue
        idx_sorted = torch.argsort(grad_norm, descending=True)
        dominant_trace = traces[int(idx_sorted[0])]
        if q_name in dominant_idxs:
            if dominant_trace != dominant_idxs[q_name]:
                reduction_factors[q_name][dominant_idxs[q_name]] = 1.0 - reducer
        dominant_idxs[q_name] = dominant_trace

        new_derivative_block = escape_conflicts(
            current_conflicts,
            grad_norm,
            new_derivative_block,
            original_derivative_block,
            output_buffer,
            q_name,
            updated_conflicts,
            idx_sorted,
            conflict_entry_points,
            escape_trigger_length=nullspace_escape_trigger_length,
            escape_trigger_similarity=nullspace_escape_trigger_similarity,
            escape_end_similarity=nullspace_escape_end_similarity,
            escape_end_length=nullspace_escape_end_length,
            escape_phase_out_ticks=nullspace_escape_phase_out_ticks,
            short_window=nullspace_escape_short_window,
            medium_window=nullspace_escape_medium_window,
            long_window=nullspace_escape_long_window,
        )

        new_derivative_block = generate_projected_gradients_local_addition(
            already_projected_at_this_level,
            grad_norm,
            new_derivative_block,
            too_old,
            new_derivative_block,
            q_name,
            time_threshold,
            idx_sorted,
        )

        new_dict[q_name] = new_derivative_block
    return new_dict, updated_conflicts, reduction_factors, dominant_idxs, conflict_entry_points


def get_main_grads_and_projected_to(derivative_block):
    projection_strings = [[x[0] == "[" for x in y] for y in derivative_block.quantity_traces]
    projection_to = []
    main_grad_traces = []
    for i, string_list in enumerate(projection_strings):
        if sum(string_list) == 0:
            projection_to.append([])
            main_grad_traces.append(derivative_block.quantity_traces[i])
        else:
            this_projected_to = filter(lambda x: string_list[x[0]], enumerate(derivative_block.quantity_traces[i]))
            this_projected_to = list(map(lambda x: x[1][1:-1].split("-"), this_projected_to))
            projection_to.append(this_projected_to)
            this_main_grad_trace = filter(
                lambda x: not string_list[x[0]], enumerate(derivative_block.quantity_traces[i])
            )
            this_main_grad_trace = list(map(lambda x: x[1], this_main_grad_trace))
            main_grad_traces.append(this_main_grad_trace)
    return main_grad_traces, projection_to


def generate_projected_gradients_local_addition(
    already_projected_at_this_level,
    grad_norm,
    new_derivative_block,
    too_old,
    original_derivative_block,
    q_name,
    time_threshold,
    idx_sorted,
):
    (
        collected_grads,
        collected_mix_ins,
        collected_timestamps,
        collected_traces,
        collected_main_grad_traces,
        projection_scaling,
    ) = exploitative_projected_gradients_computation(
        already_projected_at_this_level, grad_norm, idx_sorted, too_old, original_derivative_block, time_threshold
    )
    idx = idx_sorted[0]
    new_grad = original_derivative_block.derivatives_tensor[idx]
    trace = deepcopy(original_derivative_block.quantity_traces[idx])
    if len(collected_grads) != 0:
        # the projected gradients are flattened, so flatten the dominant one the same way
        collected_grads.append(new_grad.reshape(1, -1))
        projection_scaling.append(1.0)
        norms = [torch.norm(t).unsqueeze(0) for t in collected_grads]
        overall_norm = torch.sum(torch.cat(norms))
        temp = 0.8
        softmax_weights = torch.softmax(
            torch.cat(norms) / overall_norm / temp, dim=0
        )  # TODO scale weights according to projection
        collected_mix_ins.append(copy(original_derivative_block.mixed_in_quantity_traces[idx]))
        collected_grads = [
            t.unsqueeze(0) / norms[i] * softmax_weights[i] * projection_scaling[i]
            for i, t in enumerate(collected_grads)
        ]
        combined_grad = torch.sum(torch.cat(collected_grads), dim=0, keepdim=True)
        # TODO proper mix ins
        projected_derivative_block = QuantityRelatedDerivativeBlock(q_name)
        projected_derivative_block.derivatives_tensor = torch.reshape(
            combined_grad / torch.norm(combined_grad) * overall_norm,
            (1, *original_derivative_block.derivatives_tensor.shape[1:]),
        )

        collected_trace_string = "(" + ",".join(["-".join(x) for x in collected_traces]) + ")"
        new_trace = trace[:-1] + [collected_trace_string] + trace[-1:]
        projected_derivative_block.quantity_traces = [
            new_trace,
        ]
        projected_derivative_block.timestamps = [
            copy(original_derivative_block.timestamps[idx]),
        ]
        projected_derivative_block.mixed_in_quantity_traces = [
            copy(original_derivative_block.mixed_in_quantity_traces[idx]),
        ]
        projected_derivative_block.last_partial_derivatives = dict()
        new_derivative_block = projected_derivative_block
    else:
        new_derivative_block = QuantityRelatedDerivativeBlock(q_name)
        new_derivative_block.derivatives_tensor = torch.reshape(
            new_grad.unsqueeze(0), (1, *original_derivative_block.derivatives_tensor.shape[1:])
        )
        new_derivative_block.quantity_traces = [
            trace,
        ]
        new_derivative_block.timestamps = [
            copy(original_derivative_block.timestamps[idx]),
        ]
        new_derivative_block.mixed_in_quantity_traces = [
            copy(original_derivative_block.mixed_in_quantity_traces[idx]),
        ]
        new_derivative_block.last_partial_derivatives = dict()
    return new_derivative_block


def exploitative_projected_gradients_computation(
    already_projected_at_this_level, grad_norm, idx_sorted, too_old, original_derivative_block, time_threshold
):
    idx = idx_sorted[0]
    derivative_tensor = original_derivative_block.derivatives_tensor.reshape(
        len(original_derivative_block.quantity_traces), -1
    )
    out_dims = original_derivative_block.derivatives_tensor.shape[2:]
    out_dim = [*accumulate(out_dims, lambda a, b: a * b)][-1]
    main_grad_traces, projection_to = get_main_grads_and_projected_to(original_derivative_block)
    collected_grads = []
    collected_main_grad_traces = []
    collected_traces = []
    collected_timestamps = []
    collected_mix_ins = []
    projection_scaling = []
    new_grad = derivative_tensor[idx]
    main_grad = main_grad_traces[idx]
    collected_main_grad_traces.append(main_grad)
    while grad_norm[idx] > 0:
        # get projector
        try:
            relevant_output_quantity = original_derivative_block.quantity_traces[idx][-2]
            projector = get_appropriate_projector(
                original_derivative_block, new_grad, out_dim, out_dims, relevant_output_quantity
            )
        except torch._C._LinAlgError as e:
            print("Error while getting a Projector!" + str(e))
            break
        if torch.sum(torch.abs(projector)) == 0:
            # nothing is left to project into (a one-dimensional quantity, or a vanishing gradient): only the
            # gradients collected so far are used
            break

        # do projection and prepare norms
        derivative_tensor = torch.einsum("ij,bj->bi", projector, derivative_tensor)
        projected_grad_norm = torch.norm(derivative_tensor, dim=1)
        projected_grad_norm[too_old] = 0
        projected_grad_norm[idx] = 0
        projected_grad_norm[already_projected_at_this_level] = 0
        for t in collected_main_grad_traces:
            not_projected_on_this_mask = torch.tensor(
                [sum([t[: len(x)] == x for x in y]) == 0 for y in projection_to], device=derivative_tensor.device
            )
            not_projected_on_this_mask = torch.logical_and(
                not_projected_on_this_mask, torch.tensor([len(x) > 0 for x in projection_to])
            )
            projected_grad_norm[not_projected_on_this_mask] = 0
            same_main_grad = torch.tensor([t == x for x in main_grad_traces], device=derivative_tensor.device)
            projected_grad_norm[same_main_grad] = 0
            # one path per entry: a path arriving through the same input (second-to-last trace entry) as a
            # collected one is not added to the stack
            same_penultimate_entry = torch.tensor(
                [t[-2] == x[-2] for x in main_grad_traces], device=derivative_tensor.device
            )
            projected_grad_norm[same_penultimate_entry] = 0
        if torch.sum(projected_grad_norm != 0) == 0:
            break
        idx_old = idx
        idx = torch.argmax(projected_grad_norm)
        if projected_grad_norm[idx] < grad_norm[idx] * 0.05:
            # less than 5% remaining grad means most likely a numerical result above 0 that should be 0
            # for example in two dimensions, having already two projections, there cannot be a >0 result
            # but there numerically often is. We do it based on percentage to not take into account overall
            # magnitude
            break
        if grad_norm[idx] < grad_norm[idx_old] * 0.000001:
            # signal difference so strong that it is better to reject
            break
        new_grad = derivative_tensor[idx]
        main_grad = main_grad_traces[idx]
        true_trace = deepcopy(original_derivative_block.quantity_traces[idx])
        for t in collected_main_grad_traces:
            string_version = "[" + "-".join(t) + "]"
            true_trace.insert(-1, string_version)
        collected_traces.append(true_trace)
        collected_grads.append(
            new_grad.unsqueeze(0) / projected_grad_norm[idx] * grad_norm[idx]
        )  # rescaling for original contribution
        projection_scaling.append(projected_grad_norm[idx] / grad_norm[idx])
        collected_main_grad_traces.append(main_grad)
        collected_timestamps.append(copy(original_derivative_block.timestamps[idx]))
        collected_mix_ins.append(copy(original_derivative_block.mixed_in_quantity_traces[idx]))
    return (
        collected_grads,
        collected_mix_ins,
        collected_timestamps,
        collected_traces,
        collected_main_grad_traces,
        projection_scaling,
    )


def compute_multiscale_progress(output_buffer, q_name, short_window=3, medium_window=10, long_window=30, eps=1e-12):
    """
    Dimensionless progress at multiple timescales, measured at the quantity q where
    the conflict arises. output_buffer[q_name] is filled by EstimationComponent with
    append, so [0] is the OLDEST buffered value and [-1] the newest. Each window
    covers the oldest `window + 1` entries, so the short window looks at the start of
    the buffer, not at the most recent motion. The buffer holds output_buffer_length
    entries (30 by default); windows of that length or longer all cover the whole
    buffer.

    Progress is path efficiency: net displacement across the window divided by the
    total distance travelled within it.

        1.0  -> motion is straight; the system is going somewhere
        0.5  -> neutral
        0.0  -> motion returns to where it started: oscillation, limit cycle, stall

    Invariant under isotropic rescaling q -> c*q, so the same `sensitivity` and the
    same windows transfer between domains with different units. Windows with fewer
    than two samples return the neutral value 0.5 (no evidence either way).
    """
    values = torch.stack(list(output_buffer[q_name]))
    values = values.reshape(values.shape[0], -1)

    def efficiency(window):
        w = min(window, values.shape[0] - 1)
        if w < 1:
            return values.new_full((), 0.5)
        seg = values[: w + 1]  # oldest first
        net = torch.norm(seg[0] - seg[-1])
        path = torch.sum(torch.norm(seg[:-1] - seg[1:], dim=-1))
        return net / (path + eps)  # 0/eps = 0 when frozen

    return efficiency(short_window), efficiency(medium_window), efficiency(long_window)


def adapt_thresholds(
    short_progress,
    medium_progress,
    long_progress,
    in_conflict,
    base_trigger_similarity=-0.7,
    base_trigger_length=0.5,
    base_end_similarity=-0.55,
    base_end_length=0.3,
    sensitivity=1.0,
    max_trigger_shift=0.5,
    max_end_shift=0.3,
    max_trigger_length_frac=0.4,
    max_end_length_frac=0.5,
    neutral=0.5,
    sim_bounds=(-0.95, -0.05),
):
    """
    sensitivity=0 -> every returned threshold is exactly its base value.
    sensitivity=1 -> full deviation range at extreme progress (max_*_shift / _frac).
    Progress inputs are dimensionless efficiencies in [0, 1] (see above).

    Deviations are signed: efficiency below `neutral` pushes one way, above it the
    other. Clamps are widened to always contain the base value, so they can never
    silently override the sensitivity=0 identity.
    """

    def drive(p):  # [0,1] -> [-1,1], 0 at `neutral`
        return torch.clamp((p - neutral) / neutral, -1.0, 1.0)

    if not in_conflict:
        # Moving but not getting anywhere -> raise trigger_similarity toward 0 and
        # lower trigger_length: enter exploration earlier. Clean progress does the
        # opposite, making entry harder.
        d = -drive(medium_progress)

        trigger_similarity = base_trigger_similarity + sensitivity * max_trigger_shift * d
        trigger_length = base_trigger_length * (1.0 - sensitivity * max_trigger_length_frac * d)

        lo = min(sim_bounds[0], base_trigger_similarity)
        hi = max(sim_bounds[1], base_trigger_similarity)
        trigger_similarity = torch.clamp(trigger_similarity, min=lo, max=hi)
        trigger_length = torch.clamp(trigger_length, min=min(1e-3, base_trigger_length))

        return trigger_similarity, trigger_length, base_end_similarity, base_end_length

    # In exploration. Stalling holds us in; escaping and sustained departure let us out.
    stall = -drive(short_progress)
    escape = drive(medium_progress)
    sustained = drive(long_progress)

    d = torch.clamp(0.4 * escape + 0.2 * sustained - 0.4 * stall, -1.0, 1.0)

    end_similarity = base_end_similarity + sensitivity * max_end_shift * d
    end_length = base_end_length * (1.0 + sensitivity * max_end_length_frac * d)

    lo = min(sim_bounds[0], base_end_similarity)
    hi = max(sim_bounds[1], base_end_similarity)
    end_similarity = torch.clamp(end_similarity, min=lo, max=hi)
    end_length = torch.clamp(end_length, min=min(1e-3, base_end_length))

    return base_trigger_similarity, base_trigger_length, end_similarity, end_length


def escape_conflicts(
    current_conflicts,
    grad_norm,
    new_derivative_block,
    original_derivative_block,
    output_buffer,
    q_name,
    updated_conflicts,
    idx_sorted,
    conflict_entry_points,
    escape_trigger_length=0.5,
    escape_trigger_similarity=-0.7,
    escape_end_similarity=-0.55,
    escape_end_length=0.3,
    escape_phase_out_ticks=10,
    short_window=3,
    medium_window=10,
    long_window=30,
):
    gradients = original_derivative_block.derivatives_tensor.reshape(len(original_derivative_block.quantity_traces), -1)
    similarity = torch.nn.functional.cosine_similarity(gradients[idx_sorted[0]], gradients[idx_sorted[1]], dim=-1)

    # --- Multi-scale progress ---
    if q_name not in output_buffer:
        return original_derivative_block
    # [0] is the oldest buffered value (see compute_multiscale_progress); it is what gets recorded as the conflict
    # entry point.
    current_q = output_buffer[q_name][0]
    short_p, medium_p, long_p = compute_multiscale_progress(
        output_buffer, q_name, short_window, medium_window, long_window
    )

    in_conflict = q_name in current_conflicts

    trigger_similarity, trigger_length, end_similarity, end_length = adapt_thresholds(
        short_p,
        medium_p,
        long_p,
        in_conflict=in_conflict,
        base_trigger_similarity=escape_trigger_similarity,
        base_trigger_length=escape_trigger_length,
        base_end_similarity=escape_end_similarity,
        base_end_length=escape_end_length,
    )

    # --- Conflict detection (same logic, adaptive thresholds) ---
    conflict = False
    if in_conflict and not (
        similarity > end_similarity or grad_norm[idx_sorted[0]] * end_length > grad_norm[idx_sorted[1]]
    ):
        updated_conflicts[q_name] = 0
        conflict = True
    elif similarity < trigger_similarity and grad_norm[idx_sorted[0]] * trigger_length < grad_norm[idx_sorted[1]]:
        updated_conflicts[q_name] = 0
        conflict = True
        # Record entry point when we first enter conflict
        if conflict_entry_points is not None and q_name not in conflict_entry_points:
            conflict_entry_points[q_name] = current_q.clone()
    elif in_conflict:
        out_of_conflict_counter = current_conflicts[q_name]
        if out_of_conflict_counter < escape_phase_out_ticks:
            conflict = True
            updated_conflicts[q_name] = out_of_conflict_counter + 1
        else:
            # Fully exited — clear entry point
            if conflict_entry_points is not None and q_name in conflict_entry_points:
                del conflict_entry_points[q_name]

    if conflict:
        # Escape direction: from the buffered value farthest from the oldest one back to the oldest, projected
        # orthogonal to the dominant gradient. It replaces the dominant path's gradient, so descending it continues
        # the quantity's recent motion sideways to the dominant goal (e.g. sliding along a wall).
        last_values = torch.cat([i.unsqueeze(0) for i in output_buffer[q_name]], dim=0)
        curr_diff_vectors = output_buffer[q_name][0].unsqueeze(0) - last_values
        right_idx = torch.argmax(torch.norm(curr_diff_vectors, dim=1))
        curr_diff_vector = curr_diff_vectors[right_idx]
        relevant_output_quantity = original_derivative_block.quantity_traces[idx_sorted[0]][-2]
        new_grad = original_derivative_block.derivatives_tensor[idx_sorted[0]]
        out_dims = original_derivative_block.derivatives_tensor.shape[2:]
        out_dim = [*accumulate(out_dims, lambda a, b: a * b)][-1]
        projector = get_appropriate_projector(
            original_derivative_block, new_grad, out_dim, out_dims, relevant_output_quantity
        )
        projected_diff = torch.einsum("ij,j->i", projector, curr_diff_vector.reshape(-1)).view(
            original_derivative_block.derivatives_tensor.shape[1:]
        )
        diff_norm = torch.norm(projected_diff)
        # Without motion sideways to the dominant gradient there is no escape direction: this tick passes the
        # gradients on unchanged, the conflict stays active and the escape starts once the quantity moves.
        if diff_norm > 0:
            new_derivative_block.derivatives_tensor = torch.zeros_like(original_derivative_block.derivatives_tensor)
            new_derivative_block.derivatives_tensor[idx_sorted[0]] = (
                projected_diff / diff_norm * grad_norm[idx_sorted[0]] * 1.05
            )
            new_derivative_block.quantity_traces[idx_sorted[0]].insert(
                -1, "ConflictResolution(" + "-".join(new_derivative_block.quantity_traces[idx_sorted[1]]) + ")"
            )

    return new_derivative_block


def get_appropriate_projector(derivative_block, new_grad, out_dim, out_dims, relevant_output_quantity):
    use_nullspace = False
    use_qr_nullspace = False
    use_orthogonalspace = True
    if use_nullspace or use_qr_nullspace:
        # get correct Jacobian (if exists)
        if relevant_output_quantity not in derivative_block.last_partial_derivatives.keys():
            return get_fallback_diag_projector(new_grad)
        relevant_partial = derivative_block.last_partial_derivatives[relevant_output_quantity]
        in_dims = relevant_partial.shape[: -len(out_dims)]
        in_dim = [*accumulate(in_dims, lambda a, b: a * b)][-1]
        relevant_partial_2dim = torch.reshape(relevant_partial, shape=(in_dim, out_dim))

        if use_nullspace:
            if in_dim > 10 or out_dim > 10:
                # Too big for inversion
                return get_fallback_diag_projector(new_grad)
            else:
                return get_nullspace_projector(relevant_partial_2dim, in_dim, out_dim)
        else:
            return get_qr_nullspce_projector(relevant_partial_2dim, out_dim, new_grad)
    if use_orthogonalspace:
        return get_orthogonalspace_projector(new_grad, out_dim)
    else:
        return get_fallback_diag_projector(new_grad)


def get_orthogonalspace_projector(new_grad, out_dim):
    g_norm = torch.norm(new_grad)
    if g_norm < 1e-12:
        projector = get_fallback_diag_projector(new_grad)
    else:
        g_unit = (new_grad / g_norm).view(-1)
        projector = torch.eye(out_dim, dtype=g_unit.dtype, device=g_unit.device) - torch.ger(g_unit, g_unit)
    return projector


def get_qr_nullspce_projector(relevant_partial_2dim, out_dim, new_grad):
    try:
        # Use QR-based projector instead
        Q, _ = torch.linalg.qr(relevant_partial_2dim.T)
        projector = torch.eye(out_dim) - Q @ Q.T
    except RuntimeError:
        projector = get_fallback_diag_projector(new_grad)
    return projector


def get_fallback_diag_projector(new_grad):
    zero_entries = new_grad.view(-1) == 0
    projector = torch.diag(zero_entries.to(new_grad.dtype))
    return projector


def get_nullspace_projector(relevant_partial_2dim, in_dim, out_dim):
    if in_dim <= out_dim:
        pseudo_inv = torch.einsum(
            "ij,jk->ik",
            relevant_partial_2dim.t(),
            torch.linalg.solve(
                torch.einsum("ij,jk->ik", relevant_partial_2dim, relevant_partial_2dim.t()),
                torch.eye(relevant_partial_2dim.shape[0]),
            ),
        )
    else:
        pseudo_inv = torch.linalg.solve(
            torch.einsum(
                "ij,jk->ik",
                relevant_partial_2dim.t(),
                relevant_partial_2dim,
            ),
            relevant_partial_2dim.t(),
        )
    projector = torch.eye(out_dim) - torch.einsum("ij,jk->ik", pseudo_inv, relevant_partial_2dim)
    return projector
