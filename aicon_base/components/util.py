# -*- coding: utf-8 -*-
"""
Utility Functions for Components in AICON

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import TYPE_CHECKING, Tuple, List, Dict, Iterable

import torch
from tensordict import TensorDict

from aicon_base.derivatives.derivative_maintenance import DerivativeDict

if TYPE_CHECKING:
    from aicon_base.connections import ActiveInterconnection


def collect_args(
    all_inputs, f_signature, diff_names, f_output_signature_diff
) -> Tuple[List[torch.Tensor], Tuple[int], List[str]]:
    inputs = all_inputs.select(*f_signature).values()
    if diff_names is None:
        diff_names = [arg for arg in f_signature if arg not in [*f_output_signature_diff, "dt"]]
    unknown = [n for n in diff_names if n not in f_signature]
    if unknown:
        raise ValueError(
            "Cannot differentiate against " + str(unknown) + ", they are not arguments of " + str(f_signature)
        )
    diff_argnums = tuple(f_signature.index(n) for n in diff_names)
    return inputs, diff_argnums, list(diff_names)


def collect_inputs(quantities, connections, dt) -> TensorDict:
    collected_inputs = quantities.clone()
    for c in connections.values():
        with c.lock:
            collected_inputs = collected_inputs.update(c.connected_quantities)
    collected_inputs["dt"] = dt
    return collected_inputs


def collect_derivatives(connections) -> Tuple[DerivativeDict, DerivativeDict]:
    collected_forward = DerivativeDict()
    collected_backward = DerivativeDict()
    for c in connections.values():
        with c.lock:
            collected_forward = collected_forward.update(c.derivatives_forward_mode)
            collected_backward = collected_backward.update(c.derivatives_backward_mode)
    return collected_forward, collected_backward


def gather_partial_derivatives(
    partial_jacs: Dict[int, Dict[int, torch.Tensor]],
    input_names_diff: List[str],
    output_names_diff: List[str],
    in_and_out_shapes: Dict[str, torch.Size],
) -> TensorDict:
    partial_derivatives = TensorDict({}, batch_size=())
    for o_idx, o in enumerate(output_names_diff):
        for i_idx, i in enumerate(input_names_diff):
            partial_jac = partial_jacs[o_idx][i_idx]
            # depending on exact specification, torch can have issues for scalar values
            # the following is error handling for these to correct the shape of the jac
            if in_and_out_shapes[i] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(0)
            if in_and_out_shapes[o] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(-1)
            if sum(partial_jac.shape) == 0:
                partial_jac = partial_jac.view(1, 1)
            partial_derivatives[o, i] = partial_jac
    return partial_derivatives


def generate_outputdict(outputs: Iterable[torch.Tensor], output_names: Iterable[str]) -> TensorDict:
    return TensorDict({n: t for n, t in zip(output_names, outputs)}, batch_size=())


def collect_shapes(all_inputs, output_dict):
    in_out_shapes = dict()
    for k, v in output_dict.items():
        in_out_shapes[k] = v.shape
    for k, v in all_inputs.items():
        in_out_shapes[k] = v.shape
    return in_out_shapes


def collect_required_signs(connections: Dict[str, ActiveInterconnection]) -> Dict[str, Dict[str, int]]:
    required_signs_dict: Dict[str, Dict[str, int]] = dict()
    for c in connections.values():
        for k, v in c.required_signs_dict.items():
            if k in required_signs_dict:
                required_signs_dict[k].update(v)
            else:
                required_signs_dict[k] = v
    return required_signs_dict
