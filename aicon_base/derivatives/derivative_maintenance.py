# -*- coding: utf-8 -*-
"""
Derivative Maintenance in AICON

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from itertools import accumulate
from typing import List, Union

import torch
from tensordict import TensorDict


class QuantityRelatedDerivativeBlock:
    related_quantity: str
    derivatives_tensor: Union[torch.Tensor, None]
    timestamps: List[List[torch.Tensor]]
    quantity_traces: List[List[str]]
    mixed_in_quantity_traces: List[List[List[str]]]
    last_partial_derivatives: Union[TensorDict, None]

    def __init__(self, related_quantity: str):
        self.related_quantity = related_quantity
        self.derivatives_tensor = None
        self.timestamps = list()
        self.quantity_traces = list()
        self.mixed_in_quantity_traces = list()
        self.last_partial_derivatives = None
        self._oldest_stamp_cache = None

    def oldest_stamps(self) -> List[float]:
        """
        Oldest timestamp per path, as plain floats. Cached against the identity and size of `timestamps`, so
        reassigning the list, adding or removing a path, or appending to a path's stamps invalidates the cache.
        """
        stamps = self.timestamps
        signature = (len(stamps), sum(len(s) for s in stamps))
        cache = self._oldest_stamp_cache
        if cache is not None and cache[0] is stamps and cache[1] == signature:
            return cache[2]
        # stamps are single elements in practice; .item() is much cheaper than a torch reduction
        values = [min(t.item() if t.numel() == 1 else float(t.min()) for t in s) for s in stamps]
        self._oldest_stamp_cache = (stamps, signature, values)
        return values

    def apply_new_partial_derivative(
        self,
        partial_derivative: torch.Tensor,
        related_quantity: str,
        timestamp: torch.Tensor,
        required_sign: Union[None, int] = None,
        prevent_repeats: bool = True,
        max_length: Union[int, None] = None,
        prevent_repeats_harsh: bool = True,
    ):
        assert self.derivatives_tensor is not None
        summed_dims = self.derivatives_tensor.shape[2:]
        out_dims = partial_derivative.shape[len(summed_dims) :]
        in_dims = self.derivatives_tensor.shape[:2]
        new_derivatives = torch.einsum(
            "bas,so->bao",
            self.derivatives_tensor.reshape(*in_dims, -1),
            partial_derivative.reshape(-1, [*accumulate(out_dims, lambda a, b: a * b)][-1]),
        ).reshape(*in_dims, *out_dims)

        if required_sign is not None:
            assert required_sign == 1 or required_sign == -1
            # the minus is because we always do gradient descent (and not ascent)
            new_derivatives = -required_sign * torch.abs(new_derivatives)

        sum_dims = [i for i in range(1, len(in_dims) + len(out_dims))]  # over all non batch dims
        nonzero_derivative_mask = (
            torch.sum(torch.abs(new_derivatives), dim=sum_dims) > -0.1
        )  # FIXME trying out what happens if grads always get propagated
        non_repeat_derivative_mask = torch.ones_like(nonzero_derivative_mask, dtype=torch.bool)
        if prevent_repeats:
            for i, qt in enumerate(self.quantity_traces):
                all_idxs_of_this = []
                offset = -1
                while True:
                    try:
                        offset = qt.index(related_quantity, offset + 1)
                        all_idxs_of_this.append(offset)
                    except ValueError:
                        break  # no more instances
                for idx in all_idxs_of_this:
                    if prevent_repeats_harsh:
                        non_repeat_derivative_mask[i] = False
                    if qt[idx - 1] == qt[-1]:
                        non_repeat_derivative_mask[i] = False
        if max_length is not None:
            for i, qt in enumerate(self.quantity_traces):
                if len(qt) > max_length:
                    non_repeat_derivative_mask[i] = False

        relevant_derivative_mask = torch.logical_and(nonzero_derivative_mask, non_repeat_derivative_mask)
        if torch.all(torch.logical_not(relevant_derivative_mask)):
            raise AssertionError("All resulting derivatives are zero!")

        new_derivative_block = QuantityRelatedDerivativeBlock(related_quantity=related_quantity)
        new_derivative_block.derivatives_tensor = new_derivatives.view(*in_dims, *out_dims)[relevant_derivative_mask]
        new_derivative_block.timestamps = [
            ts + [timestamp.unsqueeze(0)] for i, ts in enumerate(self.timestamps) if relevant_derivative_mask[i]
        ]
        new_derivative_block.quantity_traces = [
            qt + [related_quantity] for i, qt in enumerate(self.quantity_traces) if relevant_derivative_mask[i]
        ]
        new_derivative_block.mixed_in_quantity_traces = [
            mt + [list()] for i, mt in enumerate(self.mixed_in_quantity_traces) if relevant_derivative_mask[i]
        ]
        new_derivative_block.last_partial_derivatives = TensorDict(
            {self.related_quantity: partial_derivative}, device=self.derivatives_tensor.device, batch_size=()
        )
        return new_derivative_block

    def combine_with(self, same_quantity_derivative_blocks: List[QuantityRelatedDerivativeBlock]):
        assert all([self.related_quantity == d.related_quantity for d in same_quantity_derivative_blocks])
        assert self.derivatives_tensor is not None
        new_derivative_block = QuantityRelatedDerivativeBlock(related_quantity=self.related_quantity)
        new_derivative_block.derivatives_tensor = torch.cat(
            [self.derivatives_tensor, *[d.derivatives_tensor for d in same_quantity_derivative_blocks]], dim=0
        )
        new_derivative_block.last_partial_derivatives = TensorDict(
            {}, device=self.derivatives_tensor.device, batch_size=()
        )
        for d in [self, *same_quantity_derivative_blocks]:
            new_derivative_block.timestamps.extend(d.timestamps)
            new_derivative_block.quantity_traces.extend(d.quantity_traces)
            new_derivative_block.mixed_in_quantity_traces.extend(d.mixed_in_quantity_traces)
            for k, v in d.last_partial_derivatives.items():
                new_derivative_block.last_partial_derivatives[k] = v
        return new_derivative_block

    def update(self, other_derivate_block: QuantityRelatedDerivativeBlock) -> QuantityRelatedDerivativeBlock:
        assert self.related_quantity == other_derivate_block.related_quantity
        new_derivative_block = QuantityRelatedDerivativeBlock(related_quantity=self.related_quantity)
        new_tensors = []
        new_derivative_block, new_tensors = extract_newer_entries(
            self, other_derivate_block, new_derivative_block, new_tensors
        )
        for this_idx, qt in enumerate(other_derivate_block.quantity_traces):
            if qt not in self.quantity_traces:
                new_tensors.append(other_derivate_block.derivatives_tensor[this_idx : this_idx + 1])
                new_derivative_block.quantity_traces.append(qt)
                new_derivative_block.timestamps.append(other_derivate_block.timestamps[this_idx])
                new_derivative_block.mixed_in_quantity_traces.append(
                    other_derivate_block.mixed_in_quantity_traces[this_idx]
                )
                for k, v in other_derivate_block.last_partial_derivatives.items():
                    if k not in new_derivative_block.last_partial_derivatives.keys():
                        new_derivative_block.last_partial_derivatives[k] = v
        new_derivative_block.derivatives_tensor = torch.cat(new_tensors)
        return new_derivative_block

    def zero_derivatives(self):
        self.derivatives_tensor = torch.zeros_like(self.derivatives_tensor)

    def update_(self, other_derivate_block: QuantityRelatedDerivativeBlock):
        assert self.related_quantity == other_derivate_block.related_quantity
        new_derivative_block = self.update(other_derivate_block)
        self.timestamps = new_derivative_block.timestamps
        self.derivatives_tensor = new_derivative_block.derivatives_tensor
        self.quantity_traces = new_derivative_block.quantity_traces
        self.mixed_in_quantity_traces = new_derivative_block.mixed_in_quantity_traces
        self.last_partial_derivatives = new_derivative_block.last_partial_derivatives

    def __str__(self):
        return self.derivative_representation()

    def derivative_representation(self, stamp=None, show_all_times=False, show_mixins=False):
        if self.derivatives_tensor is not None:
            grad_values = torch.sum(
                torch.abs(self.derivatives_tensor), dim=[i + 1 for i in range(len(self.derivatives_tensor.shape[1:]))]
            )
            if not show_all_times:
                time_info = [torch.min(torch.cat(ts)) for ts in self.timestamps]
            else:
                time_info = [torch.cat(ts) for ts in self.timestamps]
            if stamp is not None:
                time_info = [stamp - s for s in time_info]
            n_grads = len(self.quantity_traces)
            s = str(self.derivatives_tensor.shape) + "\n"
            for n, v, g, t, m in sorted(
                zip(
                    self.quantity_traces,
                    [grad_values[i] for i in range(n_grads)],
                    [self.derivatives_tensor[i] for i in range(n_grads)],
                    time_info,
                    self.mixed_in_quantity_traces,
                ),
                key=lambda x: x[1],
            ):
                if not show_mixins:
                    s += str(n) + ":" + str(v) + ":" + str(t) + "\n"
                else:
                    tmp_s = "["
                    for i, q_n in enumerate(n):
                        tmp_s += "'" + q_n + "'"
                        if len(m[i]) > 0:
                            tmp_s += "("
                            for m_i in m[i]:
                                for m_i_j in m_i:
                                    tmp_s += m_i_j + "-"
                                tmp_s = tmp_s[:-1] + ";"
                            tmp_s = tmp_s[:-1] + ")"
                        tmp_s += ","
                    tmp_s = tmp_s[:-1] + "]"
                    s += tmp_s + ":" + str(v) + ":" + str(t) + "\n"
            return s
        else:
            return "EmptyDerivativeBlock"


def extract_newer_entries(this_derivative_block, other_derivate_block, new_derivative_block, new_tensors):
    new_derivative_block.last_partial_derivatives = TensorDict(
        {}, device=this_derivative_block.derivatives_tensor.device, batch_size=()
    )
    this_oldest = this_derivative_block.oldest_stamps()
    other_oldest = other_derivate_block.oldest_stamps()
    for this_idx, qt in enumerate(this_derivative_block.quantity_traces):
        partial_quantity = qt[-2]
        try:
            other_idx = other_derivate_block.quantity_traces.index(qt)
        except ValueError:
            new_tensors.append(this_derivative_block.derivatives_tensor[this_idx : this_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(this_derivative_block.timestamps[this_idx])
            new_derivative_block.mixed_in_quantity_traces.append(
                this_derivative_block.mixed_in_quantity_traces[this_idx]
            )
            try:
                new_derivative_block.last_partial_derivatives[partial_quantity] = (
                    this_derivative_block.last_partial_derivatives[partial_quantity]
                )
            except KeyError:
                pass
            continue
        if this_oldest[this_idx] > other_oldest[other_idx]:
            new_tensors.append(this_derivative_block.derivatives_tensor[this_idx : this_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(this_derivative_block.timestamps[this_idx])
            new_derivative_block.mixed_in_quantity_traces.append(
                this_derivative_block.mixed_in_quantity_traces[this_idx]
            )
            try:
                new_derivative_block.last_partial_derivatives[partial_quantity] = (
                    this_derivative_block.last_partial_derivatives[partial_quantity]
                )
            except KeyError:
                pass
        else:
            new_tensors.append(other_derivate_block.derivatives_tensor[other_idx : other_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(other_derivate_block.timestamps[other_idx])
            new_derivative_block.mixed_in_quantity_traces.append(
                other_derivate_block.mixed_in_quantity_traces[other_idx]
            )
            try:
                new_derivative_block.last_partial_derivatives[partial_quantity] = (
                    other_derivate_block.last_partial_derivatives[partial_quantity]
                )
            except KeyError:
                pass
    return new_derivative_block, new_tensors


class DerivativeDict:
    block_dict: dict[str, QuantityRelatedDerivativeBlock]

    def __init__(self, block_dict: Union[dict[str, QuantityRelatedDerivativeBlock], None] = None):
        if block_dict is None:
            self.block_dict = dict()
        else:
            self.block_dict = block_dict

    def keys(self):
        return self.block_dict.keys()

    def values(self):
        return self.block_dict.values()

    def items(self):
        return self.block_dict.items()

    def __getitem__(self, item: str):
        return self.block_dict[item]

    def __setitem__(self, key, value):
        self.block_dict[key] = value

    def __contains__(self, item):
        return item in self.block_dict

    def zero_derivatives(self, quantity_names: List[str]):
        for q_name in quantity_names:
            if q_name in self.block_dict:
                self.block_dict[q_name].zero_derivatives()

    def update(self, derivative_dict: DerivativeDict):
        new_dict = DerivativeDict()
        for k, v in self.items():
            if k in derivative_dict:
                new_dict[k] = v.update(derivative_dict[k])
            else:
                new_dict[k] = v
        for k, v in derivative_dict.items():
            if k not in new_dict:
                new_dict[k] = v
        return new_dict

    def update_(self, derivative_dict: DerivativeDict):
        new_derivative_dict = self.update(derivative_dict)
        self.block_dict = new_derivative_dict.block_dict

    def combine_with(self, derivative_dicts: List[DerivativeDict]):
        if len(derivative_dicts) == 0:
            return DerivativeDict(self.block_dict.copy())
        new_block_dict = dict()
        all_keys = set(self.keys()).union(*[set(d.keys()) for d in derivative_dicts])
        for k in all_keys:
            all_blocks = [d[k] for d in [self, *derivative_dicts] if k in d]
            new_block_dict[k] = all_blocks[0].combine_with(all_blocks[1:])
        return DerivativeDict(new_block_dict)

    @classmethod
    def from_tensor_dict(cls, tensor_dict: TensorDict, timestamp) -> DerivativeDict:
        new_derivtaive_dict = DerivativeDict()
        for k in tensor_dict.keys(include_nested=True, leaves_only=True):
            new_block = QuantityRelatedDerivativeBlock(k[-1])
            new_block.derivatives_tensor = tensor_dict[k].unsqueeze(0)
            new_block.timestamps = [[timestamp.unsqueeze(0)]]
            new_block.quantity_traces = [list(k)]
            new_block.mixed_in_quantity_traces = [[[] for _ in list(k)]]
            new_block.last_partial_derivatives = TensorDict({}, device=tensor_dict.device, batch_size=())
            new_derivtaive_dict[k[-1]] = new_block
        return new_derivtaive_dict

    def full_derivative_string(self):
        s = ""
        for k, v in self.items():
            s += k + "\n"
            for i in range(len(v.quantity_traces)):
                s += "\t"
                for j in v.quantity_traces[i]:
                    s += j + "->"
                s = s[:-2] + "\n\t"
                s += str(v.derivatives_tensor[i])
        return s

    def __str__(self):
        return str({k: str(v) for k, v in self.items()})
