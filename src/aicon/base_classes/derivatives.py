from __future__ import annotations
from itertools import accumulate
from typing import List, Union, Iterable, Dict
import torch
from tensordict import TensorDict


class QuantityRelatedDerivativeBlock:
    """
    A block of derivatives related to a specific quantity.
    
    This class manages the computation and tracking of derivatives for a specific
    quantity, including:
    - The derivative tensors themselves
    - Timestamps of when derivatives were computed
    - Traces of quantities involved in the derivative chain
    
    Attributes:
        related_quantity (str): Name of the quantity these derivatives are related to
        derivatives_tensor (Union[torch.Tensor, None]): The actual derivative tensor
        timestamps (List[List[torch.Tensor]]): List of timestamps for each derivative path
        quantity_traces (List[List[str]]): List of quantity names involved in each derivative path
    """
    
    related_quantity: str
    derivatives_tensor: Union[torch.Tensor, None]
    timestamps: List[List[torch.Tensor]]
    quantity_traces: List[List[str]]

    def __init__(self, related_quantity: str):
        """
        Initialize a new derivative block.
        
        Args:
            related_quantity: Name of the quantity these derivatives are related to
        """
        self.related_quantity = related_quantity
        self.derivatives_tensor = None
        self.timestamps = list()
        self.quantity_traces = list()

    def apply_new_partial_derivative(self, partial_derivative : torch.Tensor, related_quantity : str, timestamp : torch.Tensor, required_sign : Union[None, int] = None, prevent_repeats : bool = True, max_length : Union[int, None] = None):
        """
        Apply a new partial derivative to this block.
        
        This method combines the existing derivatives with a new partial derivative,
        handling sign requirements and preventing loops in the derivative chain.
        
        Args:
            partial_derivative: New partial derivative to apply
            related_quantity: Name of the quantity this derivative is related to
            timestamp: Timestamp of when this derivative was computed
            required_sign: Optional sign requirement for the derivative
            prevent_repeats: Whether to prevent repeated quantities in the chain
            max_length: Optional maximum length for the derivative chain
            
        Returns:
            New QuantityRelatedDerivativeBlock with the updated derivatives
            
        Raises:
            AssertionError: If all resulting derivatives are zero
        """
        assert self.derivatives_tensor is not None
        summed_dims = self.derivatives_tensor.shape[2:]
        out_dims = partial_derivative.shape[len(summed_dims):]
        in_dims = self.derivatives_tensor.shape[:2]
        new_derivatives = torch.einsum("bas,so->bao", self.derivatives_tensor.reshape(*in_dims, -1),
                                       partial_derivative.reshape(-1, [*accumulate(out_dims, lambda a, b: a*b)][-1])).reshape(*in_dims, *out_dims)

        if required_sign is not None:
            assert (required_sign == 1 or required_sign == -1)
            # the minus is because we always do gradient descent (and not ascent)
            new_derivatives = - required_sign * torch.abs(new_derivatives)

        sum_dims = [i for i in range(1, len(in_dims) + len(out_dims))] # over all non batch dims
        nonzero_derivative_mask = torch.sum(torch.abs(new_derivatives), dim=sum_dims) > 0
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
                        break    # no more instances
                for idx in all_idxs_of_this:
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
        new_derivative_block.timestamps = [ts + [timestamp.unsqueeze(0)] for i, ts in enumerate(self.timestamps) if relevant_derivative_mask[i]]
        new_derivative_block.quantity_traces = [qt + [related_quantity] for i, qt in enumerate(self.quantity_traces) if
                                               relevant_derivative_mask[i]]
        return new_derivative_block

    def combine_with(self, same_quantity_derivative_blocks : List[QuantityRelatedDerivativeBlock]):
        """
        Combine this block with other blocks for the same quantity.
        
        Args:
            same_quantity_derivative_blocks: List of derivative blocks for the same quantity
            
        Returns:
            New QuantityRelatedDerivativeBlock containing all combined derivatives
            
        Raises:
            AssertionError: If blocks are not for the same quantity
        """
        assert all([self.related_quantity == d.related_quantity for d in same_quantity_derivative_blocks])
        assert self.derivatives_tensor is not None
        new_derivative_block = QuantityRelatedDerivativeBlock(related_quantity=self.related_quantity)
        new_derivative_block.derivatives_tensor = torch.cat([self.derivatives_tensor, *[d.derivatives_tensor for d in same_quantity_derivative_blocks]], dim=0)
        for d in [self, *same_quantity_derivative_blocks]:
            new_derivative_block.timestamps.extend(d.timestamps)
            new_derivative_block.quantity_traces.extend(d.quantity_traces)
        return new_derivative_block

    def update(self, other_derivate_block: QuantityRelatedDerivativeBlock) -> QuantityRelatedDerivativeBlock:
        """
        Update this block with newer derivatives from another block.
        
        Args:
            other_derivate_block: Another derivative block to update from
            
        Returns:
            New QuantityRelatedDerivativeBlock with updated derivatives
            
        Raises:
            AssertionError: If blocks are not for the same quantity
        """
        assert self.related_quantity == other_derivate_block.related_quantity
        new_derivative_block = QuantityRelatedDerivativeBlock(related_quantity=self.related_quantity)
        new_tensors = []
        new_derivative_block, new_tensors = extract_newer_entries(self, other_derivate_block, new_derivative_block,
                                                                 new_tensors)
        for this_idx, qt in enumerate(other_derivate_block.quantity_traces):
            if qt not in self.quantity_traces:
                new_tensors.append(other_derivate_block.derivatives_tensor[this_idx:this_idx + 1])
                new_derivative_block.quantity_traces.append(qt)
                new_derivative_block.timestamps.append(other_derivate_block.timestamps[this_idx])
        new_derivative_block.derivatives_tensor = torch.cat(new_tensors)
        return new_derivative_block

    def update_(self, other_derivate_block: QuantityRelatedDerivativeBlock):
        """
        Update this block in-place with newer derivatives from another block.
        
        Args:
            other_derivate_block: Another derivative block to update from
            
        Raises:
            AssertionError: If blocks are not for the same quantity
        """
        assert self.related_quantity == other_derivate_block.related_quantity
        new_derivative_block = self.update(other_derivate_block)
        self.timestamps = new_derivative_block.timestamps
        self.derivatives_tensor = new_derivative_block.derivatives_tensor
        self.quantity_traces = new_derivative_block.quantity_traces

    def __str__(self):
        """
        Get a string representation of this derivative block.
        
        Returns:
            String representation showing shape, gradients, and timestamps
        """
        if self.derivatives_tensor is not None:
            grad_values = torch.sum(torch.abs(self.derivatives_tensor), dim=[i + 1 for i in range(len(self.derivatives_tensor.shape[1:]))])
            oldest_stamp = [torch.min(torch.cat(ts)) for ts in self.timestamps]
            n_grads = len(self.quantity_traces)
            return str(self.derivatives_tensor.shape) + str([str(n) + ":" + str(v) + ":" +str(t) for n,v,g,t in sorted(zip(self.quantity_traces, [grad_values[i] for i in range(n_grads)], [self.derivatives_tensor[i] for i in range(n_grads)], oldest_stamp), key=lambda x: x[1])])
        else:
            return "EmptyDerivativeBlock"


def extract_newer_entries(this_derivative_block, other_derivate_block, new_derivative_block, new_tensors):
    for this_idx, qt in enumerate(this_derivative_block.quantity_traces):
        try:
            other_idx = other_derivate_block.quantity_traces.index(qt)
        except ValueError:
            new_tensors.append(this_derivative_block.derivatives_tensor[this_idx:this_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(this_derivative_block.timestamps[this_idx])
            continue
        if torch.min(torch.cat(this_derivative_block.timestamps[this_idx])) > torch.min(
                torch.cat(other_derivate_block.timestamps[other_idx])):
            new_tensors.append(this_derivative_block.derivatives_tensor[this_idx:this_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(this_derivative_block.timestamps[this_idx])
        else:
            new_tensors.append(other_derivate_block.derivatives_tensor[other_idx:other_idx + 1])
            new_derivative_block.quantity_traces.append(qt)
            new_derivative_block.timestamps.append(other_derivate_block.timestamps[other_idx])
    return new_derivative_block, new_tensors


class DerivativeDict:
    """
    A dictionary-like container for managing derivative blocks.
    
    This class provides a convenient interface for storing and managing
    multiple QuantityRelatedDerivativeBlock instances, organized by quantity name.
    
    Attributes:
        derivative_blocks (Dict[str, QuantityRelatedDerivativeBlock]): Dictionary mapping quantity names to their derivative blocks
    """
    
    derivative_blocks: Dict[str, QuantityRelatedDerivativeBlock]

    def __init__(self, block_dict: Union[dict[str, QuantityRelatedDerivativeBlock], None] = None):
        if block_dict is None:
            self.derivative_blocks = dict()
        else:
            self.derivative_blocks = block_dict

    def update(self, other_derivative_dict: DerivativeDict) -> DerivativeDict:
        """
        Update this dictionary with entries from another dictionary.
        
        Args:
            other_derivative_dict: Another derivative dictionary to update from
            
        Returns:
            New DerivativeDict with updated entries
        """
        new_derivative_dict = DerivativeDict()
        new_derivative_dict.derivative_blocks = self.derivative_blocks.copy()
        for k, v in other_derivative_dict.derivative_blocks.items():
            if k in new_derivative_dict.derivative_blocks:
                new_derivative_dict.derivative_blocks[k] = new_derivative_dict.derivative_blocks[k].update(v)
            else:
                new_derivative_dict.derivative_blocks[k] = v
        return new_derivative_dict

    def update_(self, other_derivative_dict: DerivativeDict):
        """
        Update this dictionary in-place with entries from another dictionary.
        
        Args:
            other_derivative_dict: Another derivative dictionary to update from
        """
        for k, v in other_derivative_dict.derivative_blocks.items():
            if k in self.derivative_blocks:
                self.derivative_blocks[k].update_(v)
            else:
                self.derivative_blocks[k] = v

    def combine_with(self, other_derivative_dicts: List[DerivativeDict]) -> DerivativeDict:
        """
        Combine this dictionary with multiple other dictionaries.
        
        Args:
            other_derivative_dicts: List of derivative dictionaries to combine with
            
        Returns:
            New DerivativeDict containing all combined entries
        """
        new_derivative_dict = DerivativeDict()
        new_derivative_dict.derivative_blocks = self.derivative_blocks.copy()
        for other_derivative_dict in other_derivative_dicts:
            for k, v in other_derivative_dict.derivative_blocks.items():
                if k in new_derivative_dict.derivative_blocks:
                    new_derivative_dict.derivative_blocks[k] = new_derivative_dict.derivative_blocks[k].combine_with([v])
                else:
                    new_derivative_dict.derivative_blocks[k] = v
        return new_derivative_dict

    @classmethod
    def from_tensor_dict(cls, tensor_dict : TensorDict, timestamp) -> DerivativeDict:
        new_derivtaive_dict = DerivativeDict()
        for k in tensor_dict.keys(include_nested=True, leaves_only=True):
            new_block = QuantityRelatedDerivativeBlock(k[-1])
            new_block.derivatives_tensor = tensor_dict[k].unsqueeze(0)
            new_block.timestamps = [[timestamp.unsqueeze(0)]]
            new_block.quantity_traces = [list(k)]
            new_derivtaive_dict[k[-1]] = new_block
        return new_derivtaive_dict

    def __str__(self):
        return str({k: str(v) for k, v in self.derivative_blocks.items()})

    def items(self):
        return self.derivative_blocks.items()

    def __getitem__(self, item : str):
        return self.derivative_blocks[item]

    def __setitem__(self, key, value):
        self.derivative_blocks[key] = value

    def __contains__(self, item):
        return item in self.derivative_blocks
    


def compute_derivatives(partial_derivatives : TensorDict, relevant_forward_derivatives : DerivativeDict,
                        relevant_backward_derivatives : DerivativeDict,
                        goal_derivatives : TensorDict,
                        input_names_diff:Iterable[str],
                        output_names_diff:Iterable[str],
                        timestamp : torch.Tensor,
                        required_signs_dict : Dict[str, Dict[str, int]],
                        prevent_repeats : bool = True,
                        max_length_trace : Union[int, None] = None):

    new_forward_derivatives = DerivativeDict()
    for i in input_names_diff:
        if not i in relevant_forward_derivatives:
            continue
        this_input_derivatives = relevant_forward_derivatives[i]
        for o in output_names_diff:
            this_partial_derivative = partial_derivatives[o,i]   # TODO transpose?
            required_sign = get_required_sign_if_applicable(i, o, required_signs_dict)
            if torch.norm(this_partial_derivative) <= 0:
                continue
            try:
                new_derivative = this_input_derivatives.apply_new_partial_derivative(this_partial_derivative,
                                                                                     related_quantity=o,
                                                                                     timestamp=timestamp,
                                                                                      required_sign=required_sign,
                                                                                     prevent_repeats=prevent_repeats,
                                                                                     max_length=max_length_trace)
            except AssertionError:
                continue
            if o in new_forward_derivatives:
                new_forward_derivatives[o].update_(new_derivative)
            else:
                new_forward_derivatives[o] = new_derivative

    finished_forward_derivatives = DerivativeDict()
    for o in output_names_diff:
        if not o in new_forward_derivatives:
            continue
        this_forward_deriavtives = new_forward_derivatives[o]
        for k in goal_derivatives.keys():
            this_partial_derivative = torch.transpose(goal_derivatives[k,o], 0, 1)    # TODO transpose?
            if torch.norm(this_partial_derivative) <= 0:
                continue
            try:
                new_derivative = this_forward_deriavtives.apply_new_partial_derivative(this_partial_derivative,
                                                                                                 related_quantity=k,
                                                                                                 timestamp=timestamp,
                                                                                       prevent_repeats=prevent_repeats,
                                                                                     max_length=max_length_trace)
            except AssertionError:
                continue
            if k in finished_forward_derivatives:
                finished_forward_derivatives[k].update_(new_derivative)
            else:
                finished_forward_derivatives[k] = new_derivative

    new_backward_derivatives = DerivativeDict()
    goal_derivative_dict = DerivativeDict.from_tensor_dict(goal_derivatives, timestamp=timestamp)
    relevant_backward_derivatives = relevant_backward_derivatives.update(goal_derivative_dict)
    for o in output_names_diff:
        if not o in relevant_backward_derivatives:
            continue
        this_output_derivatives = relevant_backward_derivatives[o]
        for i in input_names_diff:
            this_partial_derivative = partial_derivatives[o,i]    # TODO transpose?
            if torch.sum(torch.abs((this_partial_derivative))) == 0:
                continue
            required_sign = get_required_sign_if_applicable(i, o, required_signs_dict)
            try:
                new_derivative = this_output_derivatives.apply_new_partial_derivative(this_partial_derivative,
                                                                                                 related_quantity=i,
                                                                                                 timestamp=timestamp,
                                                                                      required_sign=required_sign,
                                                                                      prevent_repeats=prevent_repeats,
                                                                                     max_length=max_length_trace)
            except AssertionError:
                continue
            if i in new_backward_derivatives:
                new_backward_derivatives[i].update_(new_derivative)
            else:
                new_backward_derivatives[i] = new_derivative
    return new_forward_derivatives, new_backward_derivatives, finished_forward_derivatives



def get_steepest_gradient_from_derivative_block(derivative_block : QuantityRelatedDerivativeBlock, time_threshold: torch.Tensor):
    grad_norm = torch.norm(derivative_block.derivatives_tensor.view(len(derivative_block.quantity_traces), -1), dim=1)
    oldest_stamp_for_each_grad = torch.cat([torch.min(torch.cat(stamps)).unsqueeze(0) for stamps in derivative_block.timestamps])
    grad_norm[oldest_stamp_for_each_grad < time_threshold] = 0.0
    steepest_grad_idx = torch.argmax(grad_norm)
    steepest_grad = derivative_block.derivatives_tensor[steepest_grad_idx].squeeze()
    timestamps = derivative_block.timestamps[steepest_grad_idx]
    trace = derivative_block.quantity_traces[steepest_grad_idx]
    return steepest_grad, timestamps, trace



def get_required_sign_if_applicable(input_name : str, output_name : str,
                                    required_signs_dict : Dict[str, Dict[str, int]]) -> Union[None, int]:
    required_sign = None
    if output_name in required_signs_dict:
        if input_name in required_signs_dict[output_name]:
            required_sign = required_signs_dict[output_name][input_name]
    return required_sign
