from __future__ import annotations
from typing import Tuple, List, Dict, Iterable

import torch
from tensordict import TensorDict

# from aicon import ActiveInterconnection
from aicon.base_classes.derivatives import DerivativeDict


def collect_args(all_inputs, f_signature, diff_argnums, f_output_signature_diff) -> Tuple[List[torch.Tensor], Tuple[int], List[str]]:
    """
    Collect and organize function arguments for automatic differentiation.
    
    Args:
        all_inputs: Dictionary of all available inputs
        f_signature: List of argument names required by the function
        diff_argnums: Optional tuple of indices for differentiable arguments
        f_output_signature_diff: List of differentiable output names
        
    Returns:
        Tuple containing:
        - List of input tensors in order of f_signature
        - Tuple of indices for differentiable arguments
        - List of names for differentiable inputs
    """
    inputs = all_inputs.select(*f_signature).values()
    if diff_argnums is None:
        diff_argnums = tuple([i for i, arg in enumerate(f_signature) if arg not in [*f_output_signature_diff, "dt"]])
    input_names_diff = [f_signature[idx] for idx in diff_argnums]
    return inputs, diff_argnums, input_names_diff


def collect_inputs(quantities, connections, dt) -> TensorDict:
    """
    Collect all inputs from quantities and connections.
    
    Args:
        quantities: Dictionary of local quantities
        connections: Dictionary of connections to other components
        dt: Time step
        
    Returns:
        TensorDict containing all inputs including the time step
    """
    collected_inputs = quantities.clone()
    for c in connections.values():
        with c.lock:
            collected_inputs = collected_inputs.update(c.connected_quantities)
    collected_inputs["dt"] = dt
    return collected_inputs


def collect_derivatives(connections) -> Tuple[DerivativeDict, DerivativeDict]:
    """
    Collect forward and backward derivatives from all connections.
    
    Args:
        connections: Dictionary of connections to other components
        
    Returns:
        Tuple containing:
        - Dictionary of forward mode derivatives
        - Dictionary of backward mode derivatives
    """
    collected_forward = DerivativeDict()
    collected_backward = DerivativeDict()
    for c in connections.values():
        with c.lock:
            collected_forward = collected_forward.update(c.derivatives_forward_mode)
            collected_backward = collected_backward.update(c.derivatives_backward_mode)
    return collected_forward, collected_backward


def gather_partial_derivatives(partial_jacs:Dict[int, Dict[int, torch.Tensor]], input_names_diff:List[str], output_names_diff:List[str], in_and_out_shapes: Dict[str, torch.Size]) -> TensorDict:
    """
    Gather and organize partial derivatives from automatic differentiation.
    
    This function handles various edge cases in how PyTorch packs derivatives
    and ensures consistent tensor shapes.
    
    Args:
        partial_jacs: Dictionary of partial Jacobians
        input_names_diff: List of differentiable input names
        output_names_diff: List of differentiable output names
        in_and_out_shapes: Dictionary of input and output tensor shapes
        
    Returns:
        TensorDict containing organized partial derivatives
    """
    partial_derivatives = TensorDict({}, batch_size=())
    for o_idx, o in enumerate(output_names_diff):
        for i_idx, i in enumerate(input_names_diff):
            if len(output_names_diff) > 1 or sum(in_and_out_shapes[output_names_diff[0]]) > 1:
                # gets packed up as dict output -> dict input from torch.jacfwd/torch.jacrev
                partial_jac = partial_jacs[o_idx][i_idx]
            else:
                # gets packed up as single dict from torch.jacfwd/torch.jacrev
                partial_jac = partial_jacs[i_idx]
            # depending on exact specification, torch can have issues for scalar values
            # this is error handling for these
            if in_and_out_shapes[i] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(0)
            if in_and_out_shapes[o] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(-1)
            if sum(partial_jac.shape) == 0:
                partial_jac = partial_jac.view(1, 1)
            partial_derivatives[o, i] = partial_jac
    return partial_derivatives


def generate_outputdict(outputs: Iterable[torch.Tensor], output_names: Iterable[str]) -> TensorDict:
    """
    Generate a TensorDict from outputs and their names.
    
    Args:
        outputs: Iterable of output tensors
        output_names: Iterable of output names
        
    Returns:
        TensorDict mapping output names to their tensors
    """
    return TensorDict({n: t for n, t in zip(output_names, outputs)}, batch_size=())


def collect_shapes(all_inputs, output_dict):
    """
    Collect shapes of all inputs and outputs.
    
    Args:
        all_inputs: Dictionary of input tensors
        output_dict: Dictionary of output tensors
        
    Returns:
        Dictionary mapping tensor names to their shapes
    """
    in_out_shapes = dict()
    for k, v in output_dict.items():
        in_out_shapes[k] = v.shape
    for k, v in all_inputs.items():
        in_out_shapes[k] = v.shape
    return in_out_shapes


def collect_required_signs(connections : Dict[str, ActiveInterconnection]) -> Dict[str, Dict[str, int]]:
    """
    Collect required signs for derivatives from all connections.
    
    Args:
        connections: Dictionary of connections to other components
        
    Returns:
        Dictionary mapping quantity names to their required derivative signs
    """
    required_signs_dict : Dict[str, Dict[str, int]] = dict()
    for c in connections.values():
        for k, v in c.required_signs_dict.items():
            if k in required_signs_dict:
                required_signs_dict[k].update(v)
            else:
                required_signs_dict[k] = v
    return required_signs_dict
