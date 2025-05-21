"""
Active Interconnection classes for the AICON framework.

This module provides classes for managing active interconnections between components,
including the base ActiveInterconnection class and its derivatives.
"""
from __future__ import annotations
from abc import abstractmethod
from threading import Lock
from typing import Callable, Tuple, Dict, Union

import torch
from tensordict import TensorDict

from aicon.base_classes.derivatives import DerivativeDict


class ActiveInterconnection:
    """
    Base class for active interconnections between components in the AICON framework.
    
    An ActiveInterconnection manages the dynamic communication and data flow between components, handling:
    - State synchronization
    - Derivative propagation
    - Data type and device management
    
    Attributes:
        device (torch.device): Device (CPU/GPU) for computations
        dtype (torch.dtype): Data type for tensors
        connected_quantities_initialized (dict[str, bool]): Initialization status of connected quantities
        connected_quantities_timestamps (TensorDict): Timestamps of connected quantities
        name (str): Unique identifier for the active interconnection
        lock (Lock): Thread lock for thread-safe operations
        connected_quantities (TensorDict): Dictionary of connected quantities
        derivatives_backward_mode (DerivativeDict): Backward mode derivatives
        derivatives_forward_mode (DerivativeDict): Forward mode derivatives
        c_func (Callable): Active interconnection function for implicit connections
    """
    
    device: torch.device
    dtype: torch.dtype
    connected_quantities_initialized: dict[str, bool]
    connected_quantities_timestamps: TensorDict
    name: str
    lock: Lock
    connected_quantities: TensorDict
    derivatives_backward_mode: DerivativeDict
    derivatives_forward_mode: DerivativeDict
    c_func: Callable

    def __init__(self, name:str, quantity_names_and_shapes: Dict[str, Tuple[int]], dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, required_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None):
        """
        Initialize a new active interconnection.
        
        Args:
            name: Unique identifier for the active interconnection
            quantity_names_and_shapes: Dictionary mapping quantity names to their shapes
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
            required_signs_dict: Optional dictionary of required signs for quantities
        """
        self.name = name
        self.lock = Lock()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.connected_quantities = TensorDict(dict(), device=device, batch_size=())
        self.connected_quantities_timestamps = TensorDict(dict(), device=device, batch_size=())
        self.connected_quantities_initialized = dict()
        self.required_signs_dict = dict()
        self.derivatives_forward_mode = DerivativeDict()
        self.derivatives_backward_mode = DerivativeDict()
        self.initial_definitions(quantity_names_and_shapes, required_signs_dict)
        if not mockbuild:
            self.c_func = torch.func.functionalize(self.define_implicit_connection_function())

    def initial_definitions(self, quantity_names_and_shapes: Dict[str, Tuple[int]], required_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None):
        """
        Initialize the active interconnection's quantities and required signs.
        
        Args:
            quantity_names_and_shapes: Dictionary mapping quantity names to their shapes
            required_signs_dict: Optional dictionary of required signs for quantities
        """
        if required_signs_dict is not None:
            self.required_signs_dict = required_signs_dict
        for n, s in quantity_names_and_shapes.items():
            self.connected_quantities[n] = torch.zeros(s, dtype=self.dtype, device=self.device)
            self.connected_quantities_timestamps[n] = torch.zeros(1, dtype=self.dtype, device=self.device)
            self.connected_quantities_initialized[n] = False

    @abstractmethod
    def define_implicit_connection_function(self):
        """
        Define the implicit active interconnection function.
        
        This method should be implemented by subclasses to define how quantities
        are transformed between connected components.
        
        Returns:
            Callable: The active interconnection function
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass