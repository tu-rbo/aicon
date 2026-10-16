# -*- coding: utf-8 -*-
"""
Timestamps in AICON

Timestamps are float64 CPU tensors, independent of the dtype and device of the components: epoch seconds need
float64 (float32 resolves them only to about two minutes), and CPU tensors are available with every backend. Only
time differences (dt) are cast to a component's dtype and device.

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

TIME_DTYPE = torch.float64


def as_timestamp(t) -> torch.Tensor:
    """`t` (a tensor or a number of seconds) as a float64 CPU tensor."""
    return torch.as_tensor(t, dtype=TIME_DTYPE, device="cpu")


def stamps_tensor(stamps) -> torch.Tensor:
    """A list of timestamps (e.g. `QuantityRelatedDerivativeBlock.oldest_stamps()`) as a float64 CPU tensor."""
    return torch.tensor(stamps, dtype=TIME_DTYPE)
