"""
Generic Derivative Handling for AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from .derivative_maintenance import DerivativeDict, QuantityRelatedDerivativeBlock
from .derivative_processing import (
    get_steepest_gradient_from_derivative_block,
    gradient_descent,
    get_only_max_gradient,
    orthogonalize,
)

__all__ = [
    "DerivativeDict",
    "QuantityRelatedDerivativeBlock",
    "get_steepest_gradient_from_derivative_block",
    "gradient_descent",
    "get_only_max_gradient",
    "orthogonalize",
]
