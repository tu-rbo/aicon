"""
Estimators for the Blocks World Domain

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent
from .global_params import NUM_BLOCKS
from aicon_base.inference.util import gradient_preserving_clipping


class BelowEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_below": (NUM_BLOCKS, NUM_BLOCKS)}

    def initialize_quantities(self):
        c = self.connections["BelowLikelihoodDummySensing"]
        with c.lock:
            if c.connected_quantities_initialized["below_state_sensed"]:
                likelihood_below_sensed = c.connected_quantities["below_state_sensed"]
            else:
                return False
        self.quantities["likelihood_below"] = likelihood_below_sensed
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["BelowLikelihood"].c_func
        c_func_sensor = self.connections["BelowLikelihoodDummySensing"].c_func

        def f_func(likelihood_below, action_blocks, likelihood_clear, below_state_sensed):
            innovation = c_func(likelihood_below, action_blocks, likelihood_clear)
            new_likelihood = likelihood_below + innovation
            new_likelihood = (
                new_likelihood + c_func_sensor(new_likelihood, below_state_sensed).detach()
            )  # no grad from sensing
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.001, 0.999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_below"], ["likelihood_below"]


class ClearEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_clear": (NUM_BLOCKS,)}

    def initialize_quantities(self):
        c = self.connections["ClearLikelihood"]
        with c.lock:
            if c.connected_quantities_initialized["likelihood_below"]:
                likelihood_below = c.connected_quantities["likelihood_below"]
                c_func = c.c_func
            else:
                return False
        self.quantities["likelihood_clear"] = torch.clip(
            c_func(torch.zeros(NUM_BLOCKS, dtype=self.dtype, device=self.device), likelihood_below), 0.01, 0.99
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["ClearLikelihood"].c_func

        def f_func(likelihood_clear, likelihood_below):
            innovation = c_func(likelihood_clear, likelihood_below)
            new_likelihood = likelihood_clear + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.01, 0.99)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_clear"], ["likelihood_clear"]
