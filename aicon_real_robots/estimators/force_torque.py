"""
AICON Estimators for FT on the EE

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


class ExternalForceEstimator(EstimationComponent):
    state_dim = 3
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "ee_force_external": (3,),
    }

    def initialize_quantities(self):
        self.quantities["ee_force_external"] = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        return self.connections["OwnWeightCompensation"].connected_quantities_initialized[
            "pose_ee"
        ]  # we need ee pose to not produce garbage

    def define_estimation_function_f(self):
        c_func = self.connections["OwnWeightCompensation"].c_func

        def f_func(pose_ee, ee_force_measured, ee_force_external):
            innovation = c_func(pose_ee, ee_force_measured, ee_force_external)
            new_force = ee_force_external - 0.9 * innovation
            return (new_force,), (new_force,)

        self.no_differentiation = True
        return f_func, ["ee_force_external"], ["ee_force_external"]
