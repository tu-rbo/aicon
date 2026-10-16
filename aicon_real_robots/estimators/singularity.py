"""
AICON Estimator for the Panda's Closeness to Singular Configurations

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


class SingularityEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "likelihood_ee_singularity": (1,),
    }

    def initialize_quantities(self):
        if not self.connections["PandaJoint2SingularityConnection"].connected_quantities_initialized[
            "robot_joint_states"
        ]:
            return False
        self.quantities["likelihood_ee_singularity"] = -self.connections["PandaJoint2SingularityConnection"].c_func(
            torch.zeros((1,), dtype=self.dtype, device=self.device),
            self.connections["PandaJoint2SingularityConnection"].connected_quantities["robot_joint_states"],
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["PandaJoint2SingularityConnection"].c_func

        def f_func(likelihood_ee_singularity, robot_joint_states):
            new_likelihood = likelihood_ee_singularity - c_func(likelihood_ee_singularity, robot_joint_states)
            return (new_likelihood,), (new_likelihood,)

        self.differentiate_against = ("robot_joint_states",)
        return f_func, ["likelihood_ee_singularity"], ["likelihood_ee_singularity"]
