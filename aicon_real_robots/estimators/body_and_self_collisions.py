"""
AICON Estimators for Body Points and Self Collisions

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
from aicon_real_robots.constants import NUM_BODY_POINTS


class BodyEstimator(EstimationComponent):
    state_dim = 3
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"body_points": (NUM_BODY_POINTS, 3)}

    def initialize_quantities(self):
        if not self.connections["PandaJoint2BodyPoints"].connected_quantities_initialized["robot_joint_states"]:
            return False
        self.quantities["body_points"] = self.connections["PandaJoint2BodyPoints"].c_func(
            torch.zeros((self.state_dim,), dtype=self.dtype, device=self.device),
            self.connections["PandaJoint2BodyPoints"].connected_quantities["robot_joint_states"],
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["PandaJoint2BodyPoints"].c_func

        def f_func(body_points, robot_joint_states):
            new_body_points = body_points + c_func(body_points, robot_joint_states)
            return (new_body_points,), (new_body_points,)

        self.differentiate_against = ("robot_joint_states",)
        return f_func, ["body_points"], ["body_points"]


class SelfCollisionEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"likelihood_self_collision": (1,)}

    def initialize_quantities(self):
        if not self.connections["BodyPoints2SelfCollision"].connected_quantities_initialized["body_points"]:
            return False
        self.quantities["likelihood_self_collision"] = self.connections["BodyPoints2SelfCollision"].c_func(
            self.connections["BodyPoints2SelfCollision"].connected_quantities["body_points"],
            torch.zeros((1,), dtype=self.dtype, device=self.device),
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["BodyPoints2SelfCollision"].c_func

        def f_func(likelihood_self_collision, body_points):
            new_likelihood = likelihood_self_collision + c_func(body_points, likelihood_self_collision)
            return (new_likelihood,), (new_likelihood,)

        self.differentiate_against = ("body_points",)
        return f_func, ["likelihood_self_collision"], ["likelihood_self_collision"]
