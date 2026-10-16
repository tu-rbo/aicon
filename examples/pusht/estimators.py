"""
Estimators for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Dict, Tuple

import torch
from tensordict import TensorDict

from aicon_base.components import EstimationComponent, BatchedEstimationComponent
from .util import polygon_centroid
from .params import NUM_KEYPOINTS


class ContactsEstimatorSim(BatchedEstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "contactpoint_locations": (NUM_KEYPOINTS, 2),
        "contact_likelihoods": (NUM_KEYPOINTS,),
        "push_velocity_directions": (NUM_KEYPOINTS, 3),
    }
    batch_dimensions = (NUM_KEYPOINTS,)
    to_be_batched_inputs = ["pusher_point"]
    batch_backward_derivative_combination_strategy = "projection"

    def initialize_quantities(self):
        c = self.connections["ContactPoints2PushVelocityDirections"]
        with c.lock:
            if c.connected_quantities_initialized["keypoint_locations"]:
                keypoint_locations = c.connected_quantities["keypoint_locations"]
                object_COM = polygon_centroid(keypoint_locations)
            else:
                return False
        c = self.connections["KeyPoints2ContactPoints"]
        with c.lock:
            if c.connected_quantities_initialized["pusher_point"]:
                pusher_point = c.connected_quantities["pusher_point"]
            else:
                return False
        c_func_k2c = self.connections["KeyPoints2ContactPoints"].c_func
        c_func_c2cl = self.connections["ContactPoints2ContactLikelihood"].c_func
        c_func_c2v = self.connections["ContactPoints2PushVelocityDirections"].c_func
        self.quantities["contactpoint_locations"] = c_func_k2c(
            keypoint_locations, self.quantities["contactpoint_locations"], pusher_point
        )
        self.quantities["contact_likelihoods"] = c_func_c2cl(
            keypoint_locations,
            self.quantities["contactpoint_locations"],
            pusher_point,
            self.quantities["contact_likelihoods"],
        )
        self.quantities["push_velocity_directions"] = c_func_c2v(
            keypoint_locations,
            self.quantities["contactpoint_locations"],
            object_COM,
            self.quantities["push_velocity_directions"],
        )
        return True

    def define_estimation_function_f(self):
        c_func_k2c = self.connections["KeyPoints2ContactPoints"].c_func
        c_func_c2cl = self.connections["ContactPoints2ContactLikelihood"].c_func
        c_func_c2v = self.connections["ContactPoints2PushVelocityDirections"].c_func

        self.trigger_length = 0.35
        self.trigger_similarity = -0.6
        self.end_similarity = -0.4
        self.end_length = 0.25
        self.out_of_conflict_counter = 5
        self.input_buffer_length = 50

        def f_func(
            contactpoint_locations,
            pusher_point_batched,
            contact_likelihoods,
            keypoint_locations,
            push_velocity_directions,
        ):
            object_COM = polygon_centroid(keypoint_locations)
            contactpoint_locations_new = contactpoint_locations + c_func_k2c(
                keypoint_locations, contactpoint_locations, pusher_point_batched
            )
            contact_likelihoods_new = contact_likelihoods + c_func_c2cl(
                keypoint_locations, contactpoint_locations, pusher_point_batched, contact_likelihoods
            )
            push_velocity_directions_new = push_velocity_directions + c_func_c2v(
                keypoint_locations, contactpoint_locations_new, object_COM, push_velocity_directions
            )
            return (contactpoint_locations_new, contact_likelihoods_new, push_velocity_directions_new), (
                contactpoint_locations_new,
                contact_likelihoods_new,
                push_velocity_directions_new,
            )

        return (
            f_func,
            ["contactpoint_locations", "contact_likelihoods", "push_velocity_directions"],
            ["contactpoint_locations", "contact_likelihoods", "push_velocity_directions"],
        )

    def preprocess_derivatives(self, relevant_backward_derivatives: TensorDict) -> TensorDict:
        relevant_backward_derivatives = super().preprocess_derivatives(relevant_backward_derivatives)
        # The simulation is perfectly symmetric: pushing into a contact and backing away from it produce mirrored
        # gradients on a locally symmetric object. Damping the positive side slightly breaks that tie.
        if "contact_likelihoods" in relevant_backward_derivatives:
            derivative_tensor = relevant_backward_derivatives["contact_likelihoods"].derivatives_tensor
            relevant_backward_derivatives["contact_likelihoods"].derivatives_tensor[derivative_tensor > 0] = (
                derivative_tensor[derivative_tensor > 0] * 0.7
            )
        return relevant_backward_derivatives


class PusherEstimatorSim(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "pusher_point": (2,),
    }

    def initialize_quantities(self):
        c = self.connections["PusherPointMeasurement"]
        with c.lock:
            if c.connected_quantities_initialized["pusher_point_meas"]:
                pusher_point_meas = c.connected_quantities["pusher_point_meas"]
            else:
                return False
        self.quantities["pusher_point"] = c.c_func(pusher_point_meas, self.quantities["pusher_point"])
        return True

    def define_estimation_function_f(self):
        c_func_act = self.connections["PusherPointAction"].c_func
        c_func_obs = self.connections["PusherPointMeasurement"].c_func

        self.output_buffer_length = 20
        self.nullspace_escape_trigger_length = 0.7
        self.nullspace_escape_trigger_similarity = -0.7
        self.nullspace_escape_end_similarity = -0.4
        self.nullspace_escape_end_length = 0.3
        self.nullspace_escape_phase_out_ticks = 2
        self.nullspace_escape_short_window = 300
        self.nullspace_escape_medium_window = 600
        self.nullspace_escape_long_window = 1000
        self.alpha["pusher_point"] = 0.8

        def f_func(pusher_point, pusher_point_action, pusher_point_meas):
            pusher_point_pred = c_func_act(pusher_point, pusher_point_action)
            pusher_point_corr = pusher_point_pred + c_func_obs(pusher_point_meas, pusher_point_pred.detach())
            return (pusher_point_corr,), (pusher_point_corr,)

        return f_func, ["pusher_point"], ["pusher_point"]


class InteractionEstimatorSim(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"interaction_likelihood": (1,)}

    def initialize_quantities(self):
        c = self.connections["ObjectvsPush2InteractionlikelihoodSim"]
        with c.lock:
            if c.connected_quantities_initialized["pusher_point"]:
                pusher_point = c.connected_quantities["pusher_point"]
            else:
                return False
        with c.lock:
            if c.connected_quantities_initialized["object_position"]:
                object_position = c.connected_quantities["object_position"]
            else:
                return False
        self.quantities["interaction_likelihood"] = -c.c_func(
            object_position, pusher_point, torch.zeros((1,), dtype=self.dtype, device=self.device)
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["ObjectvsPush2InteractionlikelihoodSim"].c_func
        self.differentiate_against = ("pusher_point",)

        def f_func(object_position, pusher_point, interaction_likelihood):
            new_likelihood = interaction_likelihood - c_func(object_position, pusher_point, interaction_likelihood)
            return (new_likelihood,), (new_likelihood,)

        return (
            f_func,
            [
                "interaction_likelihood",
            ],
            [
                "interaction_likelihood",
            ],
        )


class ObjectEstimatorSim(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "object_position": (3,),
    }

    def initialize_quantities(self):
        c = self.connections["KeyPoints2Position"]
        with c.lock:
            if c.connected_quantities_initialized["keypoint_locations"]:
                keypoint_locations = c.connected_quantities["keypoint_locations"]
            else:
                return False
        self.quantities["object_position"] = c.c_func(keypoint_locations, self.quantities["object_position"])
        return True

    def define_estimation_function_f(self):
        c_func_contacts = self.connections["Contacts2ObjectPosition"].c_func
        c_func_obs = self.connections["KeyPoints2Position"].c_func

        def f_func(
            object_position, keypoint_locations, contact_likelihoods, push_velocity_directions, interaction_likelihood
        ):
            object_position_pred = c_func_contacts(
                object_position, contact_likelihoods, push_velocity_directions, interaction_likelihood
            )
            object_position_new = object_position_pred + c_func_obs(keypoint_locations, object_position_pred.detach())
            return (object_position_new,), (object_position_new,)

        return f_func, ["object_position"], ["object_position"]
