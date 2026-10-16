"""
Estimators for Real-Robot Object Pushing

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import time
from typing import Dict, Tuple

import torch

from aicon_base.components import EstimationComponent, BatchedEstimationComponent
from aicon_base.math.se3_and_SE3 import exponential_map_se3, homogeneous_transform_inverse
from aicon_real_robots.camera_visibility import visible_likelihood_given_relative_pos
from .util import polygon_centroid
from .params import NUM_KEYPOINTS
from aicon_real_robots.estimators import JointControlEEPoseEstimator

# While the object has not been observed for this long [s], the pusher non-penetration constraint is not applied:
# it would otherwise move a stale belief along with the robot's own motion.
BLIND_FREEZE_S = 5.0


class _BlindClock:
    """Wall-clock seconds (time.monotonic()) since the last tick that carried an object observation.

    Advanced in `preprocess_inputs` (once per tick); `reset()` is idempotent and may be called from f_func.
    """

    def __init__(self):
        self.last_seen = None  # None: nothing seen yet, nothing to freeze

    def advance(self, all_inputs) -> None:
        if "keypoints_in_CF" not in all_inputs.keys():
            return
        if not bool(torch.any(torch.isnan(all_inputs["keypoints_in_CF"]))):
            self.reset()

    def reset(self) -> None:
        self.last_seen = time.monotonic()

    @property
    def stale(self) -> float:
        return 0.0 if self.last_seen is None else time.monotonic() - self.last_seen

    @property
    def frozen(self) -> bool:
        return BLIND_FREEZE_S > 0.0 and self.last_seen is not None and self.stale >= BLIND_FREEZE_S


class _BlindFreezeMixin:
    """Gives a component its own `_blind` clock, advanced once per tick."""

    @property
    def _blind(self) -> "_BlindClock":
        clock = self.__dict__.get("_blind_clock")
        if clock is None:
            clock = self.__dict__["_blind_clock"] = _BlindClock()
        return clock

    def preprocess_inputs(self, all_inputs):
        self._blind.advance(all_inputs)
        return super().preprocess_inputs(all_inputs)


class ContactsEstimator(BatchedEstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "contactpoint_locations": (NUM_KEYPOINTS, 2),
        "contact_likelihoods": (NUM_KEYPOINTS,),
        "push_velocity_directions": (NUM_KEYPOINTS, 3),
    }
    batch_dimensions = (NUM_KEYPOINTS,)
    to_be_batched_inputs = ["pusher_point"]
    batch_backward_derivative_combination_strategy = "projection"

    def initialize_quantities(self):
        c = self.connections["ContactPoints2PushVelocityDirectionsReal"]
        with c.lock:
            if c.connected_quantities_initialized["keypoint_locations"]:
                keypoint_locations = c.connected_quantities["keypoint_locations"]
                object_COM = torch.mean(keypoint_locations, dim=0)
            else:
                return False
        c = self.connections["KeyPoints2ContactPointsReal"]
        with c.lock:
            if c.connected_quantities_initialized["pusher_point"]:
                pusher_point = c.connected_quantities["pusher_point"]
            else:
                return False
        c_func_k2c = self.connections["KeyPoints2ContactPointsReal"].c_func
        c_func_c2cl = self.connections["ContactPoints2ContactLikelihoodReal"].c_func
        c_func_c2v = self.connections["ContactPoints2PushVelocityDirectionsReal"].c_func
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
        c_func_k2c = self.connections["KeyPoints2ContactPointsReal"].c_func
        c_func_c2cl = self.connections["ContactPoints2ContactLikelihoodReal"].c_func
        c_func_c2v = self.connections["ContactPoints2PushVelocityDirectionsReal"].c_func
        self.trigger_length = 0.15
        self.trigger_similarity = -0.6
        self.end_similarity = -0.2
        self.end_length = 0.09
        # ticks the conflict escape is kept up after the conflict condition stops (0.67 s at 15 Hz)
        self.out_of_conflict_counter = 10
        self.input_buffer_length = 150
        self.differentiate_against = ("pusher_point_batched",)

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

    def preprocess_derivatives(self, relevant_backward_derivatives):
        relevant_backward_derivatives = super(self.__class__, self).preprocess_derivatives(
            relevant_backward_derivatives
        )
        # Pushing into a contact and backing away from it give mirrored gradients on a locally symmetric object;
        # damping the positive side breaks the tie.
        if "contact_likelihoods" in relevant_backward_derivatives:
            derivative_tensor = relevant_backward_derivatives["contact_likelihoods"].derivatives_tensor
            relevant_backward_derivatives["contact_likelihoods"].derivatives_tensor[derivative_tensor > 0] = (
                derivative_tensor[derivative_tensor > 0] * 0.5
            )
        return relevant_backward_derivatives


class PusherEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"pusher_point": (2,), "pusher_height": (1,)}

    def initialize_quantities(self):
        c = self.connections["EE2Pusher"]
        with c.lock:
            if c.connected_quantities_initialized["pose_ee"]:
                pose_ee = c.connected_quantities["pose_ee"]
                new_pusher_point = -c.c_func(torch.zeros_like(self.quantities["pusher_point"]), pose_ee)
            else:
                return False
        c = self.connections["EE2PusherHeight"]
        with c.lock:
            if c.connected_quantities_initialized["pose_ee"]:
                pose_ee = c.connected_quantities["pose_ee"]
                new_pusher_height = -c.c_func(torch.zeros_like(self.quantities["pusher_height"]), pose_ee)
            else:
                return False
        self.quantities["pusher_point"] = new_pusher_point
        self.quantities["pusher_height"] = new_pusher_height
        return True

    def define_estimation_function_f(self):
        c_func_point = self.connections["EE2Pusher"].c_func
        c_func_height = self.connections["EE2PusherHeight"].c_func
        self.differentiate_against = ("pose_ee",)
        self.alpha["pusher_height"] = 0.4
        self.alpha["pusher_point"] = 1.0
        self.nullspace_escape_trigger_length = 0.2
        self.nullspace_escape_trigger_similarity = -0.6
        self.nullspace_escape_end_similarity = -0.2
        self.nullspace_escape_end_length = 0.1
        self.nullspace_escape_phase_out_ticks = 5
        self.nullspace_escape_short_window = 120
        self.nullspace_escape_medium_window = 300
        self.nullspace_escape_long_window = 600
        self.output_buffer_length = 600

        def f_func(pusher_point, pusher_height, pose_ee):
            new_pusher_height = pusher_height - c_func_height(pusher_height, pose_ee)
            new_pusher_point = pusher_point - c_func_point(pusher_point, pose_ee)
            return (new_pusher_point, new_pusher_height), (new_pusher_point, new_pusher_height)

        return f_func, ["pusher_point", "pusher_height"], ["pusher_point", "pusher_height"]


class InteractionEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"interaction_likelihood": (1,)}

    def initialize_quantities(self):
        c = self.connections["ObjectvsPush2Interactionlikelihood"]
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
        c_func = self.connections["ObjectvsPush2Interactionlikelihood"].c_func
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


class ObjectEstimator(_BlindFreezeMixin, EstimationComponent):
    """Object pose (x, y, theta), height and pose uncertainty.

    Predicted with the contact model (gradient only), corrected by the keypoint measurement and the pusher
    non-penetration constraint (values only).
    """

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "object_position": (3,),
        "object_height": (1,),
        "object_uncertainty": (1,),
    }

    def initialize_quantities(self):
        c = self.connections["KeyPoints2PositionReal"]
        with c.lock:
            if (
                c.connected_quantities_initialized["keypoint_locations"]
                and c.connected_quantities_initialized["keypoint_locations"]
            ):
                keypoint_locations = c.connected_quantities["keypoint_locations"]
                object_angle_meas = c.connected_quantities["object_angle_meas"]
                angle_has_arrived = bool(c.connected_quantities_initialized.get("object_angle_meas", False))
            else:
                return False
        c = self.connections["Keypoints2ObjectHeight"]
        with c.lock:
            if c.connected_quantities_initialized["pose_ee"] and c.connected_quantities_initialized["keypoints_in_CF"]:
                pose_ee = c.connected_quantities["pose_ee"]
                keypoints_in_CF = c.connected_quantities["keypoints_in_CF"]
                object_height = -c.c_func(0.0, keypoints_in_CF, pose_ee)
                object_uncertainty = -self.connections["KeyPoint2ObjectUncertainty"].c_func(
                    keypoint_locations, 0.0, object_height, pose_ee
                )
            else:
                return False

        # the orientation is only measured by the object detector, so wait until the sensor publishes it
        if not angle_has_arrived:
            self._no_angle = getattr(self, "_no_angle", 0) + 1
            if self._no_angle % 100 == 0:
                print(
                    "ObjectEstimator waiting: `object_angle_meas` has never arrived (%d ticks). Check that the "
                    "sensors run the same version of this example and that the object detector became active."
                    % self._no_angle
                )
            return False

        pos2d = polygon_centroid(keypoint_locations)
        # The object frame is defined at registration (TwoDimKeypointEstimator) with angle 0. Use the measured angle if
        # there is one already.
        orientation = (
            object_angle_meas.reshape(()).to(dtype=pos2d.dtype, device=pos2d.device)
            if not torch.any(torch.isnan(object_angle_meas))
            else torch.zeros((), dtype=pos2d.dtype, device=pos2d.device)
        )
        self.quantities["object_position"] = torch.cat((pos2d, orientation.unsqueeze(0)))
        self.quantities["object_height"] = object_height
        self.quantities["object_uncertainty"] = object_uncertainty
        print("Object Intial", self.quantities["object_position"], self.quantities["object_height"])
        return True

    def define_estimation_function_f(self):
        c_func_contacts = self.connections["Contacts2ObjectPosition"].c_func
        c_func_obs = self.connections["KeyPoints2PositionReal"].c_func
        c_func_height = self.connections["Keypoints2ObjectHeight"].c_func
        c_func_uncertainty = self.connections["KeyPoint2ObjectUncertainty"].c_func
        c_func_nonpen = self.connections["PusherNonPenetration2ObjectPosition"].c_func
        H_ee_to_cam = self.connections["Keypoints2ObjectHeight"].H_ee_to_cam
        self.differentiate_against = (
            "contact_likelihoods",
            "push_velocity_directions",
            "pose_ee",
            "interaction_likelihood",
        )

        def f_func(
            object_position,
            object_height,
            object_uncertainty,
            keypoint_locations,
            contact_likelihoods,
            push_velocity_directions,
            pose_ee,
            keypoints_in_CF,
            interaction_likelihood,
            pusher_point,
            pusher_height,
            object_angle_meas,
        ):
            # estimate current height
            if not torch.any(torch.isnan(keypoints_in_CF)):
                self._blind.reset()
                object_height_new = object_height - 0.1 * c_func_height(object_height, keypoints_in_CF, pose_ee)
            else:
                object_height_new = object_height

            H_base_to_ee = exponential_map_se3(pose_ee).detach()

            def _visible_enough():
                """Outlier check for the orientation measurement: could the camera actually see the contour?"""
                keypoints_h = torch.cat(
                    [
                        keypoint_locations,
                        object_height.unsqueeze(0).expand(keypoint_locations.shape[0], -1),
                        torch.ones(NUM_KEYPOINTS, 1, dtype=keypoint_locations.dtype, device=keypoint_locations.device),
                    ],
                    dim=1,
                )
                relative_pos = torch.einsum(
                    "ki,ij,bj->bk",
                    homogeneous_transform_inverse(H_ee_to_cam),
                    homogeneous_transform_inverse(H_base_to_ee),
                    keypoints_h,
                )[:, :3]
                likelihood_visible = visible_likelihood_given_relative_pos(
                    relative_pos,
                    steepness=1000.0,
                    steepness_depth=None,
                    depth_midpoint=None,
                    fov_max_x=0.6,
                    fov_max_y=0.37,
                    fov_adder=0.1,
                )
                if torch.sum(likelihood_visible < 0.5) > 1:  # if 2 or more not visible
                    print("Reject not visible enough!", likelihood_visible)
                    return False
                return True

            angle_meas = object_angle_meas
            if not torch.any(torch.isnan(angle_meas)) and not _visible_enough():
                angle_meas = torch.full_like(angle_meas, float("nan"))
            # estimate uncertainty
            object_uncertainty_new = object_uncertainty - 1.0 * c_func_uncertainty(
                keypoint_locations, object_uncertainty, object_height, pose_ee
            )

            # contact-based prediction, as gradient only: the value stays at object_position
            object_position_pred = c_func_contacts(
                object_position, contact_likelihoods, push_velocity_directions, interaction_likelihood
            )
            object_position_pred_filtered = object_position + object_position_pred - object_position_pred.detach()
            # correct with the keypoint measurement (value only)
            object_position_new = (
                object_position_pred_filtered
                + 1.0 * c_func_obs(keypoint_locations, object_position_pred_filtered, angle_meas).detach()
            )

            # pusher non-penetration, last so the measurement does not overwrite it; value only, and not while the
            # belief is stale
            if not self._blind.frozen:
                object_position_nonpen = c_func_nonpen(
                    object_position_new, keypoint_locations, pusher_point, pusher_height, object_height_new
                )
                object_position_new = object_position_new + (object_position_nonpen - object_position_new).detach()

            return (object_position_new, object_uncertainty_new), (
                object_position_new,
                object_uncertainty_new,
                object_height_new,
            )

        return (
            f_func,
            ["object_position", "object_uncertainty"],
            ["object_position", "object_uncertainty", "object_height"],
        )


class TwoDimKeypointEstimator(_BlindFreezeMixin, EstimationComponent):
    """The object's 2D contour keypoints in the base frame, and the registered template (the contour in the object
    frame, created once from the first good observation). The object frame is the base frame at registration, so all
    object angles (and the goal angle) are relative to the object's orientation then.
    """

    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "keypoint_locations": (NUM_KEYPOINTS, 2),
        "keypoints_registered": (NUM_KEYPOINTS, 2),
    }

    def initialize_quantities(self):
        c = self.connections["TwoDimKeypointsFrom3D"]
        with c.lock:
            if c.connected_quantities_initialized["keypoints_in_CF"] and c.connected_quantities_initialized["pose_ee"]:
                keypoints_in_CF = c.connected_quantities["keypoints_in_CF"]
                pose_ee = c.connected_quantities["pose_ee"]
                keypoints_valid = c.connected_quantities["keypoints_valid"]
            else:
                return False

        # register only from an observed contour
        if torch.any(torch.isnan(keypoints_in_CF)):
            return False
        # The template is created once and everything is fitted against it, so it must come from an (almost)
        # complete view, not from a contour cut by the image border.
        n_valid = int(torch.sum(keypoints_valid > 0.5))
        if n_valid < NUM_KEYPOINTS - 1:
            self._reg_stall = getattr(self, "_reg_stall", 0) + 1
            if self._reg_stall % 100 == 0:
                print(
                    "Registration waiting: %d accepted contours so far, none with >= %d keypoints off the "
                    "image border (last had %d). The object may be at the edge of the image."
                    % (self._reg_stall, NUM_KEYPOINTS - 1, n_valid)
                )
            return False
        H_base_to_ee = exponential_map_se3(pose_ee)
        keypoints_pred = torch.einsum(
            "ij,jk,bk->bi",
            H_base_to_ee,
            c.H_ee_to_cam,
            torch.cat([keypoints_in_CF, torch.ones(NUM_KEYPOINTS, 1, dtype=self.dtype, device=self.device)], dim=1),
        )[:, :2]
        # the template: the contour about its centroid, in the base frame's orientation (object angle 0)
        object_position_center = polygon_centroid(keypoints_pred.to(dtype=self.dtype))
        self.quantities["keypoints_registered"] = keypoints_pred - object_position_center
        self.quantities["keypoint_locations"] = keypoints_pred
        return True

    def define_estimation_function_f(self):
        conn_sensor = self.connections["TwoDimKeypointsFrom3D"]
        c_func_sensor = conn_sensor.c_func
        c_func_nonpen = self.connections["PusherNonPenetration2ObjectPosition"].c_func
        self.no_differentiation = True

        def f_func(
            keypoint_locations,
            keypoints_in_CF,
            pose_ee,
            keypoints_registered,
            object_position,
            keypoints_valid,
            keypoints_partial_in_CF,
            pusher_point,
            pusher_height,
            object_height,
        ):
            # a contour the sensor rejected may still be usable if its shape matches the template
            if torch.any(torch.isnan(keypoints_in_CF)):
                salvaged = conn_sensor.salvage_partial(
                    keypoints_partial_in_CF, keypoints_valid, pose_ee, keypoints_registered, object_position
                )
                if salvaged is not None:
                    keypoints_in_CF = salvaged

            if not torch.any(torch.isnan(keypoints_in_CF)) and not (
                torch.any(torch.isnan(object_position)) or not torch.is_nonzero(torch.sum(object_position))
            ):
                self._blind.reset()
                keypoint_locations_new = keypoint_locations - c_func_sensor(
                    keypoint_locations, keypoints_in_CF, pose_ee, keypoints_registered, object_position, keypoints_valid
                )
            else:
                keypoint_locations_new = keypoint_locations

            # Pusher non-penetration on the contour itself (value only), expressed about the contour's own centroid.
            # Not applied while the belief is stale (see BLIND_FREEZE_S).
            if not self._blind.frozen and not torch.any(
                torch.isnan(torch.cat([keypoint_locations_new.flatten(), pusher_point, pusher_height, object_height]))
            ):
                centroid = polygon_centroid(keypoint_locations_new)
                pseudo_position = torch.cat([centroid, torch.zeros_like(centroid[:1])])
                delta = (
                    c_func_nonpen(pseudo_position, keypoint_locations_new, pusher_point, pusher_height, object_height)
                    - pseudo_position
                )[:2]
                keypoint_locations_new = keypoint_locations_new + delta.detach().unsqueeze(0)
            return (keypoint_locations_new,), (keypoint_locations_new,)

        return f_func, ["keypoint_locations"], ["keypoint_locations"]


class CorrectHeightEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"height_difference": (1,)}

    def initialize_quantities(self):
        c = self.connections["HeightDifference"]
        with c.lock:
            if (
                c.connected_quantities_initialized["pusher_height"]
                and c.connected_quantities_initialized["object_height"]
                and c.connected_quantities_initialized["ee_force_external"]
            ):
                self.quantities["height_difference"] = -c.c_func(
                    c.connected_quantities["pusher_height"],
                    c.connected_quantities["object_height"],
                    torch.zeros_like(self.quantities["height_difference"]),
                    c.connected_quantities["ee_force_external"],
                )
            else:
                return False
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["HeightDifference"].c_func
        self.differentiate_against = ("pusher_height",)

        def f_func(pusher_height, object_height, height_difference, ee_force_external):
            height_difference_new = height_difference - c_func(
                pusher_height, object_height, height_difference, ee_force_external
            )
            return (height_difference_new,), (height_difference_new,)

        return f_func, ["height_difference"], ["height_difference"]


class AdaptedJointControlEEPoseEstimator(JointControlEEPoseEstimator):
    """End-effector pose estimator whose gradients to the joints are preconditioned so that descending them moves the
    pusher tip in the direction the goals ask for (see precondition_ee_gradients)."""

    # damping of the preconditioning, relative to the largest singular value of the Jacobian
    precondition_damping_rel: float = 0.05
    # joints that keep their plain J^T g entry; joint 7 only rotates about the pusher axis
    precondition_exclude_joints: Tuple[int, ...] = (6,)

    def _tip_jacobian(self, pose_ee, ee_jacobian):
        """d(pusher tip)/d(joints), 3x7, from the EE2Pusher / EE2PusherHeight connections and the EE Jacobian."""
        try:
            c_point = self.connections["EE2Pusher"].c_func
            c_height = self.connections["EE2PusherHeight"].c_func
        except (KeyError, AttributeError) as exc:
            print("EE estimator: no tip Jacobian (%s), skipping the tip row-space restriction!" % exc)
            return None
        zeros_point = torch.zeros((2,), dtype=self.dtype, device=self.device)
        zeros_height = torch.zeros((1,), dtype=self.dtype, device=self.device)

        def tip_from_pose(pose):
            # the connections return quantity - prediction, so the negated residual at zero is the prediction
            return torch.cat([-c_point(zeros_point, pose), -c_height(zeros_height, pose)])

        return torch.func.jacfwd(tip_from_pose)(pose_ee) @ ee_jacobian

    def inference_and_differentiation(self, args, diff_argnums):
        outputs, partial_jacs = super().inference_and_differentiation(args, diff_argnums)
        # keep d(pose_ee)/d(joints) for the preconditioning, instead of recomputing it
        self._ee_jacobian = None
        if (
            partial_jacs
            and partial_jacs[0]
            and partial_jacs[0][0].ndim == 2
            and partial_jacs[0][0].shape[0] == self.state_dim
        ):
            self._ee_jacobian = partial_jacs[0][0]
        return outputs, partial_jacs

    def precondition_ee_gradients(self, backward_derivatives, relevant_backward_derivatives, inputs):
        """Turn the transposed-Jacobian step at pose_ee into a damped Gauss-Newton step.

        The chain rule maps a gradient g on pose_ee to J^T g on the joints, which moves the EE along J J^T g instead
        of g; J J^T is badly conditioned here. The step is therefore solved through (J J^T + damping)^-1, restricted
        to the part that moves the pusher tip (the goals below pose_ee only concern the tip position), and joint 7
        keeps its J^T entry. Only directions change: every path keeps its original norm, so the priority ordering
        in combine_backwards_derivatives is unaffected.
        """
        if "robot_joint_states" not in backward_derivatives or "pose_ee" not in relevant_backward_derivatives:
            return
        joint_grads = backward_derivatives["robot_joint_states"].derivatives_tensor
        ee_grads = relevant_backward_derivatives["pose_ee"].derivatives_tensor
        if joint_grads.shape[0] != ee_grads.shape[0]:
            print(
                "Unexpected shape mismatch in EE estimator, skipping preconditioning!",
                joint_grads.shape,
                ee_grads.shape,
            )
            return

        jacobian = getattr(self, "_ee_jacobian", None)
        if jacobian is None:
            jacobian = torch.func.jacfwd(lambda q: self._c_joints(inputs["pose_ee"], q))(inputs["robot_joint_states"])
        gramian = torch.einsum("ij,kj->ik", jacobian, jacobian)
        largest_sv = torch.linalg.matrix_norm(jacobian, ord=2)
        damping = torch.pow(self.precondition_damping_rel * largest_sv, 2)
        damped = gramian + damping * torch.eye(gramian.shape[0], dtype=self.dtype, device=self.device)

        preconditioned = torch.einsum("...i,ij,jk->...k", ee_grads, torch.linalg.inv(damped), jacobian)
        old_norms = torch.norm(joint_grads, dim=-1, keepdim=True)
        precond_norms = torch.norm(preconditioned, dim=-1, keepdim=True)
        # same scale as the J^T step, so entries of both can be combined below
        preconditioned = torch.nan_to_num(preconditioned / precond_norms) * old_norms

        # Keep only the component that moves the pusher tip (row space of the damped tip Jacobian); the rest of the
        # full 6-D solution is redundant motion.
        tip_jacobian = self._tip_jacobian(inputs["pose_ee"], jacobian)
        if tip_jacobian is not None:
            tip_gramian = torch.einsum("ij,kj->ik", tip_jacobian, tip_jacobian)
            tip_damping = torch.pow(self.precondition_damping_rel * torch.linalg.matrix_norm(tip_jacobian, ord=2), 2)
            tip_damped = tip_gramian + tip_damping * torch.eye(
                tip_gramian.shape[0], dtype=self.dtype, device=self.device
            )
            restricted = torch.einsum(
                "...i,ji,jk,kl->...l", preconditioned, tip_jacobian, torch.linalg.inv(tip_damped), tip_jacobian
            )
            restricted_norms = torch.norm(restricted, dim=-1, keepdim=True)
            restricted = torch.nan_to_num(restricted / restricted_norms) * old_norms
            preconditioned = torch.where((restricted_norms > 1e-12).expand_as(restricted), restricted, preconditioned)

        # Joint 7 cannot move the tip; the full solution would still assign it motion, so it keeps its J^T entry.
        excluded = torch.zeros_like(preconditioned, dtype=torch.bool)
        excluded[..., list(self.precondition_exclude_joints)] = True
        preconditioned = torch.where(excluded, joint_grads, preconditioned)

        new_norms = torch.norm(preconditioned, dim=-1, keepdim=True)
        rescaled = torch.nan_to_num(preconditioned / new_norms) * old_norms
        # paths whose preconditioned direction vanished keep their original gradient
        keep = (precond_norms > 1e-12).expand_as(rescaled)
        backward_derivatives["robot_joint_states"].derivatives_tensor = torch.where(keep, rescaled, joint_grads)

    def postprocess_derivatives(
        self,
        forward_derivatives,
        backward_derivatives,
        finished_forward_derivatives,
        inputs,
        outputs,
        relevant_backward_derivatives,
    ):
        self.precondition_ee_gradients(backward_derivatives, relevant_backward_derivatives, inputs)
        return super().postprocess_derivatives(
            forward_derivatives,
            backward_derivatives,
            finished_forward_derivatives,
            inputs,
            outputs,
            relevant_backward_derivatives,
        )

    def define_estimation_function_f(self):
        self.alpha["object_uncertainty"] = 0.5
        self._c_joints = self.connections["RobotForwardKinematics"].c_func
        return super().define_estimation_function_f()
