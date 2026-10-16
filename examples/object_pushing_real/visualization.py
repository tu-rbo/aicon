"""
Visualization for Real-Robot Object Pushing (RViz markers, a 2D plot of contacts and gradients, and the network graph)

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Callable, Dict, Union

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from aicon_base.math.se3_and_SE3 import get_only_se3_V, unpack_se3_pose, exponential_map_se3
from aicon_real_robots.constants import NUM_BODY_POINTS

import matplotlib.colors as mcolors
import torch

from aicon_base.visualization.visualization_component import VisualizationComponent
from aicon_ros_python_wrapping.ros_marker_visualization import (
    get_markerarray_publisher,
    construct_cov_markerarray,
    visualize_as_broadcasted_tf,
    construct_point_markerarray,
    UsedGradientViaROS,
    FullGradientViaROS,
)

colors = [*mcolors.TABLEAU_COLORS.values()]


class RealPusherExperimentVisualizer(VisualizationComponent):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.ee_publisher = get_markerarray_publisher("visualizations/ee_pos")
        self.pusher_publisher = get_markerarray_publisher("visualizations/pusher_pos")
        self.object_publisher = get_markerarray_publisher("visualizations/object_pos")
        self.keypoint_publisher = get_markerarray_publisher("visualizations/keypoints")
        self.contactpoint_publisher = get_markerarray_publisher("visualizations/contactpoints")
        self.bodypoint_publisher = get_markerarray_publisher("visualizations/body_points")
        self.grads_sub = UsedGradientViaROS("action_velo_joints")
        self.ee_grad_sub = FullGradientViaROS("pose_ee")
        self.twodim_ax = None

        def visualize_ee_pos(pose_ee, uncertainty_ee, timestamp):
            J = get_only_se3_V(pose_ee, is_batch=False)
            omega, t_se3 = unpack_se3_pose(pose_ee, is_batch=False)
            mu_visu = torch.einsum("ij,j->i", J, t_se3)
            Sigma_visu = torch.einsum("ij,jl,lk->ik", J, uncertainty_ee[:3, :3], torch.transpose(J, -1, -2))
            msg = construct_cov_markerarray(
                "panda_link0",
                torch.tensor([1]),
                mu_visu.unsqueeze(0),
                Sigma_visu.unsqueeze(0),
                timestamp,
                (0, 1, 0),
                ns="visualizations/ee_pos",
            )
            self.ee_publisher.publish(msg)

        def visualize_object_pos(object_position, object_height, object_uncertainty, timestamp):
            try:
                position = torch.cat([object_position[:2], object_height], dim=0)
                uncertainty = torch.eye(3) * object_uncertainty
                msg = construct_cov_markerarray(
                    "panda_link0",
                    torch.tensor([1]),
                    position.unsqueeze(0),
                    uncertainty.unsqueeze(0) * 1.5,
                    timestamp,
                    (1, 0, 0),
                    ns="visualizations/object_pos",
                )
                self.object_publisher.publish(msg)
            except RuntimeError:
                return

        def visualize_keypoints(keypoint_locations, object_height, timestamp):
            keypoint_heights = object_height.unsqueeze(0).repeat(keypoint_locations.shape[0], 1)
            color_increment = 1.0 / (keypoint_locations.shape[0] + 1)
            colors = [(0, 0, color_increment + color_increment * i) for i in range(keypoint_locations.shape[0])]
            msg = construct_point_markerarray(
                "panda_link0",
                torch.arange(keypoint_locations.shape[0]),
                torch.cat([keypoint_locations, keypoint_heights], dim=1),
                timestamp,
                colors,
                ns="visualizations/keypoints",
                scale=0.03,
            )
            self.keypoint_publisher.publish(msg)

        def visualize_contactpoints(contactpoint_locations, object_height, timestamp):
            keypoint_heights = object_height.unsqueeze(0).repeat(contactpoint_locations.shape[0], 1)
            color_increment = 1.0 / (contactpoint_locations.shape[0] + 1)
            colors = [(color_increment + color_increment * i, 0, 0) for i in range(contactpoint_locations.shape[0])]
            msg = construct_point_markerarray(
                "panda_link0",
                torch.arange(contactpoint_locations.shape[0]),
                torch.cat([contactpoint_locations, keypoint_heights], dim=1),
                timestamp,
                colors,
                ns="visualizations/contactpoint_locations",
                scale=0.03,
            )
            self.contactpoint_publisher.publish(msg)

        def visualize_body_points(body_points, timestamp):
            msg = construct_point_markerarray(
                "panda_link0",
                torch.arange(NUM_BODY_POINTS),
                body_points,
                timestamp,
                (1, 1, 0),
                ns="visualizations/body_points",
            )
            self.bodypoint_publisher.publish(msg)

        def visualize_pusher_pos(pusher_point, pusher_height, timestamp):
            position = torch.cat([pusher_point, pusher_height], dim=0)
            msg = construct_point_markerarray(
                "panda_link0",
                torch.tensor([1]),
                position.unsqueeze(0),
                timestamp,
                (0, 1, 0),
                ns="visualizations/pusher_pos",
            )
            self.pusher_publisher.publish(msg)

        def twodim_viz_pushing(all_inputs, all_forward_derivatives, all_backward_derivatives):
            try:
                pusher_point = all_inputs["pusher_point"]
                keypoints = all_inputs["keypoint_locations"]
                contactpoints = all_inputs["contactpoint_locations"]
                push_velocities = all_inputs["push_velocity_directions"]  # noqa: F841 (raises if not available yet)
                batch_grad = all_backward_derivatives["pusher_point_batched"].derivatives_tensor[0, 0]
                pusher_grads = all_backward_derivatives["pusher_point"].derivatives_tensor
                traces = all_backward_derivatives["pusher_point"].quantity_traces
                oldest_stamp_for_each_grad = torch.cat(
                    [
                        torch.min(torch.cat(stamps)).unsqueeze(0)
                        for stamps in all_backward_derivatives["pusher_point"].timestamps
                    ]
                )
            except KeyError:
                return
            relevant_contact_idx = torch.argmax(torch.norm(batch_grad, dim=-1))
            norms = torch.norm(pusher_grads, dim=-1, keepdim=True)
            norms[oldest_stamp_for_each_grad < self.timestamp - 1.0] = 0
            pusher_grad_idx = torch.argmax(norms.squeeze())
            pusher_grad = (pusher_grads[pusher_grad_idx] / norms[pusher_grad_idx]).squeeze()
            norms[pusher_grad_idx] = 0
            pusher_grad_idx2 = torch.argmax(norms.squeeze())
            minor_pusher_grad = (pusher_grads[pusher_grad_idx2] / norms[pusher_grad_idx2]).squeeze()

            if self.twodim_ax is None:
                fig = plt.figure()
                self.twodim_ax = fig.add_subplot(111)
            self.twodim_ax.clear()
            p = Polygon(keypoints, facecolor="grey")
            self.twodim_ax.add_patch(p)
            self.twodim_ax.scatter(contactpoints[:, 0], contactpoints[:, 1], 100, color="blue", marker="o")
            self.twodim_ax.scatter(
                contactpoints[relevant_contact_idx, 0],
                contactpoints[relevant_contact_idx, 1],
                100,
                color="red",
                marker="x",
            )
            self.twodim_ax.scatter(pusher_point[0], pusher_point[1], 50, color="green", marker="o")
            for i in range(contactpoints.shape[0]):
                self.twodim_ax.text(contactpoints[i, 0] + 0.005, contactpoints[i, 1] + 0.005, str(i))
            self.twodim_ax.arrow(
                pusher_point[0],
                pusher_point[1],
                -pusher_grad[0] * 0.03,
                -pusher_grad[1] * 0.03,
                width=0.005,
                color="red",
            )
            self.twodim_ax.text(
                pusher_point[0] - pusher_grad[0] * 0.03,
                pusher_point[1] - pusher_grad[1] * 0.03,
                "-".join(traces[pusher_grad_idx][2:-1]),
            )
            self.twodim_ax.arrow(
                pusher_point[0],
                pusher_point[1],
                -minor_pusher_grad[0] * 0.015,
                -minor_pusher_grad[1] * 0.015,
                width=0.005,
                color="pink",
            )
            self.twodim_ax.text(
                pusher_point[0] - minor_pusher_grad[0] * 0.015,
                pusher_point[1] - minor_pusher_grad[1] * 0.015,
                "-".join(traces[pusher_grad_idx2][2:-1]),
            )
            self.twodim_ax.axis("equal")
            self.twodim_ax.figure.canvas.draw_idle()

        def ee_grad_viz(pose_ee, timestamp):
            try:
                grad = self.ee_grad_sub.get_derivatives()["pose_ee"].derivatives_tensor[0].squeeze()
            except KeyError:
                return
            H_ee = exponential_map_se3(pose_ee)
            velo = exponential_map_se3(-grad.to(H_ee.dtype) / torch.norm(grad, dim=-1, keepdim=True) * 0.2)
            H_new = torch.einsum("ij,jk->ik", H_ee, velo)
            visualize_as_broadcasted_tf(H_new, timestamp, "panda_link0", "grad_goal_ee")

        quantity_viz_funcs = {
            ("pose_ee", "uncertainty_ee", "timestamp"): visualize_ee_pos,
            ("pusher_point", "pusher_height", "timestamp"): visualize_pusher_pos,
            ("object_position", "object_height", "object_uncertainty", "timestamp"): visualize_object_pos,
            ("body_points", "timestamp"): visualize_body_points,
            ("keypoint_locations", "object_height", "timestamp"): visualize_keypoints,
            ("contactpoint_locations", "object_height", "timestamp"): visualize_contactpoints,
            ("pose_ee", "timestamp"): ee_grad_viz,
        }
        derivative_viz_funcs_forward = {}
        derivative_viz_funcs_backward = {}
        generic_viz_funcs = [twodim_viz_pushing]
        super().__init__(
            name,
            connections,
            quantity_viz_funcs,
            derivative_viz_funcs_forward,
            derivative_viz_funcs_backward,
            generic_viz_funcs,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )
        self.obtain_gradient_funcs.append(self.grads_sub.get_traces)

        self.name_remappings = {
            "OwnWeightCompensation": "Gravity\nCompensation",
            "uncertainty_ee": "Uncertainty",
            "pose_ee": "State",
            "object_position": "State",
            "object_uncertainty": "Uncertainty",
            "ee_force_measured": "Force-Torque Sensor",
            "BodyPoints2SelfCollision": "Simple Collision Model",
            "SelfCollisionGoal": "Avoid Self Collisions",
            "EE2Pusher": " Static \nTransformation",
            "KeyPoints2PositionReal": "Static\n Transformation",
            "Contacts2ObjectPosition": "Contact-Based\nDynamics Model",
            "contact_likelihoods": "Contact Likelihoods",
            "interaction_likelihood": "Interaction Likelihood",
            "push_velocity_directions": " Possible Contact\nLocations",
            "GoalHeight": "Minimize\nHeight Difference",
            "Push2Goal": "Reach Object\nGoal Pose",
            "SingularityGoal": "Avoid Singularities",
            "PusherNonPenetration2ObjectPosition": "Non-Penetration\nConstraint",
        }
        self.simple_combinations = {
            "action_velo_joints": "Arm Proprioception\nand Actuation",
            "robot_joint_states_meas": "Arm Proprioception\nand Actuation",
            "DirectMeasurementJoints": "Linear\nDynamics Model",
            "LinearVelocityJoints": "Linear\nDynamics Model",
            "RobotForwardKinematics": "Robot Kinematics\n& Manipulability",
            "PandaJoint2BodyPoints": "Robot Kinematics\n& Manipulability",
            "PandaJoint2SingularityConnection": "Robot Kinematics\n& Manipulability",
            "JointLimitLikelihoodConnection": "Joint Specifications",
            "JointRangeUtilization": "Joint Specifications",
            # posture and joint-limit goals are drawn as one goal
            "PostureGoal": "Keep Joints in\nNominal Range",
            "JointLimitGoal": "Keep Joints in\nNominal Range",
            "HeightDifference": "Keypoint Projection &\n Relative Height Model",
            "EE2PusherHeight": "Keypoint Projection &\n Relative Height Model",
            "Keypoints2ObjectHeight": "Keypoint Projection &\n Relative Height Model",
            "TwoDimKeypointsFrom3D": "Keypoint Projection &\n Relative Height Model",
            "KeyPoint2ObjectUncertainty": "Keypoint Projection &\n Relative Height Model",
            "KeyPoints2ContactPointsReal": "Geometry-Based\nContact Model",
            "ContactPoints2ContactLikelihoodReal": "Geometry-Based\nContact Model",
            "ContactPoints2PushVelocityDirectionsReal": "Geometry-Based\nContact Model",
            "keypoints_in_CF": "RGB-D\n   Camera   ",
            "keypoints_valid": "RGB-D\n   Camera   ",
            "keypoints_partial_in_CF": "RGB-D\n   Camera   ",
            "keypoints_reject_cause": "RGB-D\n   Camera   ",
            "object_angle_meas": "RGB-D\n   Camera   ",
            "ObjectvsPush2Interactionlikelihood": "Geometry-Based\nContact Model",
        }
        self.recursive_estimators = {
            "pose_ee": "End-Effector Pose",
            "uncertainty_ee": "End-Effector Pose",
            "object_position": "Object Pose",
            "object_uncertainty": "Object Pose",
            "body_points": "Points on Body",
            "likelihood_self_collision": "Likelihood of Self Collision",
            "robot_joint_states": "Robot Joints",
            "ee_force_external": "External Force\non Pusher",
            "pusher_point": "Pusher 2D Location",
            "keypoint_locations": "Object 2D Keypoints",
            "push_velocity_directions": "Contacts",
            "contact_likelihoods": "Contacts",
            "interaction_likelihood": "Contacts",
            "pusher_height": "Height Difference\nObject vs. Pusher",
            "object_height": "Height Difference\nObject vs. Pusher",
            "height_difference": "Height Difference\nObject vs. Pusher",
            "likelihood_joint_at_limit": "Utilization of\nJoint Ranges",
            "joint_close_to_limit": "Utilization of\nJoint Ranges",
            "likelihood_ee_singularity": "Likelihood\nEE Singularity",
            "joint_range_utilization": "Utilization of\nJoint Ranges",
        }
        self.type_combinations = {
            "Action": "Real-World Sensors and Actuators",
            "Sensor": "Real-World Sensors and Actuators",
        }
        self.summarized_recursive_estimators = [
            "Height Difference\nObject vs. Pusher",
            "Utilization of\nJoint Ranges",
        ]
        # edges that only appear because of how the network is summarised in the figure
        self.cut_edges = [
            ("Height Difference\nObject vs. Pusher", "Push2Goal"),
            ("Pusher 2D Location", "Height Difference\nObject vs. Pusher"),
            ("Height Difference\nObject vs. Pusher", "Object Pose"),
            ("RGB-D\n   Camera   ", "KeyPoints2PositionReal"),
        ]
        self.cut_nodes = ["contactpoint_locations", "keypoints_registered", "PusherNonPenetration2ObjectPosition"]
        self.node_name_to_layer = None
