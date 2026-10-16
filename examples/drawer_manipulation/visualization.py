from typing import Callable, Dict, Union

import torch
import numpy as np

from aicon_base.derivatives import gradient_descent, get_steepest_gradient_from_derivative_block
from .constants import NUM_OBSTACLE_POINTS
from aicon_real_robots.constants import NUM_BODY_POINTS
from aicon_base.math.se3_and_SE3 import unpack_se3_pose, get_only_se3_V, exponential_map_se3
from aicon_ros_python_wrapping.ros_marker_visualization import (
    get_markerarray_publisher,
    construct_cov_markerarray,
    visualize_as_broadcasted_tf,
    get_pris_markers,
    make_marker_array,
    construct_point_markerarray,
    UsedGradientViaROS,
)
from aicon_base.visualization import VisualizationComponent


class DrawerExperimentVisualizer(VisualizationComponent):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self.ee_publisher = get_markerarray_publisher("visualizations/ee_pos")
        self.drawer_publisher = get_markerarray_publisher("visualizations/drawer_pos")
        self.env_point_publisher = get_markerarray_publisher("visualizations/env_points")
        self.kinjoint_publisher = get_markerarray_publisher("visualizations/kinematic_joint")
        self.bodypoint_publisher = get_markerarray_publisher("visualizations/body_points")
        self.obstaclepoint_publisher = get_markerarray_publisher("visualizations/obstacle_points")
        self.grads_sub = UsedGradientViaROS("action_velo_joints")

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

        def visualize_drawer_pos(position_drawer, uncertainty_drawer, timestamp):
            msg = construct_cov_markerarray(
                "panda_link0",
                torch.tensor([1]),
                position_drawer[:3].unsqueeze(0),
                uncertainty_drawer[:3, :3].unsqueeze(0) * 1.5,
                timestamp,
                (1, 0, 0),
                ns="visualizations/drawer_pos",
            )
            self.drawer_publisher.publish(msg)

        def visualize_env_points(
            environment_point_positions, environment_point_uncertainties, environment_point_tracked, timestamp
        ):
            environment_point_tracked = environment_point_tracked.bool()
            if environment_point_positions.shape[0] > environment_point_tracked.shape[0]:
                environment_point_tracked = torch.cat(
                    [
                        environment_point_tracked,
                        torch.ones(
                            environment_point_positions.shape[0] - environment_point_tracked.shape[0],
                            device=self.device,
                            dtype=bool,
                        ),
                    ]
                )
            elif environment_point_positions.shape[0] < environment_point_tracked.shape[0]:
                environment_point_tracked = environment_point_tracked[: environment_point_positions.shape[0]]
            try:
                environment_point_positions = environment_point_positions[environment_point_tracked]
                environment_point_uncertainties = environment_point_uncertainties[environment_point_tracked]
            except IndexError:
                return  # uncertainties and points are not synched right now
            msg = construct_cov_markerarray(
                "panda_link0",
                torch.arange(environment_point_positions.shape[0]),
                environment_point_positions,
                environment_point_uncertainties * 0.5,
                timestamp,
                (0, 0, 1),
                ns="visualizations/drawer_pos",
            )
            self.env_point_publisher.publish(msg)

        def visualize_kin_joint(kinematic_joint, uncertainty_joint, timestamp):
            try:
                pris_markers = get_pris_markers(
                    "panda_link0",
                    timestamp,
                    ns="visualizations/kinematic_joint",
                    id_start=0,
                    kinematic_joint_mu=kinematic_joint,
                    kinematic_joint_sigma=uncertainty_joint,
                )
                msg = make_marker_array(pris_markers)
                self.kinjoint_publisher.publish(msg)
            except np.linalg.LinAlgError:
                return

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

        def visualize_obstacle_points(obstacle_points, timestamp):
            msg = construct_point_markerarray(
                "panda_link0",
                torch.arange(NUM_OBSTACLE_POINTS),
                obstacle_points,
                timestamp,
                (0, 1, 1),
                ns="visualizations/obstacle_points",
            )
            self.obstaclepoint_publisher.publish(msg)

        def visualize_ee_gradient_as_tf(all_quantities, _, all_backward_derivatives):
            try:
                pose_ee = all_quantities["pose_ee"]
                timestamp = all_quantities["timestamp"]
                last_action = all_quantities["action_velo_ee"]
                derivative_block = all_backward_derivatives["action_velo_ee"]
            except KeyError:
                return
            dt = 2.0
            steepest_grad, _, _ = get_steepest_gradient_from_derivative_block(
                derivative_block, timestamp - torch.ones_like(timestamp) * 2
            )
            velocity = gradient_descent(0.5, last_action, steepest_grad)
            t_part, r_part = velocity[:3], velocity[3:]
            t_max, r_max = 0.04, 0.15
            if torch.norm(t_part) > t_max:
                t_part = t_part / torch.norm(t_part) * t_max
            if torch.norm(r_part) > r_max:
                r_part = r_part / torch.norm(r_part) * r_max
            velocity = torch.cat([t_part, r_part])
            new_pose = torch.einsum("ij,jk->ik", exponential_map_se3(pose_ee), exponential_map_se3(velocity * dt))
            visualize_as_broadcasted_tf(new_pose, timestamp, "panda_link0", "current_grad_goal")

        quantity_viz_funcs = {
            ("pose_ee", "uncertainty_ee", "timestamp"): visualize_ee_pos,
            ("position_drawer", "uncertainty_drawer", "timestamp"): visualize_drawer_pos,
            ("kinematic_joint", "uncertainty_joint", "timestamp"): visualize_kin_joint,
            ("obstacle_points", "timestamp"): visualize_obstacle_points,
            ("body_points", "timestamp"): visualize_body_points,
            (
                "environment_point_positions",
                "environment_point_uncertainties",
                "environment_point_tracked",
                "timestamp",
            ): visualize_env_points,
        }
        derivative_viz_funcs_forward = {}
        derivative_viz_funcs_backward = {}
        generic_viz_funcs = []  # [visualize_ee_gradient_as_tf]
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
            "hand_synergy_activation": "Hand Proprioception\nand Actuators",
            "OwnWeightCompensation": "Gravity\nCompensation",
            "DrawerKinematics": "Drawer Kinematics",
            "uncertainty_ee": "Uncertainty",
            "pose_ee": "State",
            "uncertainty_drawer": "Uncertainty",
            "position_drawer": "State",
            "uncertainty_dist": "Uncertainty",
            "distance_ee_drawer": "State",
            "uncertainty_joint": "Uncertainty",
            "kinematic_joint": "State",
            "ee_force_measured": "Force-Torque Sensor",
            "ReduceDistToPoint": "Minimize Distance\nto Target Position",
            "ReduceJointStateDifference": "Goal Joint State",
            "SelfCollisionGoal": "Avoid Self Collisions",
            "ObstacleAvoidanceGoal": "Avoid Obstacle Collisions",
            "JointLimitGoal": "Avoid Joint Limits",
            "environment_point_positions": "State",
            "environment_point_uncertainties": "Uncertainty",
            "Environment2ObstacleConnection": "Distance-Based\nPrioritization of\nObstacle Points",
            "GraspedLikelihood": "Geometry- and\nForce-Based\nLikelihood Function",
        }
        self.simple_combinations = {
            "action_velo_ee": "Arm Proprioception\nand Actuators",
            "pose_measured_ee": "Arm Proprioception\nand Actuators",
            "VisibleLikelihood": "Object Detection\nand\nProjective Geometry",
            "ProjectiveGeometry": "Object Detection\nand\nProjective Geometry",
            "DirectMeasurement": "Robot Kinematics",
            "ForwardKinematics": "Robot Kinematics",
            "GraspedDrawerKinematics": "Object Dynamics\nand\nDistance Calculation",
            "E[dist]": "Object Dynamics\nand\nDistance Calculation",
            "time_since_hand_change": "Likelihood (Grasped)",
            "action_velo_joints": "Arm Proprioception\nand Actuation",
            "robot_joint_states_meas": "Arm Proprioception\nand Actuation",
            "DirectMeasurementJoints": "Linear Dynamics Model",
            "LinearVelocityJoints": "Linear Dynamics Model",
            "RobotForwardKinematics": "Robot Kinematics",
            "PandaJoint2BodyPoints": "Robot Kinematics",
            "relative_position_in_CF_drawer": "RGB-D Camera",
            "coarse_bounding_box_drawer": "RGB-D Camera",
            "relative_position_in_CF_environment_points": "RGB-D Camera",
            "depth_in_CF_environment_points": "RGB-D Camera",
            "present_measurement_points_mask": "RGB-D Camera",
            "depth_in_CF_drawer": "RGB-D Camera",
            "all_depth_values": "RGB-D Camera",
            "BodyPoints2SelfCollision": "Simple Collision Model",
            "ObstacleCollision": "Simple Collision Model",
            "ProjectiveGeometryEnvironment": "Object Detection\nand\nProjective Geometry",
            "DepthCameraProjection": "Object Detection\nand\nProjective Geometry",
        }
        self.recursive_estimators = {
            "pose_ee": "End-Effector Pose",
            "uncertainty_ee": "End-Effector Pose",
            "position_drawer": "Object Position",
            "uncertainty_drawer": "Object Position",
            "distance_ee_drawer": "Distance",
            "uncertainty_dist": "Distance",
            "likelihood_grasped_drawer": "Likelihood (Grasped)",
            "likelihood_visible_drawer": "Likelihood (Visible)",
            "ee_force_external": "External Force",
            "kinematic_joint": "Kinematic Joint",
            "uncertainty_joint": "Kinematic Joint",
            "body_points": "Points on Body",
            "likelihood_self_collision": "Likelihood of Self Collision",
            "likelihood_obstacle_collision": "Likelihood of Obstacle Collision",
            "robot_joint_states": "Robot Joints",
            "environment_point_positions": "Environment Points",
            "environment_point_uncertainties": "Environment Points",
            "obstacle_points": "Current Obstacle Points",
        }
        self.type_combinations = {
            "Action": "Real-World Sensors and Actuators",
            "Sensor": "Real-World Sensors and Actuators",
        }
        self.node_name_to_layer = None
        self.cut_nodes = [
            "environment_point_tracked",
            "environment_point_time_since_last_update",
            "environment_point_tracked_time_total",
        ]
        self.cut_edges = [
            ("Likelihood (Grasped)", "Environment2ObstacleConnection"),
            ("Object Position", "Environment2ObstacleConnection"),
        ]
