"""
Estimators for the drawer experiment.
This module contains components that estimate various states in the drawer manipulation environment,
including end-effector pose, drawer position, distances, grasp states, and joint kinematics.
"""

from typing import Union, Callable, Dict, Iterable

import torch
from loguru import logger

from aicon.base_classes.components import EstimationComponent
from aicon.drawer_experiment.util import likelihood_dist_func, likelihood_func_visible
from aicon.inference.ekf import update_ekf, predict_ekf_other_quantity, predict_ekf, \
    update_switching_ekf_triple_connection, update_switching_ekf, update_shifting_ekf
from aicon.inference.util import gradient_preserving_clipping
from aicon.math.util_3d import exponential_map_se3, homogeneous_transform_inverse
from aicon.middleware.util_ros import wait_for_tf_frame
from aicon.base_classes.connections import ActiveInterconnection


class EEPoseEstimator(EstimationComponent):
    """
    Estimator for the end-effector pose.
    This component estimates the 6D pose of the robot's end-effector using proprioceptive feedback
    and forward kinematics from velocity commands.
    """
    state_dim = 3

    def define_estimation_function_f(self):
        """
        Define the estimation function for end-effector pose.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the pose estimate
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_proprio = self.connections["DirectMeasurement"].c_func
        c_action = self.connections["ForwardKinematics"].c_func

        def f_func(pose_ee, uncertainty_ee, ee_pos_meas, action_velo_ee):
            """
            Update the end-effector pose estimate using EKF.
            
            Args:
                pose_ee: Current pose estimate
                uncertainty_ee: Current uncertainty estimate
                ee_pos_meas: Measured pose from proprioception
                action_velo_ee: Applied velocity command
            
            Returns:
                tuple: Updated pose and uncertainty estimates
            """
            mu_pred, Sigma_pred = predict_ekf_other_quantity(c_action, pose_ee, uncertainty_ee, action_velo_ee,
                                                             torch.zeros(3, 3, dtype=self.dtype, device=self.device),
                                                             torch.eye(3, dtype=self.dtype, device=self.device) * 0.01)
            mu_new, Sigma_new = update_ekf(c_proprio, mu_pred, Sigma_pred, ee_pos_meas,
                                           torch.eye(3, dtype=self.dtype, device=self.device) * 0.01,
                                           torch.eye(3, dtype=self.dtype, device=self.device) * 0.01)
            Sigma_new = Sigma_new.detach()  # currently no grad possible
            # Work around
            delta = c_proprio(mu_new, ee_pos_meas).detach()
            mu_new = mu_new + delta
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]

    def initial_definitions(self):
        """
        Initialize the quantities for end-effector pose estimation.
        """
        self.quantities["pose_ee"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_ee"] = torch.zeros(self.state_dim, self.state_dim, dtype=self.dtype,
                                                        device=self.device)

    def initialize_quantities(self):
        """
        Initialize the quantities using proprioceptive measurements.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_proprio = self.connections["DirectMeasurement"]
        with c_proprio.lock:
            if c_proprio.connected_quantities_initialized["ee_pos_meas"]:
                pose_measured_ee = c_proprio.connected_quantities["ee_pos_meas"]
            else:
                return False
        self.quantities["pose_ee"] = pose_measured_ee
        self.quantities["uncertainty_ee"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * 0.001
        return True


class DrawerPositionEstimator(EstimationComponent):
    """
    Estimator for the drawer position.
    This component estimates the 3D position of the drawer using visual measurements
    and grasp kinematics.
    """
    state_dim = 3

    def initialize_quantities(self):
        """
        Initialize the quantities using visual measurements.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_proprio = self.connections["DrawerDirectMeasurement"]
        with c_proprio.lock:
            if c_proprio.connected_quantities_initialized["drawer_pos_meas"]:
                drawer_pose_measured_ee = c_proprio.connected_quantities["drawer_pos_meas"]
            else:
                return False
        self.quantities["position_drawer"] = drawer_pose_measured_ee
        self.quantities["uncertainty_drawer"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * 0.001
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function for drawer position.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the position estimate
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        def forward_drawer(mu):
            """
            Forward model for drawer position.
            
            Args:
                mu: Current position estimate
            
            Returns:
                tensor: Predicted position
            """
            return mu  # Drawers do not naturally move by themselves

        c_grasped = self.connections["GraspedDrawerKinematics"].c_func
        c_direct_meas = self.connections["DrawerDirectMeasurement"].c_func

        def f_func(position_drawer, uncertainty_drawer, pose_ee, uncertainty_ee, drawer_pos_meas,
                   likelihood_grasped_drawer, dt):
            """
            Update the drawer position estimate using EKF.
            
            Args:
                position_drawer: Current position estimate
                uncertainty_drawer: Current uncertainty estimate
                pose_ee: Current end-effector pose
                uncertainty_ee: End-effector uncertainty
                drawer_pos_meas: Measured drawer position
                likelihood_grasped_drawer: Likelihood of drawer being grasped
                dt: Time step
            
            Returns:
                tuple: Updated position and uncertainty estimates
            """
            forward_noise = likelihood_grasped_drawer * 0.15 * dt + 0.005 * dt
            mu_new, Sigma_new = predict_ekf(forward_drawer, position_drawer, uncertainty_drawer,
                                            torch.eye(3, dtype=self.dtype, device=self.device) * forward_noise)
            mu_new, Sigma_new = update_switching_ekf(c_grasped, mu_new, Sigma_new, pose_ee, uncertainty_ee,
                                                     torch.eye(3, 3, dtype=self.dtype, device=self.device) * 0.03,
                                                     likelihood_grasped_drawer, outlier_rejection_treshold=0.05,
                                                     prevent_uncertainty_state_grad=False)
            mu_new, Sigma_new = update_ekf(c_direct_meas, mu_new, Sigma_new, drawer_pos_meas, 
                                           torch.eye(3, 3, dtype=self.dtype, device=self.device) * 0.03, 
                                                     torch.eye(3, 3, dtype=self.dtype, device=self.device) * 0.03,
                                                     prevent_uncertainty_state_grad=False)
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["position_drawer", "uncertainty_drawer"], ["position_drawer", "uncertainty_drawer"]

    def initial_definitions(self):
        """
        Initialize the quantities for drawer position estimation.
        """
        self.quantities["position_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_drawer"] = torch.zeros(self.state_dim, self.state_dim, dtype=self.dtype,
                                                            device=self.device)


class GraspedEstimator(EstimationComponent):
    """
    Estimator for the grasp state of the drawer.
    This component estimates the likelihood that the drawer is currently grasped
    by the end-effector.
    """
    state_dim = 1

    def initialize_quantities(self):
        """
        Initialize the quantities for grasp state estimation.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.quantities["likelihood_grasped_drawer"] = torch.ones(self.state_dim, dtype=self.dtype, device=self.device) * 0.01
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function for grasp state.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the grasp likelihood
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["GraspedLikelihood"].c_func

        def f_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            """
            Update the grasp likelihood estimate.
            
            Args:
                likelihood_grasped_drawer: Current grasp likelihood
                ee_pose: Current end-effector pose
                drawer_position: Current drawer position
                gripper_activation: Current gripper activation
                ee_force_mag_meas: Current end-effector force magnitude
            
            Returns:
                tuple: Updated grasp likelihood
            """
            innovation = c_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation)
            new_likelihood = likelihood_grasped_drawer + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 1e-10, 0.9999999)
            return (new_likelihood), (new_likelihood)

        return f_func, ["likelihood_grasped_drawer"], ["likelihood_grasped_drawer"]

    def initial_definitions(self):
        """
        Initialize the quantities for grasp state estimation.
        """
        self.quantities["likelihood_grasped_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)


class KinematicJointEstimator(EstimationComponent):
    """
    Estimator for the kinematic joint state of the drawer.
    This component estimates the joint state (azimuth, elevation, state, initial_point)
    of the drawer's kinematic model.
    """
    state_dim = 1 + 1 + 1 + 3   # azimuth elevation state initial_point

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals: Union[None, Dict[str, Callable]] = None,
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, no_differentiation: bool = False,
                 prevent_loops_in_differentiation: bool = True,
                 max_length_differentiation_trace: Union[int, None] = None,
                 initial_rotation_xy : Union[float, None] = None,
                 initial_uncertainty_scale : Union[float, None] = None,
                 sample_init_mean : bool = False):
        """
        Initialize the kinematic joint estimator.
        
        Args:
            name: Name of the estimator
            connections: Dictionary of connections
            goals: Dictionary of goal functions
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
            no_differentiation: Whether to disable differentiation
            prevent_loops_in_differentiation: Whether to prevent loops in differentiation
            max_length_differentiation_trace: Maximum length of differentiation trace
            initial_rotation_xy: Initial rotation in XY plane
            initial_uncertainty_scale: Scale factor for initial uncertainty
            sample_init_mean: Whether to sample initial mean
        """
        self.initial_rotation_xy = initial_rotation_xy
        self.initial_uncertainty_scale = initial_uncertainty_scale
        self.sample_init_mean = sample_init_mean
        super().__init__(name, connections, goals, dtype, device, mockbuild, no_differentiation,
                         prevent_loops_in_differentiation, max_length_differentiation_trace)

    def initialize_quantities(self):
        if not self.connections["DrawerKinematics"].connected_quantities_initialized["position_drawer"]:
            return False
        self.quantities["kinematic_joint"] = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        self.quantities["kinematic_joint"][1] = - torch.pi / 2 # is always on the xy plane
        # TODO: this a bit arbitrary, but we can change it later
        self.quantities["kinematic_joint"][0] = - torch.pi / 2 * 0.5

        self.quantities["kinematic_joint"][3:6] = self.connections["DrawerKinematics"].connected_quantities["position_drawer"]
        self.quantities["uncertainty_joint"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_joint"][3:6,3:6] = self.connections["DrawerKinematics"].connected_quantities["uncertainty_drawer"]

        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function for kinematic joint state.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the joint state estimate
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["DrawerKinematics"].c_func

        def forward_joint(kinematic_joint):
            """
            Forward model for joint state.
            
            Args:
                kinematic_joint: Current joint state estimate
            
            Returns:
                tensor: Predicted joint state
            """
            return kinematic_joint  # Joints do not naturally move by themselves

        R_additive_grasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.02
        R_additive_ungrasped = torch.eye(3, device=self.device, dtype=self.dtype) * 0.00005
        Q_diag = torch.tensor([0.001, 0.001, 0.1, 0.02, 0.02, 0.02], device=self.device, dtype=self.dtype)

        def f_func(kinematic_joint, uncertainty_joint, position_drawer, uncertainty_drawer, likelihood_grasped_drawer, dt):
            """
            Update the joint state estimate.
            
            Args:
                kinematic_joint: Current joint state estimate
                uncertainty_joint: Current joint uncertainty
                position_drawer: Current drawer position
                uncertainty_drawer: Drawer position uncertainty
                likelihood_grasped_drawer: Likelihood of drawer being grasped
                dt: Time step
            
            Returns:
                tuple: Updated joint state and uncertainty estimates
            """
            # starting from 0.5 grasp likelihood we do not want to integrate any starting point info anymore
            # but instead only about the joint
            unlikelihood_grasped = gradient_preserving_clipping((0.2 - likelihood_grasped_drawer) / 0.2, 0.0000000001, 1.0).unsqueeze(0)
            likelihood_grasped = gradient_preserving_clipping(likelihood_grasped_drawer, 0.000000000001, 1.0).unsqueeze(0)
            shift_diagonal_matrix = torch.diag(torch.cat([likelihood_grasped, likelihood_grasped, likelihood_grasped,
                                                          unlikelihood_grasped, unlikelihood_grasped, unlikelihood_grasped]))
            shift_diagonal_matrix = gradient_preserving_clipping(shift_diagonal_matrix, 0.0, 1.0)
            mu_new, Sigma_new = predict_ekf(forward_joint, kinematic_joint, uncertainty_joint,
                                            shift_diagonal_matrix * Q_diag * dt)
            # azi, ele, and state are highly influenced when grasped, initial point else
            R_additive = R_additive_grasped * likelihood_grasped + R_additive_ungrasped * unlikelihood_grasped
            mu_new, Sigma_new = update_shifting_ekf(c_func, mu_new, Sigma_new, position_drawer, uncertainty_drawer, R_additive, shift_diagonal_matrix, outlier_rejection_treshold = 1.0)
            Sigma_new = torch.cat([torch.cat([Sigma_new[:3, :3], torch.zeros_like(Sigma_new[:3, :3])], dim=0),
                                          torch.cat([torch.zeros_like(Sigma_new[3:, 3:]), Sigma_new[3:, 3:]], dim=0)], dim=1)
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["kinematic_joint", "uncertainty_joint"], ["kinematic_joint", "uncertainty_joint"]

    def initial_definitions(self):
        """
        Initialize the quantities for kinematic joint estimation.
        """
        self.quantities["kinematic_joint"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_joint"] = torch.zeros(self.state_dim, self.state_dim, dtype=self.dtype,
                                                            device=self.device)
