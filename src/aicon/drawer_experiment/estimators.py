"""
Estimators for the drawer experiment.
This module contains components that estimate various states in the drawer manipulation environment,
including end-effector pose, drawer position, distances, grasp states, and joint kinematics.
"""

from typing import Union, Callable, Dict, Iterable

import torch

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
    state_dim = 6

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

        def f_func(pose_ee, uncertainty_ee, pose_measured_ee, action_velo_ee):
            """
            Update the end-effector pose estimate using EKF.
            
            Args:
                pose_ee: Current pose estimate
                uncertainty_ee: Current uncertainty estimate
                pose_measured_ee: Measured pose from proprioception
                action_velo_ee: Applied velocity command
            
            Returns:
                tuple: Updated pose and uncertainty estimates
            """
            mu_pred, Sigma_pred = predict_ekf_other_quantity(c_action, pose_ee, uncertainty_ee, action_velo_ee,
                                                             torch.zeros(6, 6, dtype=self.dtype, device=self.device),
                                                             torch.eye(6, dtype=self.dtype, device=self.device) * 0.01)
            mu_new, Sigma_new = update_ekf(c_proprio, mu_pred, Sigma_pred, pose_measured_ee,
                                           torch.eye(6, dtype=self.dtype, device=self.device) * 0.01,
                                           torch.eye(6, dtype=self.dtype, device=self.device) * 0.01)
            Sigma_new = Sigma_new.detach()  # currently no grad possible
            # Work around
            delta = c_proprio(mu_new, pose_measured_ee).detach()
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
            if c_proprio.connected_quantities_initialized["pose_measured_ee"]:
                pose_measured_ee = c_proprio.connected_quantities["pose_measured_ee"]
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

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals: Union[None, Dict[str, Callable]] = None,
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, no_differentiation: bool = False,
                 prevent_loops_in_differentiation: bool = True,
                 max_length_differentiation_trace: Union[int, None] = None,
                 initial_depth : Union[float, None] = None,
                 initial_uncertainty_scale : Union[float, None] = None,
                 sample_init_mean : bool = False,
                 meas_noise_factor : float = 0.025,
                 initial_drawer_pos : Union[None, Iterable] = None):
        """
        Initialize the drawer position estimator.
        
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
            initial_depth: Initial depth estimate
            initial_uncertainty_scale: Scale factor for initial uncertainty
            sample_init_mean: Whether to sample initial mean
            meas_noise_factor: Factor for measurement noise
            initial_drawer_pos: Initial drawer position
        """
        self.initial_depth = initial_depth
        self.initial_uncertainty_scale = initial_uncertainty_scale
        self.sample_init_mean = sample_init_mean
        self.meas_noise_factor = meas_noise_factor
        self.initial_drawer_pos = initial_drawer_pos
        super().__init__(name, connections, goals, dtype, device, mockbuild, no_differentiation,
                         prevent_loops_in_differentiation, max_length_differentiation_trace)

    def initialize_quantities(self):
        """
        Initialize the quantities using visual measurements.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_cam = self.connections["ProjectiveGeometry"]
        with c_cam.lock:
            if c_cam.connected_quantities_initialized["relative_position_in_CF_drawer"] and \
                    c_cam.connected_quantities_initialized["pose_ee"]:
                pose_ee = c_cam.connected_quantities["pose_ee"]
                relative_position = c_cam.connected_quantities["relative_position_in_CF_drawer"]
            else:
                return False
        H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera_color_optical_frame", timeout=5.0).to(dtype=self.dtype,
            device=self.device)
        H = torch.einsum("ij,jk->ik", exponential_map_se3(pose_ee), H_ee_to_cam)
        if self.initial_drawer_pos is not None:
            initial_mu = torch.tensor(self.initial_drawer_pos, dtype=self.dtype, device=self.device)
        else:
            if torch.all(torch.isnan(relative_position)):
                return False
            if self.initial_depth is None:
                relative_position[2] = 0.8  # prior for where it should be
            else:
                relative_position[2] = self.initial_depth
            initial_mu = torch.einsum("ij,j->i", H, torch.concat(
                [relative_position, torch.ones(1, dtype=self.dtype, device=self.device)]))[:3]
        if self.initial_uncertainty_scale is None:
            Sigma = torch.eye(3, dtype=self.dtype, device=self.device) * 0.1
            Sigma[2, 2] = 0.3 #1.2
            initial_Sigma = torch.einsum("ij,jk,kl->il", H[:3, :3], Sigma, torch.transpose(H[:3, :3], dim0=0, dim1=1))
        else:
            Sigma = torch.eye(3, dtype=self.dtype, device=self.device) * 0.2
            Sigma[2, 2] = 1.0 #1.2
            initial_Sigma = torch.einsum("ij,jk,kl->il", H[:3, :3], Sigma * self.initial_uncertainty_scale, torch.transpose(H[:3, :3], dim0=0, dim1=1))
        if self.sample_init_mean:
            likelihood = 0.0
            dist = 0.0
            while likelihood < 0.6 or dist < 0.25:
                sampled_mu = torch.distributions.multivariate_normal.MultivariateNormal(initial_mu, initial_Sigma).sample()
                relative_pos = torch.einsum("ki,ij,j->k",
                                            homogeneous_transform_inverse(H_ee_to_cam),
                                            homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                                            torch.cat([sampled_mu, torch.ones(1, dtype=sampled_mu.dtype,
                                                                                   device=sampled_mu.device)]))[:3]
                likelihood = likelihood_func_visible(pose_ee, sampled_mu, H_ee_to_cam)
                dist = relative_pos[2]
            initial_mu = sampled_mu
            initial_Sigma = initial_Sigma * 2.0
        self.quantities["position_drawer"] = initial_mu
        self.quantities["uncertainty_drawer"] = initial_Sigma
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

        c_cam = self.connections["ProjectiveGeometry"].c_func
        c_grasped = self.connections["GraspedDrawerKinematics"].c_func
        try:
            H_ee_to_cam = wait_for_tf_frame("panda_link8", "camera_color_optical_frame", timeout=5.0).to(
                dtype=self.dtype, device=self.device)
        except AssertionError:
            print("Fallback on saved transform")
            H_ee_to_cam = torch.tensor([[2.5214e-01, -9.6767e-01, 6.4494e-03, -6.3752e-02],
                                        [9.6769e-01, 2.5213e-01, -2.1942e-03, -1.6633e-02],
                                        [4.9714e-04, 6.7943e-03, 9.9998e-01, 4.0590e-02],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],
                                       dtype=self.dtype, device=self.device)
        def process_visual_measurement(Sigma, likelihood_grasped_drawer, likelihood_visible_drawer, mu,
                                       pose_ee, relative_position_in_CF_drawer, uncertainty_ee, dt, dtype, device):
            """
            Process visual measurements for drawer position estimation.
            
            Args:
                Sigma: Current uncertainty estimate
                likelihood_grasped_drawer: Likelihood of drawer being grasped
                likelihood_visible_drawer: Likelihood of drawer being visible
                mu: Current position estimate
                pose_ee: Current end-effector pose
                relative_position_in_CF_drawer: Relative position in camera frame
                uncertainty_ee: End-effector uncertainty
                dt: Time step
                dtype: Data type
                device: Device
            
            Returns:
                tuple: Updated uncertainty and position estimates
            """
            # compute uncertainty contributors
            additive_noise_visibility = (1.0 - likelihood_func_visible(pose_ee, mu, H_ee_to_cam, steepness=5.0))
            additive_noise_distance = (1.0 - likelihood_dist_func(pose_ee, mu, H_ee_to_cam))
            R_add = 5 * (torch.eye(3, dtype=dtype, device=device) * additive_noise_visibility +
                         torch.eye(3, dtype=dtype, device=device) * additive_noise_distance)
            R_meas = torch.eye(3, dtype=dtype, device=device) * self.meas_noise_factor
            likelihood_existent_measurement = likelihood_visible_drawer * (1.0 - likelihood_grasped_drawer.detach())

            # attempt to integrate measurement
            if torch.all(torch.isnan(relative_position_in_CF_drawer)):
                # we are mocking a measurement to determine current simulated gradient
                mock_relative_position_in_CF_drawer = - c_cam(mu, pose_ee,
                                                              torch.zeros_like(relative_position_in_CF_drawer)).detach()
                # always reject mocked measurement though
                mu_new, Sigma_new = update_switching_ekf_triple_connection(c_cam, mu, Sigma, pose_ee,
                                                                   uncertainty_ee,
                                                                   mock_relative_position_in_CF_drawer,
                                                                   R_meas,
                                                                   R_add * 0.01,
                                                                   likelihood_existent_measurement,
                                                                   outlier_rejection_treshold=0.0)
            else:
                # just integrate measurement
                mu_new, Sigma_new = update_switching_ekf_triple_connection(c_cam, mu, Sigma, pose_ee,
                                                                   uncertainty_ee,
                                                                   relative_position_in_CF_drawer,
                                                                   R_meas,
                                                                   R_add,
                                                                   likelihood_existent_measurement,
                                                                   outlier_rejection_treshold=0.5)
            # increase uncertainty if we received a good measurement when we should not have or vice versa
            measurement_not_integrated = torch.equal(mu, mu_new)
            if measurement_not_integrated and likelihood_existent_measurement > 0.5:
                # visual measurements while grasped can be deceiving due to the hand being similarly blue
                if likelihood_grasped_drawer < 0.1:
                    Sigma_new = Sigma + torch.eye(3, dtype=dtype, device=device) * 0.1 * dt * likelihood_existent_measurement
            elif (measurement_not_integrated) and (likelihood_existent_measurement < 0.5) and (not torch.all(torch.isnan(relative_position_in_CF_drawer))):
                if likelihood_grasped_drawer < 0.1 and not likelihood_existent_measurement == 0.0:  # exactly zero means probbaly not fully initialized
                    Sigma_new = Sigma + torch.eye(3, dtype=dtype, device=device) * 0.1 * dt * (1.0 - likelihood_existent_measurement)
            return Sigma_new, mu_new

        def f_func(position_drawer, uncertainty_drawer, pose_ee, uncertainty_ee, relative_position_in_CF_drawer,
                   likelihood_grasped_drawer, likelihood_visible_drawer, time_since_hand_change, dt):
            """
            Update the drawer position estimate using EKF.
            
            Args:
                position_drawer: Current position estimate
                uncertainty_drawer: Current uncertainty estimate
                pose_ee: Current end-effector pose
                uncertainty_ee: End-effector uncertainty
                relative_position_in_CF_drawer: Relative position in camera frame
                likelihood_grasped_drawer: Likelihood of drawer being grasped
                likelihood_visible_drawer: Likelihood of drawer being visible
                time_since_hand_change: Time since last hand state change
                dt: Time step
            
            Returns:
                tuple: Updated position and uncertainty estimates
            """
            forward_noise = likelihood_grasped_drawer * 0.15 * dt + 0.005 * dt
            mu_new, Sigma_new = predict_ekf(forward_drawer, position_drawer, uncertainty_drawer,
                                            torch.eye(3, dtype=self.dtype, device=self.device) * forward_noise)
            Sigma_new, mu_new = process_visual_measurement(Sigma_new, likelihood_grasped_drawer,
                                                                           likelihood_visible_drawer, mu_new, pose_ee,
                                                                           relative_position_in_CF_drawer,
                                                                           uncertainty_ee, dt, self.dtype, self.device)
                # 0.584 = more than 10 % chance of an outlier
            mu_new, Sigma_new = update_switching_ekf(c_grasped, mu_new, Sigma_new, pose_ee, uncertainty_ee,
                                                     torch.eye(3, 3, dtype=self.dtype, device=self.device) * 0.03,
                                                     likelihood_grasped_drawer, outlier_rejection_treshold=0.05,
                                                     prevent_uncertainty_state_grad=False)
            if time_since_hand_change[0] < 0.5 and time_since_hand_change[1] == 0 and likelihood_grasped_drawer != 0.0:
                # possibly in the process of loosing grasp
                Sigma_new = Sigma_new + torch.eye(3, dtype=self.dtype, device=self.device) * dt
            return (mu_new, Sigma_new), (mu_new, Sigma_new)

        return f_func, ["position_drawer", "uncertainty_drawer"], ["position_drawer", "uncertainty_drawer"]

    def initial_definitions(self):
        """
        Initialize the quantities for drawer position estimation.
        """
        self.quantities["position_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_drawer"] = torch.zeros(self.state_dim, self.state_dim, dtype=self.dtype,
                                                            device=self.device)


class DistEEDrawerEstimator(EstimationComponent):
    """
    Estimator for the distance between end-effector and drawer.
    This component estimates the 2D distance (position and orientation) between
    the end-effector and the drawer.
    """
    state_dim = 2

    def initialize_quantities(self):
        """
        Initialize the quantities using distance measurements.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_dist = self.connections["E[dist]"]
        with c_dist.lock:
            if c_dist.connected_quantities_initialized["position_drawer"] and c_dist.connected_quantities_initialized[
                "pose_ee"]:
                pose_ee = c_dist.connected_quantities["pose_ee"]
                position_drawer = c_dist.connected_quantities["position_drawer"]
                uncertainty_ee = c_dist.connected_quantities["uncertainty_ee"]
                uncertainty_drawer = c_dist.connected_quantities["uncertainty_drawer"]
                c_func = c_dist.c_func
            else:
                return False
        self.quantities["distance_ee_drawer"] = c_func(torch.zeros(self.state_dim, dtype=self.dtype, device=self.device),
                                                                pose_ee, position_drawer)
        self.quantities["uncertainty_dist"] = torch.sum(torch.trace(uncertainty_drawer)) + torch.sum(torch.trace(uncertainty_ee))
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function for end-effector to drawer distance.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the distance estimate
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["E[dist]"].c_func

        def f_func(distance_ee_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            """
            Update the distance estimate.
            
            Args:
                distance_ee_drawer: Current distance estimate
                pose_ee: Current end-effector pose
                uncertainty_ee: End-effector uncertainty
                position_drawer: Current drawer position
                uncertainty_drawer: Drawer position uncertainty
            
            Returns:
                tuple: Updated distance and uncertainty estimates
            """
            innovation = c_func(distance_ee_drawer, pose_ee, position_drawer)
            new_dist = distance_ee_drawer + innovation
            uncertainty = torch.sum(torch.trace(uncertainty_drawer)) + torch.sum(torch.trace(uncertainty_ee))
            return (new_dist, uncertainty), (new_dist, uncertainty)

        return f_func, ["distance_ee_drawer", "uncertainty_dist"], ["distance_ee_drawer", "uncertainty_dist"]

    def initial_definitions(self):
        """
        Initialize the quantities for distance estimation.
        """
        self.quantities["distance_ee_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_dist"] = torch.zeros(1, dtype=self.dtype, device=self.device)


class GraspedEstimator(EstimationComponent):
    """
    Estimator for the grasp state of the drawer.
    This component estimates the likelihood that the drawer is currently grasped
    by the end-effector.
    """
    state_dim = 1

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals: Union[None, Dict[str, Callable]] = None,
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, no_differentiation: bool = False,
                 prevent_loops_in_differentiation: bool = True,
                 max_length_differentiation_trace: Union[int, None] = None, initially_grasped: bool = False):
        """
        Initialize the grasped estimator.
        
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
            initially_grasped: Initial grasp state
        """
        self.initially_grasped = initially_grasped
        super().__init__(name, connections, goals, dtype, device, mockbuild, no_differentiation,
                         prevent_loops_in_differentiation, max_length_differentiation_trace)

    def initialize_quantities(self):
        """
        Initialize the quantities for grasp state estimation.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_grasped = self.connections["GraspedLikelihood"]
        with c_grasped.lock:
            if c_grasped.connected_quantities_initialized["distance_ee_drawer"] and \
                    c_grasped.connected_quantities_initialized["hand_synergy_activation"] and \
                    c_grasped.connected_quantities_initialized["ee_force_external"] and \
                    c_grasped.connected_quantities_initialized["uncertainty_dist"] and \
                    c_grasped.connected_quantities_initialized["time_since_hand_change"]:
                distance_ee_drawer = c_grasped.connected_quantities["distance_ee_drawer"]
                hand_synergy_activation = c_grasped.connected_quantities["hand_synergy_activation"]
                ee_force_external = c_grasped.connected_quantities["ee_force_external"]
                uncertainty_dist = c_grasped.connected_quantities["uncertainty_dist"]
                time_since_hand_change = c_grasped.connected_quantities["time_since_hand_change"]
                c_func = c_grasped.c_func
            else:
                return False
        if self.initially_grasped:
            self.quantities["likelihood_grasped_drawer"] = torch.ones(self.state_dim, dtype=self.dtype, device=self.device) * 0.99
            self.quantities["time_since_hand_change"] = torch.zeros(2, dtype=self.dtype, device=self.device)
        else:
            self.quantities["likelihood_grasped_drawer"] = torch.clip(
                c_func(torch.tensor([0.05], dtype=self.dtype, device=self.device), distance_ee_drawer, uncertainty_dist,
                       hand_synergy_activation, torch.zeros(3, dtype=self.dtype, device=self.device),
                       torch.zeros(2, dtype=self.dtype, device=self.device)), 0.02, 0.98)
            self.quantities["time_since_hand_change"] = torch.zeros(2, dtype=self.dtype, device=self.device)
            self.quantities["time_since_hand_change"][0] = 2.0
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

        def f_func(likelihood_grasped_drawer, distance_ee_drawer, hand_synergy_activation, ee_force_external, uncertainty_dist, time_since_hand_change, dt):
            """
            Update the grasp likelihood estimate.
            
            Args:
                likelihood_grasped_drawer: Current grasp likelihood
                distance_ee_drawer: Current distance to drawer
                hand_synergy_activation: Current hand synergy activation
                ee_force_external: Current external force
                uncertainty_dist: Current distance uncertainty
                time_since_hand_change: Time since last hand state change
                dt: Time step
            
            Returns:
                tuple: Updated grasp likelihood
            """
            time_since_hand_change_new = torch.clone(time_since_hand_change).detach()
            if hand_synergy_activation > 0.5:
                time_since_hand_change_new[0] = 0.0
                time_since_hand_change_new[1] = dt + time_since_hand_change[1]
            else:
                time_since_hand_change_new[0] = time_since_hand_change[0] + dt
                time_since_hand_change_new[1] = 0.0
            force_magnitude = torch.norm(ee_force_external)
            if (distance_ee_drawer[0] < 0.03 and uncertainty_dist < 0.25 and time_since_hand_change[0] > 1.0) or (hand_synergy_activation > 0.5):
                FT_tresh = torch.minimum(time_since_hand_change[1] * 0.75, torch.ones_like(time_since_hand_change[1]) * 2.25)
                likelihood_from_hand_and_force = torch.clip(1 - torch.exp(-(force_magnitude - FT_tresh)), 0, 1) * hand_synergy_activation
                if (likelihood_grasped_drawer < 0.1) and (likelihood_from_hand_and_force < 0.1) and (hand_synergy_activation > 0.5):
                    likelihood_from_hand_and_force = (likelihood_from_hand_and_force.detach() -
                                                   hand_synergy_activation + hand_synergy_activation.detach())
            innovation = c_func(likelihood_grasped_drawer, distance_ee_drawer, uncertainty_dist, hand_synergy_activation, ee_force_external, time_since_hand_change_new)
            if innovation < - 0.05:
                new_likelihood = likelihood_grasped_drawer + 0.1 * innovation   # slowly decide that it isnt grasped anymore
            else:
                new_likelihood = likelihood_grasped_drawer + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.000000001, 0.9999999)
            return (new_likelihood), (new_likelihood, time_since_hand_change_new)

        return f_func, ["likelihood_grasped_drawer"], ["likelihood_grasped_drawer", "time_since_hand_change"]

    def initial_definitions(self):
        """
        Initialize the quantities for grasp state estimation.
        """
        self.quantities["likelihood_grasped_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["time_since_hand_change"] = torch.zeros(2, dtype=self.dtype, device=self.device)


class VisibleEstimator(EstimationComponent):
    """
    Estimator for the visibility of the drawer.
    This component estimates the likelihood that the drawer is currently visible
    from the camera's perspective.
    """
    state_dim = 1

    def initialize_quantities(self):
        """
        Initialize the quantities for visibility estimation.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c_visible = self.connections["VisibleLikelihood"]
        with c_visible.lock:
            if c_visible.connected_quantities_initialized["pose_ee"] and \
                    c_visible.connected_quantities_initialized["position_drawer"] and \
                    c_visible.connected_quantities_initialized["uncertainty_ee"] and \
                    c_visible.connected_quantities_initialized["uncertainty_drawer"]:
                pose_ee = c_visible.connected_quantities["pose_ee"]
                position_drawer = c_visible.connected_quantities["position_drawer"]
                c_func = c_visible.c_func
            else:
                return False
        self.quantities["likelihood_visible_drawer"] = torch.clip(
            c_func(torch.tensor([0.95], dtype=self.dtype, device=self.device), pose_ee, position_drawer), 0.02, 0.98)
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function for visibility.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: Function that updates the visibility likelihood
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["VisibleLikelihood"].c_func

        def f_func(likelihood_visible_drawer, pose_ee, uncertainty_ee, position_drawer, uncertainty_drawer):
            """
                Update the visibility likelihood estimate.
                
                Args:
                    likelihood_visible_drawer: Current visibility likelihood
                    pose_ee: Current end-effector pose
                    uncertainty_ee: End-effector uncertainty
                    position_drawer: Current drawer position
                    uncertainty_drawer: Drawer position uncertainty
                
                Returns:
                    tuple: Updated visibility likelihood
            """
            innovation = c_func(likelihood_visible_drawer, pose_ee, position_drawer)
            new_likelihood = likelihood_visible_drawer + 0.5 * innovation
            new_likelihood = torch.clip(new_likelihood, 0.000000001, 0.9999999)
            return (new_likelihood), (new_likelihood)

        return f_func, ["likelihood_visible_drawer"], ["likelihood_visible_drawer"]

    def initial_definitions(self):
        """
        Initialize the quantities for visibility estimation.
        """
        self.quantities["likelihood_visible_drawer"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)


class ExternalForceEstimator(EstimationComponent):
    """
    Estimator for external forces on the end-effector.
    This component estimates the 3D external force vector acting on the end-effector,
    compensating for the end-effector's own weight.
    """
    state_dim = 3

    def initialize_quantities(self):
        self.quantities["ee_force_external"] = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        return self.connections["OwnWeightCompensation"].connected_quantities_initialized["pose_ee"]    # we need ee pose to not produce garbage

    def define_estimation_function_f(self):
        c_func = self.connections["OwnWeightCompensation"].c_func

        def f_func(pose_ee, ee_force_measured, ee_force_external):
            innovation = c_func(pose_ee, ee_force_measured, ee_force_external)
            new_force = ee_force_external - 0.2 * innovation
            return (new_force,), (new_force,)

        return f_func, ["ee_force_external"], ["ee_force_external"]

    def initial_definitions(self):
        self.no_differentiation = True
        self.quantities["ee_force_external"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)


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
        if self.initial_rotation_xy is None:
            self.quantities["kinematic_joint"][0] = - torch.pi / 2 * 0.5
            pass
        else:
            # depends on rotation of robot to drawer cabinet
            self.quantities["kinematic_joint"][0] = self.initial_rotation_xy
        self.quantities["kinematic_joint"][3:6] = self.connections["DrawerKinematics"].connected_quantities["position_drawer"]
        if self.initial_uncertainty_scale is None:
            self.quantities["uncertainty_joint"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device)
        else:
            self.quantities["uncertainty_joint"] = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) * self.initial_uncertainty_scale
        self.quantities["uncertainty_joint"][3:6,3:6] = self.connections["DrawerKinematics"].connected_quantities["uncertainty_drawer"]
        if self.sample_init_mean:
            init_angles = torch.distributions.multivariate_normal.MultivariateNormal(self.quantities["kinematic_joint"][0:2],self.quantities["uncertainty_joint"][:2,:2]).sample()
            self.quantities["kinematic_joint"][0:2] = init_angles
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
            unlikelihood_grasped = gradient_preserving_clipping((0.2 - likelihood_grasped_drawer) / 0.2, 0.0000000001, 1.0)
            likelihood_grasped = gradient_preserving_clipping(likelihood_grasped_drawer, 0.000000000001, 1.0)
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
