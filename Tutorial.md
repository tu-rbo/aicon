
# Drawer Tutorial AICON

## Our First Estimation Component

AICON is a completely novel approach to problem solving and can be challenging to understand both conceptually and in implementation. In this tutorial, we will rebuild the drawer experiment in simulation step by step and explain how to approach problems with AICON.

Opening a drawer is a complex task: we must locate the drawer, orient ourselves, move to the handle, grasp it, estimate its kinematic structure, and pull the drawer. Each subtask is non-trivial, and planning them individually is not robust to environmental changes. 

AICON works by computing gradients through Components to generate actions and can therefore adapt to these changes in realtime. There are three types of components: `SensorComponent`, `ActionComponent`, and `EstimationComponent`. 

But where do we start? Let's start simple: how do we move the robot? We move the robot by commanding the end-effector to a target position. Therefore, we need two estimators: one for our end-effector position and one for the target position.

Let's begin with our `EndEffectorPoseEstimator`, which inherits from `EstimationComponent`:

```python
class EEPoseEstimator(EstimationComponent):
	state_dim=3
	def define_estimation_function_f(self):
		def f_func():
			mu_new = None
			Sigma_new = None
			return (mu_new, Sigma_new), (mu_new, Sigma_new)
		return f_func, _, _
	
	def initial_definitions(self):
		pass
	def initialize_quantities(self):
		pass
```

An `EstimationComponent` performs recursive estimation with a Kalman filter and implements three functions:

- `initial_definitions`: defines the estimator's quantities (state and uncertainty).
- `initialize_quantities`: assigns initial values, often from sensors.
- `define_estimation_function_f`: returns the filter's predict-update function.

 `initial_definitions`: 

Since the end-effector position is a 3D coordinate, `state_dim = 3`. We need `pose_ee` and `uncertainty_ee` quantities:

```python
   def initial_definitions(self):
        """
        Initialize the quantities for end-effector pose estimation.
        """
        self.quantities["pose_ee"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.quantities["uncertainty_ee"] = torch.zeros(self.state_dim, self.state_dim, dtype=self.dtype,
                                                        device=self.device)
```

`initilaize_quantities`: We initialize the state from proprioceptive measurements via a `DirectMeasurement` connection to the sensor:

```python
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
```

Because connections are shared resources in a multi-thread system, we wait for availability. If a measurement is ready, we assign it; otherwise, we return `False` and retry later. We set a low initial uncertainty (0.001).

- `define_estimation_function_f`: This method returns the EKF's predict-update function. In the prediction step, we advance the state using forward kinematics; in the update step, we incorporate sensor measurements:
    
     And our estimation loop looks like this:
    
    ```python
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
                mu_pred, Sigma_pred = predict_ekf_other_quantity(c_action, pose_ee, uncertainty_ee, action_velo_ee, torch.eye(3, dtype=self.dtype, device=self.device) * 0.01)
                mu_new, Sigma_new = update_ekf(c_proprio, mu_pred, Sigma_pred, ee_pos_meas,
                                               torch.eye(3, dtype=self.dtype, device=self.device) * 0.01,
                                               torch.eye(3, dtype=self.dtype, device=self.device) * 0.01)
                Sigma_new = Sigma_new.detach()  # currently no grad possible
                # Work around
                delta = c_proprio(mu_new, ee_pos_meas).detach()
                mu_new = mu_new + delta
                return (mu_new, Sigma_new), (mu_new, Sigma_new)
    
        return f_func, ["pose_ee", "uncertainty_ee"], ["pose_ee", "uncertainty_ee"]
    ```
    
    Here, we predict the state via forward kinematics and update with proprioceptive measurements. Finally, we force the estimate to match the measurement (with `detach()`) so that the position remains differentiable.
    
    > Lesson: Everything in AICON must be differentiable!
    >  
    
    Lets write another Estimator for our target point.

## Estimator for the Drawer Position

Next, we estimate the drawer handle's position:

```python
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
```

Here, we assume the drawer is stationary (no forward motion) but increase uncertainty over time. We use visual measurements and kinematic information when the drawer is grasped.

We incorporate two `ActiveInterconnection` links: one between our sensor measurements and the `GraspedComponent`, and another directly to our `GraspedEstimator`. The sensor measurements feed into our state estimator, while the gripper-pose information—used only when the drawer is grasped—further refines the estimate, since in that situation the two states become tightly coupled. We'll discuss this coupling in more detail later.

### **Estimator For the Kinematic Joint**

To open the drawer, we maximize the displacement of its kinematic joint. We estimate its state (angle, displacement, position):

```python
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

```

We define three core functions in this tutorial, assuming the kinematic joint angle is known and concentrating solely on translating the drawer handle to open the drawer. First, we initialize the joint's position and uncertainty to reflect our estimated drawer position. Since the joint cannot move on its own, we omit a forward model and rely entirely on our update step—specifically the `update_shifting_ekf` routine—to shift the joint position whenever the drawer is grasped. To control which measurements influence this update, we introduce two grasp-dependent functions, `likelihood_grasped` and `unlikelihood_grasped`, which gate the information flow based on whether the gripper holds the drawer. Finally, we add an `EstimationComponent` called `graspedLikelihood` that computes the probability of a successful grasp and selects the appropriate update function. 


## Estimator for Grasp Likelihood

Finally, we estimate the likelihood that the drawer is grasped, using distance, force/torque measurements, and gripper activation:

```python
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

```

The grasped likelihood estimator only has one state dimension and does not require a Kalman filter. Instead, we use a heuristic-based approach and accumulate evidence based on the distance between gripper and drawer, as well as the Force Torque measurements and the gripper activation. The `c_func`looks like this 

```python
def connection_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            expected_dist = torch.norm(position_drawer - pose_ee)
            likelihood_given_dist = torch.exp(-expected_dist * 4)
            innovation_from_dist = likelihood_given_dist - likelihood_grasped_drawer

            # hand and force measurements are only relevant if we are close (otherwise from other source...)
            if (expected_dist < 0.05 and ee_force_mag_meas > 10):
                # under 2N is just FT noise
                likelihood_from_hand_and_force = torch.clip(1 - torch.exp(-(ee_force_mag_meas - 5)), 0, 1) * gripper_activation
                # but we need an open hand to increase this likelihood
                # so we generate a negative gradient
                if (likelihood_grasped_drawer < 0.1) and (likelihood_from_hand_and_force < 0.1) and (gripper_activation > 0.5):
                    # print("Should open")
                    likelihood_from_hand_and_force = (likelihood_from_hand_and_force.detach() -
                                                      gripper_activation + gripper_activation.detach())
            else:
                # essentially no likelihood
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_dist) * 0.00000001

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()
            return innovation_from_dist + innovation_from_hand_and_force
            
```

The closer the gripper gets to the handle, the higher our likelihood of a grasp—but true grasping depends on more than proximity alone. Once we're within range, we begin to incorporate force-torque (FT) sensor readings and gripper activation status. In practice, being close enough, detecting contact through the FT sensor, and triggering the gripper all combine to boost our grasped-state probability.

With these components defined, we've set up all the states needed for this simplified example. We've made a number of assumptions to keep the tutorial straightforward; a real-world system would handle many of these details differently. For a more complete implementation, see the `drawer_experiment` module in our GitHub repository:

[https://github.com/tu-rbo/aicon/tree/sim_demo_dev/src/aicon/drawer_experiment](https://github.com/tu-rbo/aicon/tree/sim_demo_dev/src/aicon/drawer_experiment)

## Adding Sensors

Sensors inherit from `SensorComponent` and implement `initial_definitions` and `obtain_measurements`.

```python
class EEPoseSensor(SensorComponent):
    """
    Sensor for obtaining the end-effector position from the simulation environment.
    """
    state_dim = 3

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False, sim_env_pointer = None):
        self.sim_env_pointer = sim_env_pointer
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)

    def obtain_measurements(self) -> bool:
        return True

    def initial_definitions(self):
				pass
```

We give our end-effector (EE) estimator a "position_ee" quantity that's driven by proprioceptive readings. In simulation, we grab these directly from our sim environment; on a real robot, you'd pull the EE pose via ROS TF instead.

`obtain_measurements`: On each update, we query the sim for the current EE pose and timestamp, store them in our quantities, and return True to indicate success:

```python
def obtain_measurements(self) -> bool:
		ee_pos_np = self.sim_env_pointer.env.get_ee_pos()
    curr_sim_time = self.sim_env_pointer.get_sim_time()
    self.timestamp = torch.tensor(curr_sim_time)
    self.quantities["ee_pos_meas"] = torch.tensor(ee_pos_np, dtype=self.dtype, device=self.device)
		return True
```

`initial_definitions`: At startup, we declare an empty tensor for the measured EE position:

```python
def initial_definitions(self):
    self.quantities["ee_pos_meas"] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)
```

With these two methods, our `EEPositionSensor` feeds EE-pose data into the estimator. In the next section, we add a similar sensor for the drawer postion.

```python
class DrawerPoseSenser(SensorComponent):
    """
    Sensor for obtaining the drawer handle position directly from the simulation environment.
    """

    state_dim = 3

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False, sim_env_pointer = None):
        """
        Initialize the drawer pose sensor.
        
        Args:
            name: Name of the sensor
            connections: Dictionary of active interconnections
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
            sim_env_pointer: Pointer to the simulation environment to get measurements from.
        """
        self.sim_env_pointer = sim_env_pointer
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)

    def obtain_measurements(self) -> bool:
        """
        Obtain drawer handle position measurements from the simulation environment.
        
        Returns:
            bool: True if measurements were successfully obtained, False otherwise
        """
        drawer_pos = self.sim_env_pointer.env.get_drawer_handle_pos()
        curr_sim_time = self.sim_env_pointer.get_sim_time()

        self.timestamp = torch.tensor(curr_sim_time)
        self.quantities["drawer_pos_meas"] = drawer_pos
        return True

    def initial_definitions(self):
        """
        Initialize the sensor's quantities.
        Sets up the relative position quantity with the correct dimensions.
        """
        self.quantities["drawer_pos_meas"] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)
```

**Gripper State Sensor**

When we defined the GraspedLikelihoodEstimator, we leveraged the gripper's activation status and force/torque measurements, both of which we obtain directly from our simulation environment.

```python
class GripperStateSensor(SensorComponent):
    """
    Sensor for obtaining the gripper joint positions (state) from the simulation environment.
    """

    # Assuming Panda gripper with 2 joints
    state_dim = 2 

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False, sim_env_pointer = None):
        self.sim_env_pointer = sim_env_pointer
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)

    def obtain_measurements(self) -> bool:
        """
        Obtain gripper joint position measurements from the simulation environment.
        Returns:
            bool: True if measurements were successfully obtained.
        """
        gripper_qpos_np = self.sim_env_pointer.get_gripper_qpos()
        curr_sim_time = self.sim_env_pointer.get_sim_time()
        self.timestamp = torch.clone(curr_sim_time)
        # Ensure state_dim matches the gripper qpos dimension
        if len(gripper_qpos_np) != self.state_dim:
             print(f"Warning: GripperStateSensor expected state_dim={self.state_dim}, but got {len(gripper_qpos_np)} elements.")
             # Adjust state_dim or handle mismatch appropriately
             # For now, we'll assume the first self.state_dim elements if mismatched
             if len(gripper_qpos_np) > self.state_dim:
                 gripper_qpos_np = gripper_qpos_np[:self.state_dim]
             else: # Pad with zeros if too few - might need better handling
                 gripper_qpos_np = np.pad(gripper_qpos_np, (0, self.state_dim - len(gripper_qpos_np)))

        self.quantities["gripper_qpos_meas"] = torch.tensor(gripper_qpos_np, dtype=self.dtype, device=self.device)
        return True

    def initial_definitions(self):
        self.quantities["gripper_qpos_meas"] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)
```

```python
class EEForceSensor(SensorComponent):
    """
    Sensor for obtaining the end-effector force magnitude from the simulation environment.
    """
    state_dim = 1

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection],
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False, sim_env_pointer = None):
        self.sim_env_pointer = sim_env_pointer
        super().__init__(name, connections, dtype, device, mockbuild=mockbuild)

    def obtain_measurements(self) -> bool:
        """
        Obtain end-effector force magnitude measurement from the simulation environment.
        Returns:
            bool: True if measurements were successfully obtained.
        """
        ee_force_mag_np = self.sim_env_pointer.get_ee_force_magnitude()
        curr_sim_time = self.sim_env_pointer.get_sim_time()
        self.timestamp = torch.tensor(curr_sim_time)
        self.quantities["ee_force_mag_meas"] = torch.tensor([ee_force_mag_np], dtype=self.dtype, device=self.device)
        return True

    def initial_definitions(self):
        self.quantities["ee_force_mag_meas"] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)
```

## Defining Goals

Our goal is to open the drawer to a specific joint value. We define a `Goal` that measures the difference between the current joint state and the desired open value:

```python
lass DrawerOpenViaJointGoal(Goal):
    """
    Goal to open a drawer to a specific joint value.
    The goal is considered fulfilled when the drawer joint reaches the target value.
    """

    def __init__(self, is_active: bool, open_value: float = -0.5,
                 dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        self.open_value = torch.tensor(open_value, dtype=dtype, device=device)
        super().__init__(name="ReduceJointStateDifference", is_active=is_active, mockbuild=False)
	def define_goal_cost_function(self):
		def goal_func():
			pass
		return goal
	
		
```

This class returns a goal function that we want to minimize. To open a drawer we want to maximize the displacement of our kinematic joint which we stored in our `kinematic_joint` vector. Therefore our goal function looks like this.

```python
def goal_func(kinematic_joint):
    dist = kinematic_joint[2] - open_value
    cost = dist
    return cost, cost
```

However, for gradients to flow trough our estimators, they need to be interconnected.  


## The Heart of AICON: Active Interconnections

When we want to generate actions—such as moving or grasping—we must propagate gradients from our goal function, through the `EstimationComponents`, to our `ActionComponents`. To do this, every estimator and connection in the system must be expressed as a differentiable function.

The key question is: which estimators should be directly connected? Should we link the `EEPoseEstimator` (which estimates end‐effector pose) directly to the `KinematicJointEstimator` (which estimates joint displacement). In isolation, this makes little sense—until the gripper has actually grasped the handle, changes in end‐effector pose cannot affect drawer displacement.

However, once the handle is gripped, a change in end‐effector pose will indeed move the drawer and alter the kinematic joint's value. To capture this relationship, we can connect to the `GraspedEstimator` and `DrawerPositionEstimator`, and the `DrawerPositionEstimator` encodes all the necessary information. Therefore, we will connect the `KinematicJointEstimator`<- `DrawerPositionEstimator` via an `ActiveInterconnection`.

```python

class KinematicJointConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"kinematic_joint": (6,), "uncertainty_joint": (6,6,),
                                "position_drawer": (3,), "uncertainty_drawer": (3,3,), "likelihood_grasped_drawer": (1,),
                                "time_since_hand_change": (2,)},
                         dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):

        def connection_func(kinematic_joint, position_drawer):
            pass

        return None
```

An `ActiveInterconnection` returns an implicit connection function that leverages information from one estimator to enhance another. It is implicit, in the sense that the difference between the used quantities should be zero. In other words, it allows us to calculate an innovation term for our current Kalman Filter based on the estimates of other `EstimationComponents`.

The kinematic joint is directly linked to our drawer position, because shifting the initial point of our kinematic joint by the displacement along the orientation vector should be equal to the position of the drawer. Therefore, our connection function looks like this

```python
def connection_func(kinematic_joint, position_drawer):
	      azimuth, elevation, state, initial_point = kinematic_joint[0:1], kinematic_joint[1:2], kinematic_joint[2:3], kinematic_joint[3:]
        orientation_vector = torch.cat([torch.cos(azimuth) * torch.sin(elevation),
                                        torch.sin(azimuth) * torch.sin(elevation),
                                        torch.cos(elevation)], dim=0)
        translation = orientation_vector * state
        predicted_point = initial_point + translation
        return predicted_point - position_drawer
```

With our estimators now linked, we can share information between them and calculate how changes in our goal affect other `EstimationComponents`. Additionally, the `KinematicJointEstimator` already uses information from the `GraspedEstimator` in its internal function, so these two are connected.. 


### Connecting the DrawerPositionEstimator

As stated earlier, the drawer's behavior depends on our sensor measurements, as we use the difference between the measurements and our current prediction (the innovation) to refine our belief of the drawer's state. Consequently, we establish a connection between the `DrawerPoseSensor` and our `DrawerPositionEstimator` through the `DrawerDirectMeasurement`.

```python
class DrawerDirectMeasurementConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"position_drawer": (3,), "uncertainty_drawer": (3,3), "drawer_pos_meas": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, drawer_pos_meas):
            # the measured and the current pose should be the same
            return drawer_pos_meas - position_drawer

        return connection_func
```

The core concept is that when the gripper has grasped the object, the difference between the relevant quantities should ideally be zero. We incorporate the likelihood of a successful grasp as an input to the `DrawerPositionEstimator`'s `f_func` and utilize this information during the estimation update.

Our second connection establishes a link between the `EEPoseEstimator` and the `DrawerPositionEstimator` specifically when the object is grasped. The rationale here is that the relative pose between the drawer and the gripper should be zero under a grasping condition.

```python

class EEDrawerGraspedConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3, 3), "position_drawer": (3,),
                                "uncertainty_drawer": (3, 3), "likelihood_grasped_drawer": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,
                         # for on/off switching connection we can require gradient signs
                         # if we want to move the position of the drawer more we want it to be grasped
                         # if we want to change ee more, we do not want it to be grasped
                         required_signs_dict={"position_drawer" : {"likelihood_grasped_drawer": 1},
                                              "uncertainty_drawer": {"likelihood_grasped_drawer": 1},
                                              "pose_ee" : {"likelihood_grasped_drawer": -1}})

    def define_implicit_connection_function(self):
        def connection_func(position_drawer, pose_ee):
            relative_vector = position_drawer - pose_ee
            return relative_vector

        return connection_func
```

### Connecting the GraspedEstimator

The `GraspedEstimator` is a central component with a detailed connection process. Our ability to manipulate the drawer depends on grasping it, which requires the gripper to be sufficiently close. Thus, we determine a grasp likelihood based on the distance between the gripper and the drawer – the closer the gripper, the higher the likelihood. Once the gripper is near the drawer, we start incorporating data from the Force/Torque (F/T) sensor. We alsoencode a hand opening mechanism to address situations where the gripper is closed, but the F/T readings and grasp likelihood remain low.

In other words, we calculate a grasp likelihood from the F/T measurements and match it against the grasp likelihood based on distance. To reduce these differences, the robot should move to the drawer and then close the gripper. The code implementation looks like this.


```python
class DistGraspHandConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "position_drawer": (3,), "uncertainty_drawer": (3,3), "uncertainty_ee": (3,3),
                                "likelihood_grasped_drawer": (1,), "gripper_activation": (1,), "ee_force_mag_meas": (1,)}, dtype=dtype,
                         device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):

        def connection_func(likelihood_grasped_drawer, pose_ee, position_drawer, ee_force_mag_meas, gripper_activation):
            expected_dist = torch.norm(position_drawer - pose_ee)
            likelihood_given_dist = torch.exp(-expected_dist * 4)
            innovation_from_dist = likelihood_given_dist - likelihood_grasped_drawer

            # hand and force measurements are only relevant if we are close (otherwise from other source...)
            if (expected_dist < 0.05 and ee_force_mag_meas > 10):
                # under 2N is just FT noise
                likelihood_from_hand_and_force = torch.clip(1 - torch.exp(-(ee_force_mag_meas - 5)), 0, 1) * gripper_activation
                # but we need an open hand to increase this likelihood
                # so we generate a negative gradient
                if (likelihood_grasped_drawer < 0.1) and (likelihood_from_hand_and_force < 0.1) and (gripper_activation > 0.5):
                    # print("Should open")
                    likelihood_from_hand_and_force = (likelihood_from_hand_and_force.detach() -
                                                      gripper_activation + gripper_activation.detach())
            else:
                # essentially no likelihood
                likelihood_from_hand_and_force = torch.ones_like(likelihood_given_dist) * 0.00000001

            innovation_from_hand_and_force = likelihood_from_hand_and_force - likelihood_given_dist.detach()
            return innovation_from_dist + innovation_from_hand_and_force

        return connection_func
```

### Adding the Remaining Connections

Now we established all connections between our `EstimationComponents` and only need to connect our `EEPoseEstimator`to the sensor measurements and more importantly to our `ActionComponent` to generate actions from the gradient. Connecting to the sensor is straightforward.

```python
class EEProprioConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3,3), "ee_pos_meas": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, ee_pos_meas):
            # the measured and the current pose should be the same
            return ee_pos_meas - pose_ee

        return connection_func

```

And the EE velocity and gripper position are linked through forward kinematics. ***(@ADRIAN this is not an implicit function right?)***

```python
class EEVeloConnection(ActiveInterconnection):

    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, {"pose_ee": (3,), "uncertainty_ee": (3,3), "action_velo_ee": (3,)}, dtype=dtype, device=device, mockbuild=mockbuild,)

    def define_implicit_connection_function(self):
        def connection_func(pose_ee, action_velo_ee):
           
            new_position = pose_ee + action_velo_ee
            return new_position

        return connection_func

```

Now we have established all connections in our AICON network, but our `ActionModule` is missing

## Actions from Gradients

We can now send gradients based on a cost through our AICON network and generate action that reduce the cost. Many necessities for real robots will be handled by the simulation. For a deeper dive we recommend the drawer_experiment code. The basic Action Component looks like this.  

```python
class VeloEEAction(ActionComponent):
    """
    End-effector velocity control action for the drawer experiment.
    This action component controls the robot's end-effector using translational velocity commands,
    with safety limits and gradient-based optimization.
    """

    state_dim = 3
    v_max_trans = 0.3

    def _start(self):
        pass

    def _stop(self):
        pass

    def determine_new_actions(self):
				pass

    def perform_gradient_descent(self, last_action: torch.Tensor, steepest_grad: torch.Tensor) -> torch.Tensor:
				pass

    def safety_limiting(self, t_part: torch.Tensor) -> torch.Tensor:
				pass

    def send_action_values(self) -> None:
				pass

    def initialize_quantities(self) -> bool:
				pass

    def initial_definitions(self):
				pass
```

`_start` , `_stop` , and `send_action_velocity` will be handled by our simulation. We again initialize our quantities and definitions

```python
def initialize_quantities(self) -> bool:
        """
        Initialize the action quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        new_action = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.quantities["action_velo_ee"] = new_action
        return True

def initial_definitions(self):
    """
    Define initial values for the action quantities.
    """
    self.quantities["action_velo_ee"] = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
```

Since, we dont have an inital action we set it to zero and wait for the gradient.

Once, we compute our first gradient we follow along the steepest descent

```python
def perform_gradient_descent(self, last_action: torch.Tensor, steepest_grad: torch.Tensor) -> torch.Tensor:
        """
        Perform gradient descent to update the translational velocity commands.
        
        Args:
            last_action: Previous translational velocity command
            steepest_grad: Steepest gradient for optimization
        
        Returns:
            torch.Tensor: Updated translational velocity (t_part)
        """
        gain = 0.1
        t_part = gradient_descent(gain, last_action[:3], steepest_grad[:3])
        return t_part
```

After that apply safety measures to the translational velocity commands

```python
   def safety_limiting(self, t_part: torch.Tensor) -> torch.Tensor:
        """
        Apply safety limits to translational velocity commands.
        
        Args:
            t_part: Translational velocity
        
        Returns:
            torch.Tensor: Limited translational velocity (t_part)
        """
        # safety limiting
        if torch.norm(t_part) > self.v_max_trans:
            t_part = t_part / torch.norm(t_part) * self.v_max_trans
        return t_part
```

and perform the new action by setting our quantity

```python
   def determine_new_actions(self):
        """
        Determine new velocity commands based on gradient descent optimization.
        """
        last_action = self.quantities["action_velo_ee"]
        steepest_grad, timestamps, trace = self.get_steepest_gradient("action_velo_ee",
                                                                    time_threshold=self.timestamp - torch.ones_like(self.timestamp) * 2.0)

        t_part = self.perform_gradient_descent(last_action, steepest_grad)

        t_part = self.safety_limiting(t_part)

        new_action_limited = torch.nan_to_num(t_part)

        self.quantities["action_velo_ee"] = new_action_limited
```

Since, we get a collection of gradients through the network we follow the steepest one.

## Setting Everything up

Now that we have all our building block we want to connect them and start acting. For this we use the `component_builder` that starts and connects all estiamtors. You have to define your components by giving them a name and naming the interconnections. For example, our DrawerPositionEstimator is connected to `DrawerDirectMeasurement`, `GraspedDrawerKinematics` and `DrawerKinematics` and we would specify it like this

```python
component_builders = { "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator("DrawerPosEstimator",
                                                                        connections={k: connection_builders[k] for k in
                                                                                     ("DrawerDirectMeasurement",
                                                                                      "GraspedDrawerKinematics",
                                                                                      "DrawerKinematics",
                                                                                     )},
                                                                        device=torch.device("cpu"),
                                                                        dtype=torch.double, mockbuild=mockbuild,
                                                                        ),}
```

For the whole network it looks like this.

```python

def get_building_functions_basic_drawer_motion(sim_env_pointer):
    connection_builders = build_connections()
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction("GripperAction",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor("EEForceSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "EEPosSensor": lambda mockbuild: EEPoseSensor("EEPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "DrawerPosSensor": lambda mockbuild: DrawerPoseSenser("DrawerPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DrawerDirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "EEVelocities": lambda mockbuild: VeloEEAction("EEVelocities",
                                                       connections={k: connection_builders[k] for k in
                                                                    ("ForwardKinematics",)},
                                                       device=torch.device("cpu"),
                                                       dtype=torch.double, mockbuild=mockbuild,
                                                       ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator("EEPoseEstimator",
                                                             connections={k: connection_builders[k] for k in
                                                                          ("ForwardKinematics",
                                                                           "DirectMeasurement",
                                                                           "GraspedDrawerKinematics",
                                                                           "GraspedLikelihood",
                                                                           )},
                                                             device=torch.device("cpu"),
                                                             dtype=torch.double, mockbuild=mockbuild),
        "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator("DrawerPosEstimator",
                                                                        connections={k: connection_builders[k] for k in
                                                                                     ("DrawerDirectMeasurement",
                                                                                      "GraspedDrawerKinematics",
                                                                                      "DrawerKinematics",
                                                                                     )},
                                                                        device=torch.device("cpu"),
                                                                        dtype=torch.double, mockbuild=mockbuild,
                                                                        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator("GraspLikelihoodEstimator",
                                                                       connections={k: connection_builders[k] for k in
                                                                                    ("GraspedDrawerKinematics",
                                                                                     "GraspedLikelihood",
                                                                                     "DrawerKinematics",
                                                                                     )},
                                                                       dtype=torch.double, mockbuild=mockbuild,
                                                                       ),
        "KinematicJointEstimator": lambda mockbuild: KinematicJointEstimator("KinematicJointEstimator",
                                                                         connections={k: connection_builders[k] for k in
                                                                                      ("DrawerKinematics",
                                                                                       )},
                                                                         dtype=torch.double, mockbuild=mockbuild,
                                                                         goals={"ReduceJointStateDifference": lambda d, t,
                                                                                                            mockbuild=False: DrawerOpenViaJointGoal(
                                                                             is_active=True, dtype=t, device=d,
                                                                             mockbuild=mockbuild)},
                                                                         )
    }
        frame_rates = {
        "GripperAction": 10,
        "EEForceSensor": 10,
        "EEPosSensor": 10,
        "DrawerPosSensor": 10,
        "EEVelocities": 10,
        "EEPoseEstimator": 10,
        "DrawerPosEstimator": 10,
        "GraspLikelihoodEstimator": 10,
        "KinematicJointEstimator": 10,
    }
    return component_builders, connection_builders, frame_rates
```

Now we only need to start the simulation. For this you can simply run `python run_demo.py`
