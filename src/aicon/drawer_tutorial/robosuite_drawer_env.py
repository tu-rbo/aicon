from collections import OrderedDict
import numpy as np
from copy import deepcopy
from loguru import logger

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer
import robosuite.utils.transform_utils as T

# Import the CabinetObject from robosuite_task_zoo
from robosuite_task_zoo.models.hammer_place import CabinetObject

class DrawerOpenEnv(SingleArmEnv):
    """
    Environment for opening a drawer using a single robot arm.
    
    This environment is based on the HammerPlaceEnv but simplified to focus
    only on opening the drawer.
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        contact_threshold=2.0,
        initial_qpos=None,
        **kwargs
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = (-0.2, 0, 0.90)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None
        
        self.objects = []

        # Store the custom initial joint positions
        self.custom_initial_qpos = initial_qpos

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            # **kwargs # Pass any other relevant kwargs if needed, but not initial_qpos
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Opening: in [0, 0.75], proportional to how much the drawer has been opened

        Note that a successfully completed task (drawer opened) will return 1.0 regardless of whether the environment
        is using sparse or shaped rewards

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # Check if drawer is open
        drawer_opened = self._check_success()
        
        # Sparse reward
        if drawer_opened:
            reward = 1.0
        # Dense reward
        elif self.reward_shaping:
            # Get distance between gripper and handle
            gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            handle_pos = self.sim.data.site_xpos[self.cabinet_handle_site_id]
            dist = np.linalg.norm(gripper_pos - handle_pos)
            
            # Reward for reaching the handle
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            
            # Reward for opening the drawer
            # Drawer joint position (0 when closed, negative when opened)
            drawer_pos = self.sim.data.qpos[self.cabinet_qpos_addrs]
            # Normalize drawer position to [0, 1] where 1 is fully open
            norm_drawer_pos = min(1.0, abs(drawer_pos) / 0.2)  # 0.2 is approximately the fully open position
            opening_reward = 0.75 * norm_drawer_pos
            reward += opening_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )
        
        # Custom materials
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        )
        
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        )

        # Create cabinet object
        self.cabinet_object = CabinetObject(name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj()
        cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        mujoco_arena.table_body.append(cabinet_object)
        
        # Add materials to cabinet
        for obj_body in [self.cabinet_object]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material)
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[],
        )
        
        self.objects = [self.cabinet_object]
        self.model.merge_assets(self.cabinet_object)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        
        # Cabinet joint and site references
        self.cabinet_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        self.cabinet_object_id = self.sim.model.body_name2id(self.cabinet_object.root_body)
        
        # Get the drawer body (which is a child of the cabinet)
        # The drawer is the first child body of the cabinet
        drawer_body_name = None
        for body_id in range(self.sim.model.nbody):
            if self.sim.model.body_parentid[body_id] == self.cabinet_object_id:
                drawer_body_name = self.sim.model.body_id2name(body_id)
                if "drawer" in drawer_body_name.lower():
                    self.drawer_body_id = body_id
                    break
        
        if not hasattr(self, "drawer_body_id"):
            # If we can't find a drawer body, use the cabinet body as fallback
            self.drawer_body_id = self.cabinet_object_id
            print("Warning: Couldn't find drawer body, using cabinet body instead")
        
        # Get the site id for the cabinet handle
        self.cabinet_handle_site_id = self.sim.model.site_name2id("CabinetObject_default_site")
        
        # Store object body IDs
        self.obj_body_id = {}        
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True
        
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            sensors = []
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for (i, obj) in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")

            sensors += obj_sensors
            names += obj_sensor_names
            
        @sensor(modality=modality)
        def gripper_contact(obs_cache):
            return self._has_gripper_contact

        @sensor(modality=modality)
        def force_norm(obs_cache):
            return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)

        @sensor(modality="object")
        def drawer_pos(obs_cache):
            # The drawer joint value represents how far it has been pulled out
            # Negative value means opened outward
            return self.sim.data.qpos[self.cabinet_qpos_addrs]

        @sensor(modality="object")
        def drawer_orientation(obs_cache):
            # Use the cabinet orientation since that's what matters for the drawer movement direction
            quat = T.convert_quat(self.sim.data.body_xquat[self.cabinet_object_id], to="xyzw")
            # Get the y-axis direction of the cabinet which is the drawer's opening direction
            rot_matrix = T.quat2mat(quat)
            # Extract the y-axis (column 1)
            drawer_direction = rot_matrix[:, 1]
            return np.array(drawer_direction, dtype=np.float32)

        # Add cabinet position and orientation observations to match original environment
        @sensor(modality="object")
        def cabinet_pos(obs_cache):
            # Return cabinet position
            return np.array(self.sim.data.body_xpos[self.cabinet_object_id])

        @sensor(modality="object")
        def cabinet_quat(obs_cache):
            # Return cabinet orientation as quaternion
            return T.convert_quat(self.sim.data.body_xquat[self.cabinet_object_id], to="xyzw")

        sensors += [gripper_contact, force_norm, drawer_pos, drawer_orientation, 
                    cabinet_pos, cabinet_quat]
        names += [f"{pf}contact", f"{pf}eef_force_norm", "drawer_pos", "drawer_orientation", 
                  "CabinetObject_pos", "CabinetObject_quat"]

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq
                )
                
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

        # Apply custom initial joint positions if provided
        if self.custom_initial_qpos is not None:
            self.robots[0].set_robot_joint_positions(self.custom_initial_qpos)

    def _check_success(self):
        """
        Check if drawer has been opened.

        Returns:
            bool: True if drawer has been opened beyond a threshold
        """
        # Drawer joint position (0 when closed, negative when opened)
        drawer_pos = self.sim.data.qpos[self.cabinet_qpos_addrs]
        
        # Consider the drawer open if it has been pulled out by at least 0.1 units
        # The drawer's joint value becomes more negative as it opens
        return drawer_pos < -0.1

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4:
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)
        
        self._recent_force_torque = []
        obs, reward, done, info = super().step(action)
        info["history_ft"] = np.clip(np.copy(self._history_force_torque.buf), a_min=None, a_max=2)
        info["recent_ft"] = np.array(self._recent_force_torque)
        done = self._check_success()
        return obs, reward, done, info
        
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

        self._history_force_torque.push(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        self._recent_force_torque.append(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        
    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque
            
        return reward, done, info
    
    
    def get_drawer_handle_pos(self, offset=None):
        if offset is None:
            offset = np.array([ 0.01042636,  0.16967794, -0.09888733])
        cabinet_pos = self.sim.data.body_xpos[self.cabinet_object_id]
        handle_pose = cabinet_pos - offset
        return handle_pose
        
    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surpassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact surpasses given threshold magnitude
        """
        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold

    def print_debug_info(self):
        """
        Print debug information about the drawer and cabinet.
        Useful for understanding the state of the objects.
        """
        print("\n===== DRAWER DEBUG INFO =====")
        # Print cabinet information
        print(f"Cabinet Body ID: {self.cabinet_object_id}")
        print(f"Cabinet Position: {self.sim.data.body_xpos[self.cabinet_object_id]}")
        quat = self.sim.data.body_xquat[self.cabinet_object_id]
        print(f"Cabinet Orientation (WXYZ): {quat}")
        
        # Print drawer joint information 
        print(f"Drawer Joint Addr: {self.cabinet_qpos_addrs}")
        print(f"Drawer Joint Value: {self.sim.data.qpos[self.cabinet_qpos_addrs]}")
        
        # Print handle site information
        handle_pos = self.sim.data.site_xpos[self.cabinet_handle_site_id]
        print(f"Handle Site ID: {self.cabinet_handle_site_id}")
        print(f"Handle Position: {handle_pos}")
        
        # Print available joints for the cabinet object
        print("\nCabinet Joints:")
        for joint_name in self.cabinet_object.joints:
            joint_id = self.sim.model.joint_name2id(joint_name)
            joint_addr = self.sim.model.get_joint_qpos_addr(joint_name)
            joint_type = self.sim.model.jnt_type[joint_id]
            joint_value = self.sim.data.qpos[joint_addr]
            print(f"  - {joint_name}: ID={joint_id}, Type={joint_type}, Addr={joint_addr}, Value={joint_value}")
        
        print("============================\n")

    def get_state_vector(self, obs):
        drawer_pos = obs["drawer_pos"] if isinstance(obs["drawer_pos"], np.ndarray) else np.array([obs["drawer_pos"]])
        return np.concatenate([
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            drawer_pos,
            obs["drawer_orientation"]  # This now contains the direction vector
        ]) 

    def get_sim_time(self):
        return self.sim.data.time
        
    def get_ee_pos(self) -> np.ndarray:
        """Returns the current end-effector position."""
        ee_pose = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        return ee_pose

    def get_ee_force_magnitude(self) -> float:
        """Returns the magnitude of the force measured at the end-effector."""
        force = self.robots[0].get_sensor_measurement("gripper0_force_ee")
        return np.linalg.norm(force)

    def get_gripper_qpos(self) -> np.ndarray:
        """Returns the current joint positions of the gripper fingers."""
        # Note: This assumes a single robot. Access gripper state directly.
        # The observation 'robot0_gripper_qpos' might be more readily available after a step,
        # but this provides direct access if needed before/between steps.
        return self.robots[0].gripper.current_joint_positions