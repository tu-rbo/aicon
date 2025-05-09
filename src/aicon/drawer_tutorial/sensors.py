"""
Sensors for the drawer experiment.
This module defines sensor components for the drawer manipulation environment,
including camera-based position sensing.
"""

from typing import Dict, Union

import torch
import numpy as np

from aicon.base_classes.connections import ActiveInterconnection
from aicon.drawer_experiment.util import get_sine_of_angles
from aicon.base_classes.components import SensorComponent


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
        """
        Obtain end-effector position measurement from the simulation environment.
        Returns:
            bool: True if measurements were successfully obtained.
        """
        ee_pos_np = self.sim_env_pointer.env.get_ee_pos()
        curr_sim_time = self.sim_env_pointer.get_sim_time()
        self.timestamp = torch.tensor(curr_sim_time)
        self.quantities["ee_pos_meas"] = torch.tensor(ee_pos_np, dtype=self.dtype, device=self.device)
        return True

    def initial_definitions(self):
        self.quantities["ee_pos_meas"] = torch.empty(self.state_dim, dtype=self.dtype, device=self.device)


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