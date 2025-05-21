import time
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import robosuite as suite
import torch
from robosuite.devices import Keyboard, SpaceMouse
from robosuite.utils.input_utils import input2action
from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion

# Import our custom environment
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.middleware.python_sequential import build_components, run_component_sequence


def setup_env(device_type, initial_qpos=None):
    device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    # Create our custom environment
    env = DrawerOpenEnv(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="agentview",
        horizon=100,
        control_freq=30,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSITION"),
        initial_qpos=initial_qpos,
    )

    env = VisualizationWrapper(env)
    env.reset()
    device.start_control()
    return env, device


def main(device, env, rec_save_path=None):
    env.reset()
    robot = env.robots[0]
    device.start_control()

    # Print debug information about the drawer
    env.env.print_debug_info()

    # Get the observation using the original method
    env_obs = env._get_observations()

    # Access cabinet information in the same way as the original environment
    cabinet_position = env_obs["CabinetObject_pos"]
    cabinet_orientation = quat2mat(env_obs["CabinetObject_quat"])

    print(f"\nCabinet position from obs: {cabinet_position}")
    print(f"Cabinet orientation matrix from obs:\n{cabinet_orientation}")

    # setup aicon
    component_building_functions, connection_building_functions, frame_rates = (
        get_building_functions_basic_drawer_motion(env)
    )
    components = build_components(component_building_functions)
    gripper_component = components["GripperAction"]
    gripper_velo = components["EEVelocities"]

    # Start sim
    curr_t = 0
    while True:
        action, grasp = input2action(
            device=device, robot=robot, active_arm="right", env_configuration="single-arm-opposed"
        )
        run_component_sequence(components, torch.tensor(curr_t))
        curr_commanded_vel = gripper_velo.quantities["action_velo_ee"]
        curr_commanded_gripper = gripper_component.quantities["gripper_activation"]
        action = np.concatenate(
            [curr_commanded_vel.cpu().numpy(), [2 * curr_commanded_gripper.squeeze().cpu().numpy() - 1]]
        )

        obs, rew, done, info = env.step(action)

        # Print success state and joint info
        if done:
            print("\n*** SUCCESS! Drawer opened. ***\n")

        curr_t += env.control_timestep
        env.render()


def run_demo():
    # Define the desired initial joint positions for the Panda robot (7 joints)
    initial_panda_qpos = np.array([-0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7])
    # initial_panda_qpos = None

    # Pass the initial pose to the setup function
    env, device = setup_env("keyboard", initial_qpos=initial_panda_qpos)
    main(device, env)


if __name__ == "__main__":
    run_demo()
