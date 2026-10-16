"""
Simulated Drawer Environment (robosuite)

A Panda arm with a parallel gripper in front of a cabinet with a sliding drawer. The drawer opens along the negative
y axis of the base frame and is open once its joint value is below SUCCESS_JOINT_VALUE. Besides the robosuite
interface, the environment provides the ground-truth quantities the simulated sensors read
(handle position, end-effector pose, force at the end-effector).

Written for robosuite 1.5.

Authors: Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import os
from copy import deepcopy

import numpy as np
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, add_material, new_body, new_geom
from robosuite.wrappers import VisualizationWrapper

# The drawer counts as open below this joint value [m]; the joint range of the cabinet model is [-0.04, 0].
SUCCESS_JOINT_VALUE = -0.03
# Offset from the drawer body to the handle point the sensors report [m].
HANDLE_OFFSET = np.array([0.01042636, 0.15967794, -0.02288733])

# Indicators drawn when rendering (estimated drawer position and estimated joint axis).
DRAWER_INDICATOR = "estimated_drawer_position"
AXIS_INDICATOR = "estimated_joint_axis"
AXIS_INDICATOR_LENGTH = 0.5


class CabinetObject(MujocoXMLObject):
    """Cabinet with one sliding drawer, from robosuite-task-zoo (MIT License, Copyright (c) 2021 ARISE Initiative)."""

    def __init__(self, name):
        super().__init__(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cabinet.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DrawerOpenEnv(ManipulationEnv):
    """Open the cabinet's drawer. `initial_qpos` sets the arm's initial joint positions; with `raised_cabinet` the
    cabinet stands on a 12 cm block."""

    def __init__(
        self,
        robots,
        controller_configs=None,
        has_renderer=False,
        render_camera="agentview",
        control_freq=30,
        horizon=1000,
        ignore_done=True,
        initial_qpos=None,
        raised_cabinet=False,
    ):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = (-0.2, 0, 0.90)
        self.custom_initial_qpos = initial_qpos
        self.raised_cabinet = raised_cabinet
        self.cabinet_object = None
        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=controller_configs,
            base_types="default",
            gripper_types="default",
            initialization_noise="default",
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            render_camera=render_camera,
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            # MuJoCo's own viewer; robosuite's other renderer draws offscreen, which needs EGL on Linux
            renderer="mjviewer",
            # the full physics of robosuite <= 1.4, with which the examples were tuned
            lite_physics=False,
        )

    def reward(self, action=None):
        """1 once the drawer is open, else 0."""
        return float(self._check_success())

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size, table_offset=self.table_offset, table_friction=(0.6, 0.005, 0.0001)
        )
        arena.set_origin([0, 0, 0])
        arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )
        arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657],
        )

        if self.raised_cabinet:
            block = new_body(name="block_under_drawer", pos=(0.2, 0.3, 0.05))
            block.append(
                new_geom(
                    name="block_geom",
                    type="box",
                    size=(0.102, 0.102, 0.05),
                    pos=(0.0, 0.0, 0.025),
                    group=1,
                    rgba=(1, 0, 0, 1),
                    contype=1,
                    conaffinity=1,
                )
            )
            arena.table_body.append(block)

        self.cabinet_object = CabinetObject(name="CabinetObject")
        cabinet = self.cabinet_object.get_obj()
        cabinet.set("pos", array_to_string((0.2, 0.30, 0.15 if self.raised_cabinet else 0.03)))
        arena.table_body.append(cabinet)
        materials = [
            CustomMaterial(
                texture="WoodLight",
                tex_name="lightwood",
                mat_name="MatLightWood",
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
            ),
            CustomMaterial(
                texture="WoodDark",
                tex_name="darkwood",
                mat_name="MatDarkWood",
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
            ),
            CustomMaterial(
                texture="Metal",
                tex_name="metal",
                mat_name="MatMetal",
                tex_attrib={"type": "cube"},
                mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
            ),
            CustomMaterial(
                texture="WoodRed",
                tex_name="redwood",
                mat_name="MatRedWood",
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
            ),
            CustomMaterial(
                texture="Ceramic",
                tex_name="ceramic",
                mat_name="MatCeramic",
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
            ),
        ]
        for material in materials:
            tex_element, mat_element, _, _ = add_material(
                root=self.cabinet_object.worldbody,
                naming_prefix=self.cabinet_object.naming_prefix,
                custom_material=deepcopy(material),
            )
            self.cabinet_object.asset.append(tex_element)
            self.cabinet_object.asset.append(mat_element)

        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[robot.robot_model for robot in self.robots], mujoco_objects=[]
        )
        self.model.merge_assets(self.cabinet_object)

    def _setup_references(self):
        super()._setup_references()
        self.cabinet_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        self.drawer_body_id = self.sim.model.body_name2id("CabinetObject_drawer_link")

    def _reset_internal(self):
        super()._reset_internal()
        if self.custom_initial_qpos is not None:
            self.robots[0].set_robot_joint_positions(self.custom_initial_qpos)
            self.robots[0].part_controllers["right"].update_initial_joints(np.array(self.custom_initial_qpos))

    def _check_success(self):
        return self.drawer_joint_value() < SUCCESS_JOINT_VALUE

    def step(self, action):
        obs, reward, _, info = super().step(action)
        return obs, reward, self._check_success(), info

    # --- ground truth for the simulated sensors ---

    def drawer_joint_value(self) -> float:
        """Drawer joint [m]: 0 when closed, negative when opened."""
        return float(self.sim.data.qpos[self.cabinet_qpos_addrs])

    def get_drawer_handle_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.drawer_body_id]).copy() - HANDLE_OFFSET

    def get_sim_time(self) -> float:
        return self.sim.data.time

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]).copy()

    def get_ee_pose(self) -> np.ndarray:
        """End-effector pose [x, y, z, rx, ry, rz] with the orientation as a rotation vector (axis * angle)."""
        R = np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id["right"]]).reshape(3, 3)
        theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if np.isclose(theta, 0.0):
            rotvec = np.zeros(3)
        else:
            rotvec = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta)) * theta
        return np.concatenate([self.get_ee_pos(), rotvec])

    def get_ee_force_magnitude(self) -> float:
        return float(np.linalg.norm(self.robots[0].ee_force["right"]))


def make_env(initial_qpos, controller="OSC_POSITION", raised_cabinet=False, render=False):
    """The environment at 30 Hz, wrapped for visualization, after a reset. `controller` is a robosuite default
    controller: "OSC_POSITION" takes [vx, vy, vz, gripper], "OSC_POSE" additionally three rotation velocities."""
    env = DrawerOpenEnv(
        robots="Panda",
        controller_configs=refactor_composite_controller_config(
            load_part_controller_config(default_controller=controller), "Panda", ["right"]
        ),
        has_renderer=render,
        initial_qpos=np.asarray(initial_qpos),
        raised_cabinet=raised_cabinet,
    )
    env = VisualizationWrapper(
        env,
        indicator_configs=[
            {"name": DRAWER_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [0, 1, 0, 0.95]},
            {
                "name": AXIS_INDICATOR,
                "type": "capsule",
                "size": [0.005, AXIS_INDICATOR_LENGTH / 2],
                "rgba": [0, 1, 1, 0.6],
            },
        ],
    )
    env.reset()
    return env


def show_estimates(env, drawer_position, kinematic_joint=None):
    """Move the render indicators to the estimated drawer position and, if given, along the estimated joint axis
    (kinematic_joint = [azimuth, elevation, joint value, initial point])."""
    env.set_indicator_pos(DRAWER_INDICATOR, np.asarray(drawer_position)[:3])
    if kinematic_joint is not None:
        azimuth, elevation, state = kinematic_joint[:3]
        axis = np.array([np.cos(azimuth) * np.sin(elevation), np.sin(azimuth) * np.sin(elevation), np.cos(elevation)])
        tip = np.asarray(kinematic_joint[3:6]) + axis * state
        env.set_indicator_pos(AXIS_INDICATOR, tip + axis * AXIS_INDICATOR_LENGTH / 2)
        body = env.env.sim.model.body_name2id(AXIS_INDICATOR + "_body")
        env.env.sim.model.body_quat[body] = _quaternion_z_to(axis)
    env.env.sim.forward()


def _quaternion_z_to(direction):
    """wxyz quaternion that rotates +z onto `direction`."""
    z = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1.0)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q
