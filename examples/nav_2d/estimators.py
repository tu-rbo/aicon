"""
Estimators for the 2D Navigation Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"


from typing import Dict, Tuple, Callable, Union

import torch

from aicon_base.components import EstimationComponent
from aicon_base.inference.ekf import predict_ekf, update_ekf_triple_connection

intial_target_location = torch.tensor([19.0, 11.0])
intial_target_uncertainty = torch.eye(2) * 20.0


def set_initial_target(mu, sigma):
    global intial_target_location, intial_target_uncertainty
    intial_target_location = mu
    intial_target_uncertainty = sigma


class AgentEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "agent_pos": (2,),
    }

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        goals: Union[None, Dict[str, Callable]] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
        no_differentiation: bool = False,
        prevent_loops_in_differentiation: bool = True,
        max_length_differentiation_trace: Union[int, None] = None,
        no_backwards_combinations: bool = False,
        nullspace_escape_trigger_length=0.5,
        nullspace_escape_trigger_similarity=-0.5,
        nullspace_escape_end_similarity=0.8,
        nullspace_escape_end_length=0.1,
        nullspace_escape_phase_out_ticks=10,
    ):
        super().__init__(
            name=name,
            connections=connections,
            goals=goals,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
            no_differentiation=no_differentiation,
            prevent_loops_in_differentiation=prevent_loops_in_differentiation,
            max_length_differentiation_trace=max_length_differentiation_trace,
            no_backwards_combinations=no_backwards_combinations,
        )

        self.nullspace_escape_trigger_length = nullspace_escape_trigger_length
        self.nullspace_escape_trigger_similarity = nullspace_escape_trigger_similarity
        self.nullspace_escape_end_similarity = nullspace_escape_end_similarity
        self.nullspace_escape_end_length = nullspace_escape_end_length
        self.nullspace_escape_phase_out_ticks = nullspace_escape_phase_out_ticks

    def initialize_quantities(self):
        c = self.connections["NavMeasurement"]
        with c.lock:
            if c.connected_quantities_initialized["agent_meas"]:
                agent_meas = c.connected_quantities["agent_meas"]
            else:
                return False
        self.quantities["agent_pos"] = self.quantities["agent_pos"] + c.c_func(self.quantities["agent_pos"], agent_meas)
        return True

    def define_estimation_function_f(self):
        c_func_act = self.connections["NavAction"].c_func
        c_func_obs = self.connections["NavMeasurement"].c_func

        self.differentiate_against = ("action",)

        def f_func(agent_pos, action, agent_meas):
            agent_pos_pred = c_func_act(agent_pos, action)
            agent_pos = agent_pos_pred + c_func_obs(agent_pos_pred.detach(), agent_meas)
            return (agent_pos,), (agent_pos,)

        return f_func, ["agent_pos"], ["agent_pos"]


class TargetEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {
        "target_pos": (2,),
        "target_uncertainty": (2, 2),
        "target_dist": (1,),
    }

    def initialize_quantities(self):
        c = self.connections["NavTargetMeasurement"]
        with c.lock:
            if not (
                c.connected_quantities_initialized["target_meas"] and c.connected_quantities_initialized["agent_pos"]
            ):
                return False
        self.quantities["target_pos"] = intial_target_location.to(dtype=self.dtype, device=self.device)
        self.quantities["target_uncertainty"] = intial_target_uncertainty.to(dtype=self.dtype, device=self.device)
        self.quantities["target_dist"] = torch.zeros((1,), device=self.device, dtype=self.dtype)
        return True

    def define_estimation_function_f(self):
        # argument order changed so that the target position comes first
        def c_func_obs(target_pos, agent_pos, target_meas):
            return self.connections["NavTargetMeasurement"].c_func(agent_pos, target_meas, target_pos)

        c_func_dist = self.connections["TargetDistConnection"].c_func
        self.differentiate_against = ("agent_pos",)

        Q = torch.eye(2) * 0.1
        R_meas = torch.eye(2) * 0.3
        R_agent = torch.eye(2) * 0.05

        def f_func(target_pos, target_uncertainty, target_meas, agent_pos, target_dist):
            # ekf prediction assuming no motion of the target
            target_pos_pred, target_uncertainty_pred = predict_ekf(
                lambda x: torch.clone(x), target_pos, target_uncertainty, Q
            )
            # ekf update for target position via constraint on measurement + agent position
            target_pos, target_uncertainty = update_ekf_triple_connection(
                c_func_obs,
                target_pos_pred,
                target_uncertainty_pred,
                agent_pos,
                R_agent,
                target_meas,
                R_meas,
                torch.zeros((2, 2)),
                prevent_uncertainty_state_grad=False,
            )
            # detach to prevent observation hacking (i.e. making the target look closer instead of being closer to it)
            target_dist = target_dist + c_func_dist(agent_pos, target_pos.detach(), target_dist)
            return (target_pos, target_uncertainty, target_dist), (target_pos, target_uncertainty, target_dist)

        return (
            f_func,
            ["target_pos", "target_uncertainty", "target_dist"],
            ["target_pos", "target_uncertainty", "target_dist"],
        )


class DistToObsEstimator(EstimationComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"current_dist2obs": (1,)}

    def initialize_quantities(self):
        c = self.connections["ObstacleDistConnection"]
        with c.lock:
            if not (
                c.connected_quantities_initialized["agent_pos"] and c.connected_quantities_initialized["obstacle_image"]
            ):
                return False
            agent_pos = c.connected_quantities["agent_pos"]
            obstacle_image = c.connected_quantities["obstacle_image"]

        self.quantities["current_dist2obs"] = c.c_func(
            obstacle_image, agent_pos, torch.zeros(1, device=self.device, dtype=self.dtype)
        )
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["ObstacleDistConnection"].c_func
        self.differentiate_against = ("agent_pos",)

        def f_func(current_dist2obs, obstacle_image, agent_pos):
            current_dist2obs = current_dist2obs + c_func(obstacle_image, agent_pos, current_dist2obs)
            return (current_dist2obs,), (current_dist2obs,)

        return f_func, ["current_dist2obs"], ["current_dist2obs"]
