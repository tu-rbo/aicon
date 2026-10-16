"""
2D Navigation with AICON: reach a target known only through bearing measurements while avoiding obstacles

Usage (from the repository root):
    python -m examples.nav_2d.run_experiment

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import os
import random

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from aicon_base.visualization.util import get_mpl_covariance_ellipse
from . import actions_and_sensors
from .actions_and_sensors import (
    NavAgentSensor,
    NavTargetSensor,
    ObstacleImageSensor,
    set_initial_locations,
    NavActionComp,
)
from .connections import build_connections
from .constants import obs_img_size
from .estimators import AgentEstimator, TargetEstimator, DistToObsEstimator, set_initial_target
from .goals import ObstacleGoal, TargetGoal
from .scenarios import make_obs_torch, START, TARGET, TARGET_PRIOR

INITIAL_TARGET_UNCERTAINTY = 15.0


def get_building_functions(
    debug_mode: bool = False,
    nullspace_escape_trigger_length=0.5,
    nullspace_escape_trigger_similarity=-0.7,
    nullspace_escape_end_similarity=-0.5,
    nullspace_escape_end_length=0.2,
    nullspace_escape_phase_out_ticks=5,
    goal_scale_factor=6.0,
):
    """Defaults are the best-performing parameter set of the parameter search."""
    connection_builders = build_connections()
    cpu = torch.device("cpu")
    component_builders = {
        "NavActionComp": lambda mockbuild: NavActionComp(
            "NavActionComp",
            connections={k: connection_builders[k] for k in ("NavAction",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "NavAgentSensor": lambda mockbuild: NavAgentSensor(
            "NavAgentSensor",
            connections={k: connection_builders[k] for k in ("NavMeasurement",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "NavTargetSensor": lambda mockbuild: NavTargetSensor(
            "NavTargetSensor",
            connections={k: connection_builders[k] for k in ("NavTargetMeasurement",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "ObstacleImageSensor": lambda mockbuild: ObstacleImageSensor(
            "ObstacleImageSensor",
            connections={k: connection_builders[k] for k in ("ObstacleDistConnection",)},
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "AgentEstimator": lambda mockbuild: AgentEstimator(
            "AgentEstimator",
            connections={
                k: connection_builders[k]
                for k in (
                    "NavAction",
                    "NavMeasurement",
                    "NavTargetMeasurement",
                    "ObstacleDistConnection",
                    "TargetDistConnection",
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
            nullspace_escape_trigger_length=nullspace_escape_trigger_length,
            nullspace_escape_trigger_similarity=nullspace_escape_trigger_similarity,
            nullspace_escape_end_similarity=nullspace_escape_end_similarity,
            nullspace_escape_end_length=nullspace_escape_end_length,
            nullspace_escape_phase_out_ticks=nullspace_escape_phase_out_ticks,
        ),
        "TargetEstimator": lambda mockbuild: TargetEstimator(
            "TargetEstimator",
            connections={k: connection_builders[k] for k in ("NavTargetMeasurement", "TargetDistConnection")},
            goals={
                "TargetGoal": lambda d, t, mockbuild=False: TargetGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
        "DistToObsEstimator": lambda mockbuild: DistToObsEstimator(
            "DistToObsEstimator",
            connections={k: connection_builders[k] for k in ("ObstacleDistConnection",)},
            goals={
                "ObstacleGoal": lambda d, t, mockbuild=False: ObstacleGoal(
                    is_active=True, dtype=t, device=d, mockbuild=mockbuild, scale=goal_scale_factor
                )
            },
            device=cpu,
            dtype=torch.double,
            mockbuild=mockbuild,
            debug_mode=debug_mode,
        ),
    }
    frame_rates = {k: 0.5 for k in component_builders}
    return component_builders, connection_builders, frame_rates


def run_experiment(obs_img, max_ticks=2500, progress_desc=None, **building_kwargs):
    """Returns (outcome, ticks, trajectory, components) with outcome one of "success" (the agent's estimate is within
    1 of the target), "collision" (the agent touched a wall of the obstacle map, where it stopped) or "stuck"; the
    trajectory is the agent's true path. With `progress_desc`, a progress bar with that description is shown."""
    component_builders, _, _ = get_building_functions(**building_kwargs)
    components = build_components(component_builders)
    components["ObstacleImageSensor"].set_obstacle_img(obs_img)
    actions_and_sensors.set_obstacle_map(obs_img)
    target_true = actions_and_sensors.location_target

    trajectory = [actions_and_sensors.location_agent.double()]
    for i in tqdm(range(max_ticks), desc=progress_desc, unit="tick", leave=False, disable=progress_desc is None):
        components = run_component_sequence(components, torch.tensor(i * 0.1))
        trajectory.append(actions_and_sensors.location_agent.double())
        if actions_and_sensors.collided:
            return "collision", i, torch.stack(trajectory), components
        est_pos = components["AgentEstimator"].quantities["agent_pos"]
        if torch.norm(target_true - est_pos) < 1.0:
            return "success", i, torch.stack(trajectory), components
    return "stuck", max_ticks, torch.stack(trajectory), components


def plot_run(path, obs_img, start, goal, trajectory, components, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(obs_img.T.numpy(), cmap="gray", origin="lower")
    traj = trajectory.numpy()
    ax.plot(traj[:, 0], traj[:, 1], linestyle="--", color="blue", label="Agent trajectory")
    ax.scatter(*start, color="blue", label="Start")
    ax.scatter(*goal, color="red", label="Target")
    est_target = components["TargetEstimator"].quantities["target_pos"]
    unc_target = components["TargetEstimator"].quantities["target_uncertainty"]
    ax.add_patch(
        get_mpl_covariance_ellipse(
            est_target,
            unc_target,
            scale=0.5,
            facecolor="none",
            edgecolor="orange",
            label="Belief over target",
            linewidth=2,
        )
    )
    ax.set_xlim(0, obs_img_size - 1)
    ax.set_ylim(0, obs_img_size - 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-ticks", type=int, default=2500)
    parser.add_argument("--goal-scale", type=float, default=6.0, help="weight of the obstacle avoidance goal")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "nav_2d"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.output_dir, exist_ok=True)

    obs_img = make_obs_torch()
    set_initial_locations(torch.tensor(START, dtype=torch.float), torch.tensor(TARGET, dtype=torch.float))
    set_initial_target(torch.tensor(TARGET_PRIOR), torch.eye(2) * INITIAL_TARGET_UNCERTAINTY)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    outcome, ticks, trajectory, components = run_experiment(
        obs_img, max_ticks=args.max_ticks, progress_desc="navigating", goal_scale_factor=args.goal_scale
    )
    print(f"{outcome} after {ticks} ticks")
    if not args.no_plot:
        plot_run(
            os.path.join(args.output_dir, "nav_2d.png"),
            obs_img,
            START,
            TARGET,
            trajectory,
            components,
            f"2D navigation: {outcome} ({ticks} ticks)",
        )


if __name__ == "__main__":
    main()
