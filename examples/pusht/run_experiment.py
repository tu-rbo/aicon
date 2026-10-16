"""
PushT with AICON: run episodes from random initial states

Usage (from the repository root):
    python -m examples.pusht.run_experiment --episodes 3

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import os
import pickle
import random
from datetime import datetime

import matplotlib
import numpy as np
import torch
from matplotlib.patches import Polygon, Circle
from tqdm import tqdm

from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from aicon_base.visualization.network_graphs import (
    draw_basic_graph,
    get_fancy_network_graph,
    draw_current_active_gradients,
)
from .env import env_instance
from .experiment_specifications import get_building_functions
from .params import sim_goal_offset_viz, sim_goal_offset_goal, SUCCESS_COVERAGE

T_KEYPOINTS_NEUTRAL = [
    [-60.0, -45.0],
    [60.0, -45.0],
    [60.0, -15.0],
    [15.0, -15.0],
    [15.0, 75.0],
    [-15.0, 75.0],
    [-15.0, -15.0],
    [-60.0, -15.0],
]


def sample_initial_states(n):
    """Random T poses inside the workspace, pushers starting with one or both coordinates outside it."""
    angles = (2 * torch.pi) * torch.rand(n) - torch.pi
    obj_x = 100 + (400 - 100) * torch.rand(n)
    obj_y = 100 + (400 - 100) * torch.rand(n)

    def sample_outside():
        while True:
            candidate = 50 + (650 - 50) * torch.rand(1)
            if candidate < 100 or candidate > 400:
                return candidate

    pusher_start = torch.zeros(n, 2)
    for i in range(n):
        if torch.randint(0, 2, (1,)).item() == 0:
            inside_idx = torch.randint(0, 2, (1,)).item()
            pusher_start[i, inside_idx] = 100 + (400 - 100) * torch.rand(1)
            pusher_start[i, 1 - inside_idx] = sample_outside()
        else:
            for j in range(2):
                pusher_start[i, j] = sample_outside()
    return [
        [float(pusher_start[i, 0]), float(pusher_start[i, 1]), float(obj_x[i]), float(obj_y[i]), float(angles[i])]
        for i in range(n)
    ]


def get_graph_layout():
    component_builders, connection_builders, _ = get_building_functions()
    return get_fancy_network_graph(
        component_builders,
        connection_builders,
        name_remapping={
            "contact_likelihoods": "Contact Likelihoods",
            "interaction_likelihood": "Interaction Likelihood",
            "push_velocity_directions": " Possible\nContact Locations",
            "Push2Goal": "Reach Object\nGoal Position",
            "KeyPoints2Position": "Static Transformation",
            "keypoint_locations": "Object Keypoints\nMeasurement",
        },
        simple_combinations={
            "Contacts2ObjectPosition": "Contact-Based\nDynamics Model",
            "ObjectvsPush2InteractionlikelihoodSim": "Contact-Based\nDynamics Model",
            "pusher_point_action": "Proprioception\nand Actuation",
            "pusher_point_meas": "Proprioception\nand Actuation",
            "PusherPointMeasurement": "Linear\nDynamics Model",
            "PusherPointAction": "Linear\nDynamics Model",
            "KeyPoints2ContactPoints": "Object Geometry vs.\nPusher Location",
            "ContactPoints2ContactLikelihood": "Object Geometry vs.\nPusher Location",
            "ContactPoints2PushVelocityDirections": "Object Geometry vs.\nPusher Location",
        },
        recursive_estimators={
            "object_position": "Object Pose",
            "pusher_point": "Pusher 2D Location",
            "push_velocity_directions": "          Contacts          ",
            "contact_likelihoods": "          Contacts          ",
            "interaction_likelihood": "          Contacts          ",
        },
        type_combinations={"Action": "Real-World Sensors and Actuators", "Sensor": "Real-World Sensors and Actuators"},
        cut_edges=[],
        cut_nodes=["contactpoint_locations", "object_COM"],
        summarized_recursive_estimators={},
        node_name_to_layer=None,
    )


def run_episode(
    state, folder, max_ticks=30000, plot_every=50, success_coverage=SUCCESS_COVERAGE, show=False, progress=False
):
    """Run one episode until the T covers `success_coverage` of the goal region or `max_ticks` pass; returns the
    coverage history. Frames go to `folder` if plot_every > 0 and are also displayed with `show`; `progress` shows a
    progress bar."""
    state = list(state)
    for i in range(3):
        state[2 + i] += sim_goal_offset_goal[i]
    env_instance.reset(state=np.array(state, dtype=np.float64))
    component_builders, _, _ = get_building_functions()
    components = build_components(component_builders)

    if plot_every > 0:
        import matplotlib.pyplot as plt

        G, pos, _, remappings = get_graph_layout()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19, 11))
        if show:
            plt.ion()
            fig.show()

    pusher_history, keypoint_history, conflict_history, coverage = [], [], [], []
    ticks = tqdm(range(max_ticks), desc="ticks", unit="tick", leave=False, disable=not progress)
    for i in ticks:
        components = run_component_sequence(components, torch.tensor(i * 0.1))
        if "coverage" not in env_instance.info:
            continue
        pusher_history.append(components["PusherEstimator"].quantities["pusher_point"].unsqueeze(0))
        keypoint_locations = components["KeypointSensor"].quantities["keypoint_locations"]
        keypoint_history.append(keypoint_locations)
        grads, mix_ins = components["PusherAction"].used_gradients
        # A projected (conflict-resolving) gradient carries mix-in traces.
        conflict_history.append(len("".join(x for xs in mix_ins for xss in xs for x in xss)) > 0)
        coverage.append(env_instance.info["coverage"])
        if i % 50 == 0:
            ticks.set_postfix(coverage=f"{coverage[-1]:.3f}")

        if plot_every > 0 and i % plot_every == 0 and i > 30:
            axs[0].clear()
            axs[1].clear()
            plot_trajectory(axs[1], pusher_history, keypoint_locations, conflict_history, keypoint_history)
            draw_basic_graph(G, pos, axs[0], remappings=remappings, legend=False)
            draw_current_active_gradients(grads, G, pos, axs[0], remappings=remappings)
            axs[0].axis("off")
            axs[0].set_facecolor("white")
            axs[0].set_xlim(-1.1, 1.1)
            axs[1].set_title("Object Coverage: " + str(env_instance.info["coverage"]))
            fig.savefig(os.path.join(folder, str(i).zfill(5) + ".png"))
            if show:
                plt.pause(0.001)
        if env_instance.info["coverage"] >= success_coverage:
            ticks.close()
            print(f"Success after {i} ticks, coverage {env_instance.info['coverage']:.3f}")
            break
    else:
        print(f"No success within {max_ticks} ticks")
    if plot_every > 0:
        plt.close(fig)
    with open(os.path.join(folder, "coverage.pickle"), "wb") as f:
        pickle.dump(coverage, f)
    return coverage


def plot_trajectory(ax, pusher_history, keypoint_locations, conflict_history, keypoint_history):
    """Object, goal and smoothed pusher path; green where the pusher followed a single goal, red during conflicts."""
    dtype, device = keypoint_locations.dtype, keypoint_locations.device
    goal_pose = torch.tensor(env_instance.info["goal_pose"], dtype=dtype, device=device) + torch.tensor(
        sim_goal_offset_viz
    )
    keypoints_neutral = torch.tensor(T_KEYPOINTS_NEUTRAL, dtype=dtype, device=device)
    s, c = torch.sin(goal_pose[2:3]), torch.cos(goal_pose[2:3])
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).squeeze()
    keypoints_goal = torch.einsum("ij,bj->bi", rot, keypoints_neutral) + goal_pose[:2].unsqueeze(0)
    ax.add_patch(Polygon(keypoint_locations, facecolor="#8FA3B8"))
    ax.add_patch(Polygon(keypoints_goal, facecolor="#90EE90", zorder=-10))

    pusher = torch.cat(pusher_history).cpu().numpy()
    conflicts = np.array(conflict_history).astype(float)
    ax.add_patch(Circle((pusher[-1, 0], pusher[-1, 1]), radius=15, color="#4E7EFF", fill=True, zorder=10))

    kernel_size = 29
    shadow_location_steps = 60
    kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
    pusher[:, 0] = np.convolve(pusher[:, 0], kernel, mode="same")
    pusher[:, 1] = np.convolve(pusher[:, 1], kernel, mode="same")
    conflicts = np.convolve(conflicts, kernel, mode="same")
    pusher = pusher[kernel_size:-kernel_size]
    conflicts = np.round(conflicts[kernel_size:-kernel_size]).astype(bool)

    def plot_segment(start, end, in_conflict):
        mask = conflicts[start:end] if in_conflict else np.logical_not(conflicts[start:end])
        ax.plot(pusher[start:end, 0][mask], pusher[start:end, 1][mask], color="red" if in_conflict else "green")

    last_switch = 0
    for j in range(1, len(pusher)):
        if j % shadow_location_steps == 0 and not torch.all(
            keypoint_history[j] == keypoint_history[j - shadow_location_steps]
        ):
            ax.add_patch(Polygon(keypoint_history[j], facecolor="#8FA3B8", alpha=0.2, zorder=-5))
        if conflicts[j] != conflicts[j - 1]:
            plot_segment(last_switch, j - 1, in_conflict=conflicts[j - 1])
            last_switch = j
        if j + 1 == len(pusher) and j - last_switch > 2:
            plot_segment(last_switch, None, in_conflict=conflicts[j])
    ax.invert_yaxis()
    ax.axis("equal")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to run")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-ticks", type=int, default=30000)
    parser.add_argument(
        "--success-coverage",
        type=float,
        default=SUCCESS_COVERAGE,
        help="goal coverage at which an episode counts as solved",
    )
    parser.add_argument("--plot-every", type=int, default=50, help="save a frame every N ticks (0 disables)")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "pusht"))
    parser.add_argument("--show", action="store_true", help="also display the frames while they are saved")
    parser.add_argument("--detect-anomaly", action="store_true", help="enable torch autograd anomaly detection")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        print("--show: no interactive matplotlib backend available (install e.g. python3-tk); frames are only saved")
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    states = sample_initial_states(max(args.episodes, 100))[: args.episodes]
    for state in states:
        folder = os.path.join(
            args.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + "_".join(f"{v:.1f}" for v in state)
        )
        os.makedirs(folder, exist_ok=True)
        print("Episode from state (pusher x, pusher y, T x, T y, T angle):", [round(v, 2) for v in state])
        run_episode(
            state,
            folder,
            max_ticks=args.max_ticks,
            plot_every=args.plot_every,
            success_coverage=args.success_coverage,
            show=args.show,
            progress=True,
        )


if __name__ == "__main__":
    main()
