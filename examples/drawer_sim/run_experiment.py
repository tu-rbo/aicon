"""
Simulated Drawer Opening with AICON

    python -m examples.drawer_sim.run_experiment                          # the tutorial graph (TUTORIAL.md)
    python -m examples.drawer_sim.run_experiment --variant extended       # wrist camera, drawer seen as a bearing
    python -m examples.drawer_sim.run_experiment --variant extended --no-triangulation

The robot has to find the drawer handle, grasp it and pull the drawer open; the only goal is on the estimated
joint value of the drawer. `--render` opens the MuJoCo viewer (green: estimated handle position, cyan: estimated
joint axis; on macOS, the script restarts itself under MuJoCo's `mjpython`, which the viewer needs there). A plot of
the run is written to outputs/drawer_sim/.

Authors: Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers
License: MIT
"""

__author__ = "Adrian Pfisterer, Patrick Lowin, James Reynaldo, Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import os
import random
import sys

import numpy as np
import torch

from aicon_base.experiment_util.mjpython import relaunch_under_mjpython
from aicon_base.middleware.python_sequential import build_components, run_component_sequence
from .env import make_env, show_estimates, SUCCESS_JOINT_VALUE


def run(
    variant="tutorial",
    triangulation=True,
    seconds=30.0,
    noise_scale=1.0,
    depth_prior_noise=0.01,
    seed=0,
    render=False,
    stop_on_success=False,
    verbose=True,
):
    """Run one episode. Returns a dict of per-tick records and the step at which the drawer was first open."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if variant == "tutorial":
        from .tutorial import experiment_specifications as spec

        env = make_env(spec.INITIAL_QPOS, controller="OSC_POSITION", render=render)
        builders, _, _ = spec.get_building_functions(env)
    elif variant == "extended":
        from .extended import experiment_specifications as spec

        env = make_env(spec.INITIAL_QPOS, controller="OSC_POSE", raised_cabinet=True, render=render)
        builders, _, _ = spec.get_building_functions(
            env, triangulation=triangulation, noise_scale=noise_scale, depth_prior_noise=depth_prior_noise
        )
    else:
        raise ValueError("unknown variant " + str(variant))
    components = build_components(builders)
    velocity_action, gripper_action = components["EEVelocities"], components["GripperAction"]

    record = {
        k: []
        for k in (
            "t",
            "joint_true",
            "joint_estimated",
            "gripper",
            "grasp_likelihood",
            "ee",
            "handle_true",
            "handle_estimated",
        )
    }
    first_open = None
    t = 0.0
    for step in range(int(round(seconds / env.control_timestep))):
        run_component_sequence(components, torch.tensor(t, dtype=torch.float64))
        velocity = velocity_action.quantities["action_velo_ee"].cpu().numpy()
        gripper = float(gripper_action.quantities["gripper_activation"].squeeze())
        if variant == "tutorial":
            action = np.concatenate([velocity, [2 * gripper - 1]])
        else:
            action = np.concatenate([velocity, np.zeros(3), [2 * gripper - 1]])
        _, _, done, _ = env.step(action)

        joint = components["KinematicJointEstimator"].quantities["kinematic_joint"].detach().cpu().numpy().copy()
        handle = components["DrawerPosEstimator"].quantities["position_drawer"].detach().cpu().numpy().copy()
        record["t"].append(t)
        record["joint_true"].append(env.env.drawer_joint_value())
        record["joint_estimated"].append(float(joint[2]))
        record["gripper"].append(gripper)
        record["grasp_likelihood"].append(
            float(components["GraspLikelihoodEstimator"].quantities["likelihood_grasped_drawer"].reshape(-1)[0])
        )
        record["ee"].append(env.env.get_ee_pos())
        record["handle_true"].append(env.env.get_drawer_handle_pos())
        record["handle_estimated"].append(handle)
        if render:
            show_estimates(env, handle, joint)
            env.render()
        if done and first_open is None:
            first_open = step
            if verbose:
                print(f"Drawer open after {t:.1f} s")
            if stop_on_success:
                break
        t += env.control_timestep
    env.close()
    return record, first_open


def plot_record(record, title, path):
    import matplotlib.pyplot as plt

    t = np.array(record["t"])
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, record["joint_true"], label="true")
    axes[0].plot(t, record["joint_estimated"], "--", label="estimated")
    axes[0].axhline(SUCCESS_JOINT_VALUE, color="gray", lw=0.8, label="open")
    axes[0].set_ylabel("drawer joint [m]")
    axes[0].legend(loc="lower left", fontsize=8)
    error = np.linalg.norm(np.array(record["handle_estimated"]) - np.array(record["handle_true"]), axis=1)
    axes[1].plot(t, error)
    axes[1].set_ylabel("handle error [m]")
    axes[2].plot(t, record["grasp_likelihood"], label="grasp likelihood")
    axes[2].plot(t, record["gripper"], label="gripper closed")
    axes[2].set_ylabel("likelihood")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["tutorial", "extended"], default="tutorial")
    parser.add_argument(
        "--no-triangulation",
        action="store_true",
        help="extended variant: measure the handle position directly instead of its bearing",
    )
    parser.add_argument("--seconds", type=float, default=30.0, help="simulated time")
    parser.add_argument("--noise", type=float, default=1.0, help="extended variant: scale of the sensor noise")
    parser.add_argument(
        "--depth-prior-noise",
        type=float,
        default=0.01,
        help="extended variant with triangulation: noise [m] of the initial depth guess",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "drawer_sim"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("--render needs a display (DISPLAY is not set)")
    if args.render:
        relaunch_under_mjpython()

    torch.set_default_dtype(torch.float64)
    record, first_open = run(
        args.variant,
        triangulation=not args.no_triangulation,
        seconds=args.seconds,
        noise_scale=args.noise,
        depth_prior_noise=args.depth_prior_noise,
        seed=args.seed,
        render=args.render,
    )
    name = args.variant + (
        "" if args.variant == "tutorial" else ("_direct" if args.no_triangulation else "_triangulation")
    )
    if first_open is None:
        print(f"{name}: drawer not opened within {args.seconds:.0f} s (joint value {record['joint_true'][-1]:.3f} m)")
    else:
        print(f"{name}: drawer opened after {record['t'][first_open]:.1f} s")
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, name + ".png")
        plot_record(record, name, path)
        print("Plot written to", path)


if __name__ == "__main__":
    main()
