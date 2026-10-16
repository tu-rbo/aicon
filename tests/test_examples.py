"""Short end-to-end runs of the examples that need neither ROS nor hardware."""

import pytest
import torch


@pytest.mark.parametrize(
    "init_setup, goal",
    [
        ("0towers", "StackAonBonC"),
        ("3towers", "StackAonB"),
        ("2towers", "UnstackAll"),
        # needs a move a second time, after undoing it to reach the block below
        ("2towersAlt", "StackAonBonC"),
    ],
)
def test_blocksworld_solves(init_setup, goal):
    from examples.blocksworld.run_experiment import run_experiment

    assert run_experiment(init_setup, goal, max_ticks=200) >= 0


def test_nav_2d_reaches_target():
    from examples.nav_2d.actions_and_sensors import set_initial_locations
    from examples.nav_2d.estimators import set_initial_target
    from examples.nav_2d.run_experiment import run_experiment, INITIAL_TARGET_UNCERTAINTY
    from examples.nav_2d.scenarios import make_obs_torch, START, TARGET, TARGET_PRIOR

    obs_img = make_obs_torch()
    set_initial_locations(torch.tensor(START, dtype=torch.float), torch.tensor(TARGET, dtype=torch.float))
    set_initial_target(torch.tensor(TARGET_PRIOR), torch.eye(2) * INITIAL_TARGET_UNCERTAINTY)
    outcome, ticks, trajectory, _ = run_experiment(obs_img, max_ticks=1000)
    assert outcome == "success", (outcome, ticks)


def test_pusht_runs(tmp_path):
    pytest.importorskip("gym_pusht")
    from examples.pusht.run_experiment import run_episode

    coverage = run_episode([220.0, 50.0, 220.0, 100.0, 0.0], str(tmp_path), max_ticks=150, plot_every=0)
    assert len(coverage) > 0
    assert (tmp_path / "coverage.pickle").exists()


@pytest.mark.parametrize("variant, triangulation", [("tutorial", True), ("extended", True), ("extended", False)])
def test_drawer_sim_opens_drawer(variant, triangulation):
    pytest.importorskip("robosuite")
    from examples.drawer_sim.run_experiment import run

    record, first_open = run(
        variant, triangulation=triangulation, seconds=15.0, seed=0, stop_on_success=True, verbose=False
    )
    assert first_open is not None, record["joint_true"][-1]
