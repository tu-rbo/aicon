"""The walls of the simulated 2D navigation world cannot be passed: the agent stops at the first wall it touches."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("scipy")

from examples.nav_2d import actions_and_sensors  # noqa: E402
from examples.nav_2d.scenarios import make_obs  # noqa: E402

# the map has a wall along y = 10 (cells (2..7, 10) and (12..24, 10)), whose squares cover y in [9.5, 10.5]


def test_wall_contact_on_the_path():
    actions_and_sensors.set_obstacle_map(make_obs())
    contact = actions_and_sensors.first_wall_contact((15.0, 9.0), (15.0, 11.0))
    assert contact == pytest.approx(0.25)  # reaches y = 9.5 a quarter of the way
    assert actions_and_sensors.first_wall_contact((15.0, 5.0), (15.0, 9.0)) is None


def test_no_jumping_over_a_wall():
    actions_and_sensors.set_obstacle_map(make_obs())
    # both ends are free, the wall lies in between
    contact = actions_and_sensors.first_wall_contact((15.0, 8.0), (15.0, 12.0))
    assert contact == pytest.approx(0.375)


def test_agent_stops_at_the_wall():
    actions_and_sensors.set_obstacle_map(make_obs())
    actions_and_sensors.set_initial_locations(torch.tensor([15.0, 8.0]), torch.tensor([24.0, 24.0]))
    pushing_up = SimpleNamespace(quantities={"action": torch.tensor([0.0, 0.2])})
    for _ in range(20):
        actions_and_sensors.NavActionComp.send_action_values(pushing_up)
        assert actions_and_sensors.location_agent[1] <= 9.5
    assert actions_and_sensors.collided
    assert actions_and_sensors.location_agent[1] == pytest.approx(9.5)
