"""
Goals for Real-Robot Object Pushing

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.components import Goal
from aicon_base.inference.util import gradient_preserving_clipping
from aicon_base.math.angles import shortest_angular_distance
from . import params
from .params import UNCERTAINTY_SCALE, real_goal_pose

ANGLE_TAPER_KNEE: float = 1.047198  # 60 deg
ANTIPODE_RELEASE: float = 2.617994  # 150 deg
ANTIPODE_DEBOUNCE: int = 8


class Push2GoalReal(Goal):

    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        self._latch_sign: float = 0.0
        self._latch_below: int = 0
        super().__init__(name="Push2Goal", is_active=is_active, mockbuild=False, fulfilled_value=-torch.inf)

    def define_goal_cost_function(self):
        goal_pose = torch.tensor(real_goal_pose(), dtype=self.dtype, device=self.device)
        print(
            f"[Push2GoalReal] REAL_GOAL_ID = {params.REAL_GOAL_ID}, goal_pose = "
            f"x {goal_pose[0].item():.3f} m, y {goal_pose[1].item():.3f} m, "
            f"theta {torch.rad2deg(goal_pose[2]).item():.1f} deg"
        )

        # goal tolerances; max_dist is also the midpoint of the phi schedule
        max_dist = 0.05
        max_angle = 0.349066  # 20 deg
        # phi far from / at the goal [deg]: rotation share tan(phi) / (1 + tan(phi)) of about 8 % / 88 %
        phi_far = 5.0
        phi_near = 82.0
        transition_decades = 0.3  # width of the transition, in decades of distance around max_dist

        def goal_func(object_position, object_uncertainty):
            dist = torch.norm(goal_pose[:2] - object_position[:2], keepdim=True)
            angle_signed = shortest_angular_distance(object_position[2:3], goal_pose[2:3])
            angle = torch.abs(angle_signed)

            a_now = angle.item()
            self._latch_below = self._latch_below + 1 if a_now < ANTIPODE_RELEASE else 0
            if self._latch_sign == 0.0 or self._latch_below > ANTIPODE_DEBOUNCE:
                self._latch_sign = 1.0 if angle_signed.item() >= 0.0 else -1.0
            angle_term = self._latch_sign * angle_signed
            angle_term = angle_term + (angle - angle_term).detach()

            decades = torch.log10(max_dist / dist.detach().clamp_min(1e-4))
            phi = torch.deg2rad(
                0.5 * (phi_near + phi_far) + 0.5 * (phi_near - phi_far) * torch.tanh(decades / transition_decades)
            )
            phi_far_rad = torch.deg2rad(torch.as_tensor(phi_far, dtype=phi.dtype, device=phi.device))
            taper = torch.tanh(angle.detach() / ANGLE_TAPER_KNEE)
            phi = phi_far_rad + (phi - phi_far_rad) * taper

            expected_cost = (
                100000.0 * (torch.cos(phi) * dist + torch.sin(phi) * angle_term)
                + 0.1 * UNCERTAINTY_SCALE * object_uncertainty
            )

            if dist.item() < max_dist and angle.item() < max_angle:
                print("GOAL REACHED")
            print(
                "\nX error:\t",
                (goal_pose[0] - object_position[0]).item(),
                "\nY error:\t",
                (goal_pose[1] - object_position[1]).item(),
                "\nCombined:\t",
                dist.item(),
                "\nAngle error:\t",
                (shortest_angular_distance(object_position[2], goal_pose[2]) / torch.pi * 180).item(),
                "\nAngle weight:\t",
                torch.rad2deg(phi).item(),
                "\nLatch:\t\t",
                self._latch_sign,
                "held" if self._latch_sign != (1.0 if angle_signed.item() >= 0.0 else -1.0) else "agrees",
                "\nUncertainty:\t",
                object_uncertainty.item(),
            )
            return (expected_cost,), (expected_cost,)

        return goal_func



HEIGHT_COEF = 4600000.0
HEIGHT_CLIP = 0.05


class GoalHeight(Goal):
    """Keep the pusher at the force-modulated height setpoint (see HeightDifference)."""

    def __init__(
        self,
        is_active: bool,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(name="GoalHeight", is_active=is_active, mockbuild=False, fulfilled_value=-torch.inf)

    def define_goal_cost_function(self):
        def goal_func(height_difference):
            err = gradient_preserving_clipping(height_difference, -HEIGHT_CLIP, HEIGHT_CLIP)
            cost = HEIGHT_COEF * err**2
            return (cost,), (cost,)

        return goal_func
