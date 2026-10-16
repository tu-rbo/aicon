"""
Parameters for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

NUM_KEYPOINTS: int = 8


SUCCESS_COVERAGE: float = 0.9 # 0.95 (as gym) also works but sometimes takes considerably more steps

# gym-pusht's block pose and the estimator's object pose use different reference points on the T.
# `sim_goal_offset_goal` is added to a sampled start pose before it is passed to the environment,
# `sim_goal_offset_viz` to the environment's goal pose when it is drawn.
sim_goal_offset_viz: (float, float, float) = (-32, 32, 0)
sim_goal_offset_goal: (float, float, float) = (-32, -12, 0)
