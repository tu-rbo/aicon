"""
Shared gym-pusht Environment Instance

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)  # pylint: disable=unused-import
import gymnasium as gym


class Environment:
    def __init__(self):
        self.env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="environment_state_agent_pos")
        self.observation, self.info = self.env.reset()
        self.image = None

    def reset(self, state=None):
        options = {"reset_to_state": state} if state is not None else None
        self.observation, self.info = self.env.reset(options=options)

    def perform_action(self, action):
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        self.image = self.env.render()

    def close(self):
        self.env.close()


env_instance = Environment()
