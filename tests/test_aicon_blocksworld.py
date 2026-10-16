"""A random blocks-world setup and goal from the defaults, with a random seed, is solved."""
# pylint: disable=missing-class-docstring, missing-function-docstring

import random
import unittest

import numpy as np
import torch

from examples.blocksworld.run_experiment import DEFAULT_GOALS, DEFAULT_SETUPS, run_experiment


class Test(unittest.TestCase):
    def test_aicon_blocksworld(self):
        """A random setup and goal from the defaults, with a random seed, is solved."""
        seed = random.randint(0, 1000)
        init_setup = random.choice(DEFAULT_SETUPS)
        goal = random.choice(DEFAULT_GOALS)

        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True)
        torch.random.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        ticks = run_experiment(init_setup, goal)
        self.assertGreaterEqual(ticks, 0, f"{init_setup} / {goal} not solved (seed {seed})")


if __name__ == "__main__":
    unittest.main()
