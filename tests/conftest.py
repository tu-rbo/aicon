import random

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def deterministic_double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(111)
    random.seed(111)
    np.random.seed(111)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="session")
def ros_context():
    """One ROS 2 context for all tests that need one: some real-robot components keep module-level ROS nodes, which a
    later context could not use."""
    rclpy = pytest.importorskip("rclpy")
    rclpy.init()
    yield
    rclpy.shutdown()
