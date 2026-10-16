"""
Locations of Pretrained Models Used by the Real-Robot Sensors

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import os


def _from_env(env_var: str, description: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        raise RuntimeError(f"{env_var} is not set. Point it to {description}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{env_var}={path} does not exist ({description}).")
    return path


def tapir_checkpoint() -> str:
    """Causal BootsTAPIR checkpoint (causal_bootstapir_checkpoint.pt) from https://github.com/google-deepmind/tapnet"""
    return _from_env("AICON_TAPIR_CHECKPOINT", "the causal BootsTAPIR checkpoint 'causal_bootstapir_checkpoint.pt'")


def os2d_root() -> str:
    """Checkout of https://github.com/aosokin/os2d including its models/ and data/ folders."""
    return _from_env("AICON_OS2D_ROOT", "a checkout of the os2d repository with its downloaded models")
