# -*- coding: utf-8 -*-
"""
Sensor Components in AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import traceback
from abc import abstractmethod
from typing import Dict, Callable, Union

import torch
from aicon_base.components.base_components import Component


class SensorComponent(Component):
    """A component whose quantities are measurements. Implement `obtain_measurements`, which writes
    `self.quantities` (and `self.timestamp`) and returns whether a measurement was obtained."""

    def __init__(
        self,
        name: str,
        connections: Dict[str, Callable],
        *,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        debug_mode: bool = False,
    ):

        super().__init__(
            name, connections, goals=None, dtype=dtype, device=device, mockbuild=mockbuild, debug_mode=debug_mode
        )

    @abstractmethod
    def obtain_measurements(self) -> bool:
        return False

    def initialize_quantities(self) -> bool:
        return self.obtain_measurements()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.obtain_measurements()
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()
