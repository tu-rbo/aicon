"""
Connections for the Blocks World Domain

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

from typing import Union

import torch

from aicon_base.connections import ActiveInterconnection
from .global_params import NUM_BLOCKS
from aicon_base.inference.util import gradient_preserving_clipping


def build_connections():
    connections = {
        "BelowLikelihood": lambda device, dtype, mockbuild: BelowLikelihood(
            "BelowLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "BelowLikelihoodDummySensing": lambda device, dtype, mockbuild: BelowLikelihoodDummySensing(
            "BelowLikelihoodDummySensing", device=device, dtype=dtype, mockbuild=mockbuild
        ),
        "ClearLikelihood": lambda device, dtype, mockbuild: ClearLikelihood(
            "ClearLikelihood", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
    return connections


class BelowLikelihood(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "likelihood_below": (NUM_BLOCKS, NUM_BLOCKS),
                "action_blocks": (2, NUM_BLOCKS, NUM_BLOCKS),
                "likelihood_clear": (NUM_BLOCKS,),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            required_signs_dict={"likelihood_below": {"likelihood_clear": 1}},
        )

    def define_implicit_connection_function(self):

        def connection_func(likelihood_below, action_blocks, likelihood_clear):
            likelihood_below_hard = (likelihood_below > 0.5) * 1.0
            likelihood_puttable = torch.einsum("i,j->ij", likelihood_clear, likelihood_clear) * (
                1 - likelihood_below_hard
            )  # can only do those that are not already put

            others_above = torch.clip(
                torch.abs(1 - torch.sum(likelihood_below_hard, dim=0).unsqueeze(0).expand(*likelihood_puttable.shape)),
                0.01,
                0.99,
            )
            likelihood_removable = (
                likelihood_clear.unsqueeze(1).expand(*likelihood_puttable.shape)  # top block needs to be free
                * (1 - others_above)  # bottom block should not have others above it
                * likelihood_below_hard
            )  # they should actually be on top of each other
            # remove action grad for non put/removable actions
            action_put = action_blocks[0]
            action_remove = action_blocks[1]
            innovation = likelihood_below - likelihood_puttable * action_put + likelihood_removable * action_remove
            innovation = (
                innovation - torch.eye(NUM_BLOCKS, dtype=self.dtype, device=self.device) * innovation
            )  # no diagonal entries
            return innovation

        return connection_func


class BelowLikelihoodDummySensing(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {
                "likelihood_below": (
                    NUM_BLOCKS,
                    NUM_BLOCKS,
                ),
                "below_state_sensed": (
                    NUM_BLOCKS,
                    NUM_BLOCKS,
                ),
                "action_blocks": (2, NUM_BLOCKS, NUM_BLOCKS),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):

        def connection_func(likelihood_below, below_state_sensed):
            innovation = below_state_sensed - likelihood_below
            return innovation

        return connection_func


class ClearLikelihood(ActiveInterconnection):
    def __init__(
        self,
        name: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
    ):
        super().__init__(
            name,
            {"likelihood_clear": (NUM_BLOCKS,), "likelihood_below": (NUM_BLOCKS, NUM_BLOCKS)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):

        def connection_func(likelihood_clear, likelihood_below):
            likelihood_below = likelihood_below * (likelihood_below > 0.5).detach()
            new_likelihood = 1.0 - gradient_preserving_clipping(torch.sum(likelihood_below, dim=0), 0.001, 0.999)
            return new_likelihood - likelihood_clear

        return connection_func
