"""
Connections for the blocks world experiment.
This module defines the connections between different components in the blocks world environment,
including likelihood connections for block relationships and clear states.
"""

from typing import Union

import torch

from aicon.base_classes.connections import ActiveInterconnection
from aicon.blocksworld_experiment.global_params import NUM_BLOCKS
from aicon.inference.util import gradient_preserving_clipping


def build_connections():
    """
    Build and return all connections needed for the blocks world experiment.
    
    Returns:
        dict: Dictionary of connection builders
    """
    connections = {
        "BelowLikelihood": lambda device, dtype, mockbuild: BelowLikelihood("BelowLikelihood", device=device, dtype=dtype, mockbuild=mockbuild),
        "BelowLikelihoodDummySensing": lambda device, dtype, mockbuild: BelowLikelihoodDummySensing("BelowLikelihoodDummySensing", device=device, dtype=dtype, mockbuild=mockbuild),
        "ClearLikelihood": lambda device, dtype, mockbuild: ClearLikelihood("ClearLikelihood", device=device, dtype=dtype, mockbuild=mockbuild),
    }
    return connections


class BelowLikelihood(ActiveInterconnection):
    """
    Connection that updates the likelihood of one block being below another.
    This connection handles both putting blocks on top of others and removing blocks.
    """

    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the BelowLikelihood connection.
        
        Args:
            name: Name of the connection
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name, {"likelihood_below": (NUM_BLOCKS, NUM_BLOCKS),
                                "action_blocks": (2, NUM_BLOCKS, NUM_BLOCKS),
                                "likelihood_clear": (NUM_BLOCKS,)},
                         dtype=dtype, device=device, mockbuild=mockbuild,
                         required_signs_dict={"likelihood_below": {"likelihood_clear": 1}})

    def define_implicit_connection_function(self):
        """
        Define the connection function that updates below likelihoods.
        
        Returns:
            function: Connection function that takes likelihood_below, action_blocks, and likelihood_clear as input
        """
        def connection_func(likelihood_below, action_blocks, likelihood_clear):
            """
            Update the below likelihood matrix based on actions and clear states.
            
            Args:
                likelihood_below: Current below likelihood matrix
                action_blocks: Action matrix indicating block movements
                likelihood_clear: Clear state likelihood vector
            
            Returns:
                tensor: Innovation term for updating below likelihoods
            """
            likelihood_below_hard = (likelihood_below > 0.5) * 1.0
            likelihood_puttable = (torch.einsum("i,j->ij", likelihood_clear, likelihood_clear)
                                   * (1 - likelihood_below_hard))    # can only do those that are not already put

            others_above = torch.clip(torch.abs(1 - torch.sum(likelihood_below_hard, dim=0).unsqueeze(0).expand(*likelihood_puttable.shape)), 0.01, 0.99)
            likelihood_removable = (likelihood_clear.unsqueeze(1).expand(*likelihood_puttable.shape)   # top block needs to be free
                                    * (1 - others_above)        # bottom block should not have others above it
                                    * likelihood_below_hard)         # they should actually be on top of each other
            # remove action grad for non put/removable actions
            action_put = action_blocks[0]
            action_remove = action_blocks[1]
            innovation = likelihood_below - likelihood_puttable * action_put + likelihood_removable * action_remove
            innovation = innovation - torch.eye(NUM_BLOCKS, dtype=self.dtype, device=self.device) * innovation    # no diagonal entries
            return innovation

        return connection_func


class BelowLikelihoodDummySensing(ActiveInterconnection):
    """
    Connection that updates below likelihoods based on sensor data.
    This is a simplified connection for evaluation purposes.
    """

    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the BelowLikelihoodDummySensing connection.
        
        Args:
            name: Name of the connection
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name, {"likelihood_below": (NUM_BLOCKS, NUM_BLOCKS),
                                "below_state_sensed": (NUM_BLOCKS, NUM_BLOCKS),
                                "action_blocks": (2, NUM_BLOCKS, NUM_BLOCKS)},
                         dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        """
        Define the connection function that updates below likelihoods based on sensor data.
        
        Returns:
            function: Connection function that takes likelihood_below and below_state_sensed as input
        """
        def connection_func(likelihood_below, below_state_sensed):
            """
            Update the below likelihood matrix based on sensor data.
            
            Args:
                likelihood_below: Current below likelihood matrix
                below_state_sensed: Sensor data for below relationships
            
            Returns:
                tensor: Innovation term for updating below likelihoods
            """
            innovation = below_state_sensed - likelihood_below
            return innovation

        return connection_func


class ClearLikelihood(ActiveInterconnection):
    """
    Connection that updates the likelihood of a block being clear (having no blocks on top).
    """

    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None,
                 device: Union[torch.device, None] = None, mockbuild: bool = False):
        """
        Initialize the ClearLikelihood connection.
        
        Args:
            name: Name of the connection
            dtype: Data type for computations
            device: Device for computations
            mockbuild: Whether to run in mock build mode
        """
        super().__init__(name, {"likelihood_clear": (NUM_BLOCKS,),
                                "likelihood_below": (NUM_BLOCKS, NUM_BLOCKS)},
                         dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        """
        Define the connection function that updates clear likelihoods.
        
        Returns:
            function: Connection function that takes likelihood_clear and likelihood_below as input
        """
        def connection_func(likelihood_clear, likelihood_below):
            """
            Update the clear likelihood vector based on below relationships.
            
            Args:
                likelihood_clear: Current clear likelihood vector
                likelihood_below: Below likelihood matrix
            
            Returns:
                tensor: Innovation term for updating clear likelihoods
            """
            likelihood_below = likelihood_below * (likelihood_below > 0.5).detach()
            new_likelihood = 1.0 - gradient_preserving_clipping(torch.sum(likelihood_below, dim=0), 0.001, 0.999)
            return new_likelihood - likelihood_clear

        return connection_func

