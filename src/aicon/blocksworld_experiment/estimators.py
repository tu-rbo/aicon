"""
Estimators for the blocks world experiment.
This module contains components that estimate various states in the blocks world environment,
including block relationships and clear states.
"""

import torch

from aicon.base_classes.components import EstimationComponent
from aicon.blocksworld_experiment.global_params import NUM_BLOCKS
from aicon.inference.util import gradient_preserving_clipping


class BelowEstimator(EstimationComponent):
    """
    Estimates the likelihood that one block is below another block.
    This estimator maintains and updates a matrix of probabilities representing
    the relationships between blocks.
    """

    def initialize_quantities(self):
        """
        Initialize the estimator's quantities based on sensor data.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c = self.connections["BelowLikelihoodDummySensing"]
        with c.lock:
            if c.connected_quantities_initialized["below_state_sensed"]:
                likelihood_below_sensed = c.connected_quantities["below_state_sensed"]
            else:
                return False
        self.quantities["likelihood_below"] = likelihood_below_sensed
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function that updates the below likelihood matrix.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: The estimation function
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["BelowLikelihood"].c_func
        c_func_sensor = self.connections["BelowLikelihoodDummySensing"].c_func

        def f_func(likelihood_below, action_blocks, likelihood_clear, below_state_sensed):
            """
            Update the below likelihood matrix based on actions and sensor data.
            
            Args:
                likelihood_below: Current below likelihood matrix
                action_blocks: Action matrix indicating block movements
                likelihood_clear: Clear state likelihood vector
                below_state_sensed: Sensor data for below relationships
            
            Returns:
                tuple: (new_likelihood, new_likelihood)
            """
            innovation = c_func(likelihood_below, action_blocks, likelihood_clear)
            new_likelihood = likelihood_below + innovation
            new_likelihood = new_likelihood + c_func_sensor(new_likelihood, below_state_sensed).detach() # no grad from sensing
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.001, 0.999)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_below"], ["likelihood_below"]

    def initial_definitions(self):
        """Initialize the below likelihood matrix with zeros."""
        self.quantities["likelihood_below"] = torch.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=self.dtype, device=self.device)


class ClearEstimator(EstimationComponent):
    """
    Estimates the likelihood that a block is clear (has no blocks on top of it).
    This estimator maintains a vector of probabilities for each block.
    """

    def initialize_quantities(self):
        """
        Initialize the estimator's quantities based on below likelihood data.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        c = self.connections["ClearLikelihood"]
        with c.lock:
            if c.connected_quantities_initialized["likelihood_below"]:
                likelihood_below = c.connected_quantities["likelihood_below"]
                c_func = c.c_func
            else:
                return False
        self.quantities["likelihood_clear"] = torch.clip(
            c_func(torch.zeros(NUM_BLOCKS, dtype=self.dtype, device=self.device), likelihood_below), 0.01, 0.99)
        return True

    def define_estimation_function_f(self):
        """
        Define the estimation function that updates the clear likelihood vector.
        
        Returns:
            tuple: (f_func, input_names, output_names)
            f_func: The estimation function
            input_names: Names of input quantities
            output_names: Names of output quantities
        """
        c_func = self.connections["ClearLikelihood"].c_func

        def f_func(likelihood_clear, likelihood_below):
            """
            Update the clear likelihood vector based on below relationships.
            
            Args:
                likelihood_clear: Current clear likelihood vector
                likelihood_below: Below likelihood matrix
            
            Returns:
                tuple: (new_likelihood, new_likelihood)
            """
            innovation = c_func(likelihood_clear, likelihood_below)
            new_likelihood = likelihood_clear + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.01, 0.99)
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_clear"], ["likelihood_clear"]

    def initial_definitions(self):
        """Initialize the clear likelihood vector with zeros."""
        self.quantities["likelihood_clear"] = torch.zeros(NUM_BLOCKS, dtype=self.dtype, device=self.device)