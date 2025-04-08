import inspect
import time
import traceback
from abc import abstractmethod
from threading import Lock
from typing import Dict, Type, List, Callable, Tuple, Union

import torch
from tensordict import TensorDict

from aicon.base_classes.connections import ActiveInterconnection
from aicon.base_classes.derivatives import DerivativeDict, compute_derivatives, \
    get_steepest_gradient_from_derivative_block
from aicon.base_classes.goals import evaluate_goals, Goal
from aicon.base_classes.util import collect_args, collect_inputs, collect_derivatives, gather_partial_derivatives, \
    generate_outputdict, collect_shapes, collect_required_signs


class Component:
    """
    Base class for all components in the AICON framework.
    
    A Component represents a functional unit in the system that can:
    - Maintain its own state
    - Communicate with other components through connections
    - Process data and perform computations
    - Support automatic differentiation
    
    Attributes:
        name (str): Unique identifier for the component
        lock (Lock): Thread lock for thread-safe operations
        dtype (torch.dtype): Data type for tensors
        device (torch.device): Device (CPU/GPU) for computations
        connections (Dict[str, Type[Connection]]): Dictionary of connections to other components
        quantities (TensorDict): Dictionary of component-specific quantities
        timestamp (torch.Tensor): Current timestamp
        goals (Dict[str, Type[Goal]]): Dictionary of goals to be achieved
        partial_derivatives (TensorDict): Partial derivatives of quantities
        derivatives_forward_mode (DerivativeDict): Forward mode derivatives
        derivatives_backward_mode (DerivativeDict): Backward mode derivatives
        finished_forward_derivatives (DerivativeDict): Completed forward derivatives
        other_synchronization_functions (List[Callable]): Additional sync functions
    """
    
    name: str
    lock: Lock
    dtype: torch.dtype
    device: torch.device
    connections: Dict[str, Type[ActiveInterconnection]]
    quantities: TensorDict
    timestamp: torch.Tensor
    goals: Dict[str, Type[Goal]]
    partial_derivatives: TensorDict
    derivatives_forward_mode: DerivativeDict
    derivatives_backward_mode: DerivativeDict
    finished_forward_derivatives: DerivativeDict
    other_synchronization_functions: List[Callable]

    def __init__(self, name:str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        """
        Initialize a new component.
        
        Args:
            name: Unique identifier for the component
            connections: Dictionary of connection factory functions
            goals: Optional dictionary of goal factory functions
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
        """
        self.name = name
        self.lock = Lock()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.connections = {k: connections[k](device, dtype, mockbuild) for k in connections.keys()}
        if goals is None:
            self.goals = dict()
        else:
            self.goals = {k: goals[k](device, dtype, mockbuild) for k in goals.keys()}
        self.quantities = TensorDict(dict(), device=device, batch_size=())
        self.timestamp = torch.zeros(1, device=device, dtype=dtype)
        self.partial_derivatives = TensorDict(dict(), device=device, batch_size=())
        self.derivatives_forward_mode = DerivativeDict()
        self.derivatives_backward_mode = DerivativeDict()
        self.finished_forward_derivatives = DerivativeDict()
        self.other_synchronization_functions = list()
        self.initial_definitions()
        self.initialized_all_quantities = False

    def get_dt(self, new_time: torch.tensor) -> torch.tensor:
        """
        Calculate the time difference since last update.
        
        Args:
            new_time: Current timestamp
            
        Returns:
            Time difference
        """
        return new_time - self.timestamp

    @abstractmethod
    def initialize_quantities(self) -> bool:
        """
        Initialize component-specific quantities.
        
        Returns:
            bool: True if initialization was successful
        """
        return False

    @abstractmethod
    def initial_definitions(self) -> None:
        """
        Define initial component properties and setup.
        This method should be implemented by subclasses.
        """
        pass

    def update(self, new_time: torch.tensor) -> None:
        """
        Update the component's state.
        
        Args:
            new_time: Current timestamp
        """
        if not self.initialized_all_quantities:
            self._attempt_init(new_time)
        else:
            self._attempt_update(new_time)
        self._attempt_synch(new_time)

    def _attempt_synch(self, new_time: torch.tensor) -> None:
        """
        Attempt to synchronize with connected components.
        
        Args:
            new_time: Current timestamp
        """
        try:
            with self.lock:
                self.timestamp = new_time
                self.synchronize()
        except Exception:
            print("Could not synchronize " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _attempt_update(self, new_time: torch.tensor) -> None:
        """
        Attempt to update the component's state.
        
        Args:
            new_time: Current timestamp
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def synchronize(self) -> None:
        """
        Synchronize state and derivatives with connected components.
        """
        for c in self.connections.values():
            try:
                with c.lock:
                    c.derivatives_forward_mode.update_(self.derivatives_forward_mode)
                    c.derivatives_backward_mode.update_(self.derivatives_backward_mode)
                    c.connected_quantities.update_(self.quantities)
                    for k in self.quantities.keys():
                        c.connected_quantities_timestamps[k] = self.timestamp
                        c.connected_quantities_initialized[k] = self.initialized_all_quantities
            except (ValueError, KeyError):
                print("During Synchronization with Connection "+str(c.name)+" encountered following Issue:")
                traceback.print_exc()
                print("New Updated Values from Compoenent:")
                print(self.quantities)
                print("Connection Previous Value:")
                with c.lock:
                    print(c.connected_quantities)
        for sync_func in self.other_synchronization_functions:
            sync_func(self)

    def _attempt_init(self, new_time: torch.tensor) -> None:
        """
        Attempt to initialize the component.
        
        Args:
            new_time: Current timestamp
        """
        print(self.name + " not initialized yet")
        try:
            with self.lock:
                self.initialized_all_quantities = self.initialize_quantities()
        except Exception:
            print("Could not initialize " + str(self.name) + ":")
            traceback.print_exc()


class SensorComponent(Component):
    """
    Base class for sensor components.
    
    A SensorComponent is responsible for:
    - Obtaining measurements from the environment
    - Processing sensor data
    - Providing measurements to other components
    """
    
    def __init__(self, name: str, connections: Dict[str, Callable],
                 dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        """
        Initialize a new sensor component.
        
        Args:
            name: Unique identifier for the sensor
            connections: Dictionary of connection factory functions
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
        """
        super().__init__(name, connections, None, dtype, device, mockbuild=mockbuild)

    @abstractmethod
    def obtain_measurements(self) -> bool:
        """
        Obtain new measurements from the sensor.
        
        Returns:
            bool: True if measurements were obtained successfully
        """
        return False

    def initialize_quantities(self) -> bool:
        """
        Initialize sensor quantities by obtaining initial measurements.
        
        Returns:
            bool: True if initialization was successful
        """
        return self.obtain_measurements()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        """
        Attempt to update sensor measurements.
        
        Args:
            new_time: Current timestamp
        """
        try:
            with self.lock:
                self.obtain_measurements()
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()


class ActionComponent(Component):
    """
    Base class for action components.
    
    An ActionComponent is responsible for:
    - Executing control actions
    - Computing control policies
    - Supporting gradient-based optimization
    """
    
    def __init__(self, name: str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, forward_mode:bool=False, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        """
        Initialize a new action component.
        
        Args:
            name: Unique identifier for the action component
            connections: Dictionary of connection factory functions
            goals: Optional dictionary of goal factory functions
            forward_mode: Whether to use forward mode differentiation
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
        """
        super().__init__(name, connections, goals, dtype, device, mockbuild=mockbuild)
        self.active = False
        self.forward_mode = forward_mode

    def start(self) -> None:
        """
        Start the action component.
        """
        try:
            self._start()
            self.active = True
        except Exception:
            print("Could not start " + str(self.name) + ":")
            traceback.print_exc()

    def stop(self) -> None:
        """
        Stop the action component.
        """
        try:
            self._stop()
            self.active = False
        except Exception:
            print("Could not stop " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _start(self) -> None:
        """
        Component-specific startup procedure.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        """
        Component-specific shutdown procedure.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_steepest_gradient(self, quantity : str, time_threshold: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
        """
        Get the steepest gradient for a quantity.
        
        Args:
            quantity: Name of the quantity
            time_threshold: Optional time threshold for gradient computation
            
        Returns:
            Tuple containing:
            - Steepest gradient
            - List of timestamps
            - List of component names in the gradient path
        """
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            print("No Gradients for quantity "+quantity+ " available yet")
            return torch.zeros_like(self.quantities[quantity]), [], []

        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=self.dtype, device=self.device)
        steepest_grad, timestamps, trace = get_steepest_gradient_from_derivative_block(my_derivatives, time_threshold)
        return steepest_grad, timestamps, trace

    @abstractmethod
    def determine_new_actions(self) -> None:
        """
        Determine new actions to execute.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        """
        Attempt to update the action component.
        
        Args:
            new_time: Current timestamp
        """
        try:
            with self.lock:
                self.determine_new_actions()
                if self.active:
                    self.send_action_values()
                else:
                    pass
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def send_action_values(self) -> None:
        """
        Send action values to the environment.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()


def gradient_descent(gain, last_action, steepest_grad):
    """
    Perform gradient descent on an action.
    
    Args:
        gain: Learning rate/gain
        last_action: Previous action
        steepest_grad: Steepest gradient
        
    Returns:
        New action after gradient descent
    """
    if torch.norm(steepest_grad) > 0 and torch.norm(last_action) > 0:
        # gradient descent
        new_action = last_action / torch.norm(last_action) - gain * steepest_grad / torch.norm(
            steepest_grad)
    elif torch.norm(steepest_grad) > 0:
        new_action = - steepest_grad / torch.norm(
            steepest_grad)
    else:
        # no action
        new_action = torch.zeros_like(last_action[:3])
    return new_action


class EstimationComponent(Component):
    """
    Base class for estimation components.
    
    An EstimationComponent is responsible for:
    - Processing and fusing sensor data
    - Maintaining state estimates
    - Supporting automatic differentiation
    """
    
    def __init__(self, name: str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, no_differentiation : bool = False, prevent_loops_in_differentiation : bool = True, max_length_differentiation_trace : Union[int, None] = None):
        """
        Initialize a new estimation component.
        
        Args:
            name: Unique identifier for the estimation component
            connections: Dictionary of connection factory functions
            goals: Optional dictionary of goal factory functions
            dtype: Optional data type for tensors
            device: Optional device for computations
            mockbuild: Whether to build in mock mode
            no_differentiation: Whether to disable differentiation
            prevent_loops_in_differentiation: Whether to prevent loops in differentiation
            max_length_differentiation_trace: Maximum length of differentiation trace
        """
        self.no_differentiation = no_differentiation
        self.prevent_loops_in_differentiation = prevent_loops_in_differentiation
        self.max_length_differentiation_trace = max_length_differentiation_trace
        super().__init__(name, connections, goals, dtype, device, mockbuild=mockbuild)
        if not mockbuild:
            f_func, output_signature_diff, output_signature_non_diff = self.define_estimation_function_f()
            self.f_input_signature = inspect.getfullargspec(f_func).args
            self.f_output_signature_diff = output_signature_diff
            self.f_output_signature_non_diff = output_signature_non_diff
            self.f_func = f_func
            self.idxs_diff_against_args = None
            self.goal_values = None

    @abstractmethod
    def define_estimation_function_f(self) -> Tuple[Callable, List[str], List[str]]:
        """
        Define the estimation function and its signatures.
        
        Returns:
            Tuple containing:
            - Estimation function
            - List of differentiable output names
            - List of non-differentiable output names
        """
        raise NotImplementedError()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            self.update_state_and_derivatives(new_time)
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    def update_state_and_derivatives(self, new_time: torch.tensor) -> None:
        start = time.time()
        with self.lock:
            all_inputs = collect_inputs(self.quantities, self.connections, self.get_dt(new_time))
            relevant_forward_derivatives, relevant_backward_derivatives = collect_derivatives(self.connections)
            required_signs_dict = collect_required_signs(self.connections)
        args, diff_argnums, input_names_diff = collect_args(all_inputs, self.f_input_signature,
                                                            self.idxs_diff_against_args, self.f_output_signature_diff)
        up = time.time()
        outputs, partial_jacs = self.inference_and_differentiation(args, diff_argnums)
        inf = time.time()
        output_dict = generate_outputdict(outputs, self.f_output_signature_non_diff)
        backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives = self.derivative_processing(
            all_inputs, input_names_diff, new_time, output_dict, partial_jacs, relevant_backward_derivatives,
            relevant_forward_derivatives, required_signs_dict)
        now = time.time()
        with self.lock:
            self.quantities = self.quantities.update(output_dict)
            self.partial_derivatives = self.partial_derivatives.update(partial_derivatives)
            self.derivatives_forward_mode = self.derivatives_forward_mode.update(forward_derivatives)
            self.derivatives_backward_mode = self.derivatives_backward_mode.update(backward_derivatives)
            self.finished_forward_derivatives = self.finished_forward_derivatives.update(
                finished_forward_derivatives)

    def derivative_processing(self, all_inputs, input_names_diff, new_time, output_dict, partial_jacs,
                              relevant_backward_derivatives, relevant_forward_derivatives, required_signs_dict):
        if self.no_differentiation:
            return DerivativeDict(), DerivativeDict(), DerivativeDict(), DerivativeDict()
        in_out_shapes = collect_shapes(all_inputs, output_dict)
        partial_derivatives = gather_partial_derivatives(partial_jacs, input_names_diff, self.f_output_signature_diff,
                                                         in_out_shapes)
        goal_derivatives, self.goal_values = evaluate_goals(self.goals.values(), output_dict)
        for k, v in self.goal_values.items():
            if self.goals[k].fulfilled_value is not None:
                if v <= self.goals[k].fulfilled_value:
                    print("Finished Goal "+str(k))
                    self.goals[k].is_active = False
        forward_derivatives, backward_derivatives, finished_forward_derivatives = compute_derivatives(
            partial_derivatives, relevant_forward_derivatives, relevant_backward_derivatives, goal_derivatives,
            input_names_diff, self.f_output_signature_diff, new_time, required_signs_dict=required_signs_dict,
            prevent_repeats=self.prevent_loops_in_differentiation, max_length_trace=self.max_length_differentiation_trace)
        return backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives

    def inference_and_differentiation(self, args, diff_argnums):
        if self.no_differentiation:
            _, outputs = self.f_func(*args)
            partial_jacs = tuple()
        else:
            partial_jacs, outputs = torch.func.jacfwd(self.f_func, argnums=diff_argnums, has_aux=True, randomness="same")(
                *args)
        return outputs, partial_jacs





