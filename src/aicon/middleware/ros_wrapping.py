import multiprocessing as mp
import time
import traceback
from typing import Dict, List, Union, Callable

import torch
from aicon.middleware.util_ros import ROSWrappingConnection

from aicon.base_classes.components import Component, ActionComponent
from aicon.base_classes.connections import ActiveInterconnection

all_instantiated_wrappers = []
all_ros_subprocesses = []


"""
This was based on ROS1 code.
The public version currently does not support ROS2 yet. 
"""


class ComponentROSWrapper:

    subscribers: Dict[str, object]
    callback_connections: Dict[str, List[ActiveInterconnection]]
    publishers: Dict[str, object]

    def __init__(self, component:Component, ros_namespace:str = "/direl/", collect_all_derivatives : bool = False):
        pass

    def quantity_callback(self, msg, device, dtype):
        pass

    def derivative_callback(self, msg, device, dtype):
        pass

    def publish_component_state(self, component:Component):
        pass

    def shutdown(self):
        pass

    def __del__(self):
        global all_instantiated_wrappers
        try:
            all_instantiated_wrappers.remove(self)
        except ValueError:
            pass


def ros_wrap_components(components: List[Callable], *, ros_namespace:str="direl_quantities/",
                        node_name:Union[str,None]=None, do_not_init_node:bool=False, log_level=None,
                        spin:bool=False, rate:int=10, passive_mode: bool = False,
                        component_building_functions : Union[Dict[str, Callable], None] = None,
                        connection_building_functions : Union[Dict[str, Callable], None] = None,
                        viz_kwargs : Union[Dict, None] = None):
    pass


def get_all_active_ros_wrappers():
    pass


def build_components_with_ros_middleware(component_building_functions:Dict[str, Callable], passive_mode: bool = False, connection_building_functions : Union[Dict[str, Callable], None] = None, viz_kwargs : Union[Dict, None] = None, frame_rates : Union[None,Dict[str, int]] = None):
    pass


def wait_for_ros_wrapped_components():
    pass
