import multiprocessing as mp
import time
import traceback
from typing import Dict, List, Union, Callable

import rospy
import torch
from aicon_msgs.msg import QuantityAndForwardDerivativesMsg, DerivativesMsg

from aicon.base_classes.components import Component, ActionComponent
from aicon.base_classes.connections import Connection
from aicon.middleware.util_ros import torch_tensor_to_ros_multiarray, ros_multiarray_to_torch_tensor, \
    derivative_block_to_derivative_msg, derivative_msg_to_derivative_dict, ROSWrappingConnection

all_instantiated_wrappers = []
all_ros_subprocesses = []


class ComponentROSWrapper:

    subscribers: Dict[str, rospy.Subscriber]
    callback_connections: Dict[str, List[Connection]]
    publishers: Dict[str, rospy.Publisher]

    def __init__(self, component:Component, ros_namespace:str = "/direl/", collect_all_derivatives : bool = False):
        self.active = True
        self.publishers = dict()
        self.backward_derivative_publishers = dict()
        self.subscribers = dict()
        self.backward_derivative_subscribers = dict()
        with component.lock:
            self.callback_connections = dict()
            for c in component.connections.values():
                for q in c.connected_quantities.keys():
                    if q not in component.quantities.keys():
                        # enable value subscription
                        if q not in self.callback_connections.keys():
                            self.callback_connections[q] = []
                            self.subscribers[q] = rospy.Subscriber(ros_namespace + "quantities/" + q, QuantityAndForwardDerivativesMsg, lambda x: self.quantity_callback(x, dtype=component.dtype, device=component.device), queue_size=10)
                        self.callback_connections[q].append(c)
                        # and derivative publishing
                        if q not in self.backward_derivative_publishers:
                            self.backward_derivative_publishers[q] = rospy.Publisher(ros_namespace + "derivatives/" + q, DerivativesMsg,
                                          queue_size=10)

            if "ROSWrapping" not in component.connections:
                ROS_connection = ROSWrappingConnection("ROSWrapping", {k: v.shape for k,v in component.quantities.items()}, dtype=component.dtype, device=component.device)
                component.connections["ROSWrapping"] = ROS_connection
            else:
                ROS_connection = component.connections["ROSWrapping"]

            for q in component.quantities.keys():
                self.publishers[q] = rospy.Publisher(ros_namespace + "quantities/" + q, QuantityAndForwardDerivativesMsg,
                                          queue_size=10)

            self.derivative_callback_connections = dict()
            if collect_all_derivatives:
                derivative_quantities = list(self.callback_connections.keys())
                derivative_quantities.extend(list(component.quantities.keys()))
            else:
                derivative_quantities = component.quantities.keys()
            for q in derivative_quantities:
                self.backward_derivative_subscribers[q] = rospy.Subscriber(ros_namespace + "derivatives/" + q, DerivativesMsg, lambda x: self.derivative_callback(x, dtype=component.dtype, device=component.device), queue_size=10)
                if q not in self.derivative_callback_connections:
                    self.derivative_callback_connections[q] = []
                self.derivative_callback_connections[q].append(ROS_connection)

            component.other_synchronization_functions.append(self.publish_component_state)
            global all_instantiated_wrappers
            all_instantiated_wrappers.append(self)

    def quantity_callback(self, msg, device, dtype):
        try:
            q_value = ros_multiarray_to_torch_tensor(msg.quantity.quantity_value, dtype=dtype, device=device)
            timestamp = torch.tensor(msg.header.stamp.to_sec(), dtype=dtype, device=device)
            quantity_name = msg.name
            forward_derivatives = derivative_msg_to_derivative_dict(msg.derivatives_forward, dtype=dtype, device=device)
            for c in self.callback_connections[quantity_name]:
                with c.lock:
                    c.connected_quantities[quantity_name] = q_value
                    c.connected_quantities_initialized[quantity_name] = True
                    c.connected_quantities_timestamps[quantity_name] = timestamp
                    c.derivatives_forward_mode.update_(forward_derivatives)
        except Exception as e:
            print("In ROS middleware got exception: ")
            traceback.print_exc()

    def derivative_callback(self, msg, device, dtype):
        try:
            quantity_name = msg.name
            backward_derivatives = derivative_msg_to_derivative_dict(msg, dtype=dtype, device=device)
            for c in self.derivative_callback_connections[quantity_name]:
                with c.lock:
                    c.derivatives_backward_mode.update_(backward_derivatives)
        except Exception as e:
            print("In ROS middleware got exception: ")
            traceback.print_exc()

    def publish_component_state(self, component:Component):
        start = time.time()
        try:
            if not self.active or not component.initialized_all_quantities:
                return
            timestamp = component.timestamp
            quantities = component.quantities
            forward_derivatives = component.derivatives_forward_mode
            backward_derivatives = component.derivatives_backward_mode
            # print("Forward "+str(forward_derivatives))
            # print("Backward "+str(backward_derivatives))
            for q_name, q_value in quantities.items():

                msg = QuantityAndForwardDerivativesMsg()
                msg.header.stamp = rospy.Time.from_sec(timestamp.cpu().numpy())
                msg.name = q_name
                msg.quantity.name = q_name
                msg.quantity.quantity_value = torch_tensor_to_ros_multiarray(q_value, msg.quantity.quantity_value)
                if q_name in forward_derivatives:
                    msg.derivatives_forward = derivative_block_to_derivative_msg(forward_derivatives[q_name],
                                                                                 msg.derivatives_forward)

                self.publishers[q_name].publish(msg)
            for q_name in backward_derivatives.derivative_blocks.keys():
                if q_name in self.backward_derivative_publishers:
                    derivative_msg = DerivativesMsg()
                    derivative_msg = derivative_block_to_derivative_msg(backward_derivatives[q_name], derivative_msg)
                    self.backward_derivative_publishers[q_name].publish(derivative_msg)
        except Exception:
            print("Unable to publish " + component.name)
            traceback.print_exc()
        # print(component.name, ": Publish Time", time.time() - start)

    def shutdown(self):
        self.active = False
        for s in self.subscribers.values():
            s.unregister()
        for p in self.publishers.values():
            p.unregister()

    def __del__(self):
        self.shutdown()
        global all_instantiated_wrappers
        all_instantiated_wrappers.remove(self)


def ros_wrap_components(components: List[Callable], *, ros_namespace:str="direl_quantities/",
                        node_name:Union[str,None]=None, do_not_init_node:bool=False, log_level=rospy.INFO,
                        spin:bool=False, rate:int=10, passive_mode: bool = False,
                        component_building_functions : Union[Dict[str, Callable], None] = None,
                        connection_building_functions : Union[Dict[str, Callable], None] = None,
                        viz_kwargs : Union[Dict, None] = None):

    if not do_not_init_node:
        if node_name is None:
            node_name = "RIDE_Anonymous"
            anonymous = True
        else:
            anonymous = False
        rospy.init_node(node_name, log_level=log_level, anonymous=anonymous)
        print("Started Node "+str(node_name))
    built_components = []
    for c in components:
        try:
            built_components.append(c(mockbuild=False))
            print("Done building "+node_name)
        except BaseException as e:
            print("Error in building component for node "+node_name)
            traceback.print_exc()
            raise e
    for c in built_components:
        try:
            ComponentROSWrapper(component=c, ros_namespace=ros_namespace)
            print("ROS wrapped " + str(c.name))
        except Exception as e:
            print("Error in ROS wrapping component for node "+node_name)
            traceback.print_exc()
            raise e

    def shutdown_wrappers():
        wrappers = all_instantiated_wrappers.copy()
        for c in built_components:
            if isinstance(c, ActionComponent):
                if c.active:
                    c.stop()
        for w in wrappers:
            w.shutdown()
    rospy.on_shutdown(shutdown_wrappers)

    if spin:
        rospy.sleep(0.5)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            for c in built_components:
                if isinstance(c, ActionComponent):
                    if not c.active and not passive_mode:
                        c.start()
                c.update(torch.tensor(rospy.get_rostime().to_sec(), dtype=c.dtype, device=c.device))
            rate.sleep()


def get_all_active_ros_wrappers():
    return all_instantiated_wrappers.copy()


def build_components_with_ros_middleware(component_building_functions:Dict[str, Callable], passive_mode: bool = False, connection_building_functions : Union[Dict[str, Callable], None] = None, viz_kwargs : Union[Dict, None] = None, frame_rates : Union[None,Dict[str, int]] = None):
    global all_ros_subprocesses
    ctx = mp.get_context('fork')
    if frame_rates is None:
        frame_rates = dict()
    for n, f in component_building_functions.items():
        if n in frame_rates:
            rate = frame_rates[n]
        else:
            rate = 10
        p = ctx.Process(target=lambda: ros_wrap_components([f], spin=True, do_not_init_node=False, node_name=n, passive_mode=passive_mode, rate=rate, component_building_functions=component_building_functions, connection_building_functions=connection_building_functions, viz_kwargs=viz_kwargs), name=n)
        p.start()
        print("Started Process "+str(n))
        all_ros_subprocesses.append(p)

    def shutdown_processes():
        for p in all_ros_subprocesses:
            p.terminate()

    rospy.on_shutdown(shutdown_processes)
    return all_ros_subprocesses


def wait_for_ros_wrapped_components():
    global all_ros_subprocesses
    try:
        _ = [p.join() for p in all_ros_subprocesses]
    except KeyboardInterrupt:
        [p.terminate() for p in all_ros_subprocesses]
        _ = [p.join() for p in all_ros_subprocesses]
        raise KeyboardInterrupt
