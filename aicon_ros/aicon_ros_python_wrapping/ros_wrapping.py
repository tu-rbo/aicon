"""
Python Wrappers to deploy a specified AICON system into multiple ROS Nodes that exchange messages

Author: Vito Mengers
License: MIT
"""

from __future__ import annotations

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import multiprocessing as mp
import random
import threading
import time
import traceback
from typing import Dict, List, Union, Callable, Type

import rclpy
import torch
from aicon_ros.msg import QuantityAndForwardDerivativesMsg, DerivativesMsg, ListOfStringListsMsg, StringListMsg
from aicon_ros_python_wrapping.util_ros import (
    torch_tensor_to_ros_multiarray,
    ros_multiarray_to_torch_tensor,
    derivative_block_to_derivative_msg,
    derivative_msg_to_derivative_dict,
    ROSWrappingConnection,
    time_float_to_ros,
    time_ros_to_torch,
)

from aicon_base.components import Component, ActionComponent, EstimationComponent
from aicon_base.connections import ActiveInterconnection
from aicon_base.visualization.visualization_component import VisualizationComponent
from aicon_base.timestamps import as_timestamp

all_instantiated_wrappers = []
all_ros_subprocesses = []

default_qos_settings = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=2,
)

default_ros_namespace = "aicon"


class ComponentROSWrapper:
    subscribers: Dict[str, rclpy.node.Subscription]
    callback_connections: Dict[str, List[Type[ActiveInterconnection]]]
    publishers: Dict[str, rclpy.node.Publisher]

    def __init__(self, component: Component, node: rclpy.node.Node, collect_all_derivatives: bool = False):
        self.active = True
        self.publishers = dict()
        self.backward_derivative_publishers = dict()
        self.combined_backward_publishers = dict()
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
                            self.subscribers[q] = node.create_subscription(
                                QuantityAndForwardDerivativesMsg,
                                "quantities/" + q,
                                lambda x: self.quantity_callback(x, dtype=component.dtype, device=component.device),
                                default_qos_settings,
                            )
                        self.callback_connections[q].append(c)
                        # and derivative publishing
                        if q not in self.backward_derivative_publishers:
                            self.backward_derivative_publishers[q] = node.create_publisher(
                                DerivativesMsg, "derivatives/" + q, default_qos_settings
                            )

            if "ROSWrapping" not in component.connections:
                ros_connection = ROSWrappingConnection(
                    "ROSWrapping",
                    {k: v.shape for k, v in component.quantities.items()},
                    dtype=component.dtype,
                    device=component.device,
                )
                component.connections["ROSWrapping"] = ros_connection
            else:
                ros_connection = component.connections["ROSWrapping"]

            for q in component.quantities.keys():
                self.publishers[q] = node.create_publisher(
                    QuantityAndForwardDerivativesMsg, "quantities/" + q, default_qos_settings
                )
                if isinstance(component, EstimationComponent):
                    self.combined_backward_publishers[q] = node.create_publisher(
                        DerivativesMsg, "pursued_gradient/" + q, default_qos_settings
                    )

            if isinstance(component, ActionComponent):
                quantities = list(component.quantities.keys())
                assert len(quantities) == 1
                q = quantities[0]
                self.publishers["used_gradients"] = node.create_publisher(
                    ListOfStringListsMsg, "used_gradients/" + q, default_qos_settings
                )

            self.derivative_callback_connections = dict()
            if collect_all_derivatives:
                derivative_quantities = list(self.callback_connections.keys())
                derivative_quantities.extend(list(component.quantities.keys()))
            else:
                derivative_quantities = component.quantities.keys()
            for q in derivative_quantities:
                self.backward_derivative_subscribers[q] = node.create_subscription(
                    DerivativesMsg,
                    "derivatives/" + q,
                    lambda x: self.derivative_callback(x, dtype=component.dtype, device=component.device),
                    default_qos_settings,
                )
                if q not in self.derivative_callback_connections:
                    self.derivative_callback_connections[q] = []
                self.derivative_callback_connections[q].append(ros_connection)

            # TODO how to handle goals

            component.other_synchronization_functions.append(self.publish_component_state)
            all_instantiated_wrappers.append(self)

    def quantity_callback(self, msg, device, dtype):
        try:
            q_value = ros_multiarray_to_torch_tensor(msg.quantity.quantity_value, dtype=dtype, device=device)
            timestamp = as_timestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            quantity_name = msg.name
            forward_derivatives = derivative_msg_to_derivative_dict(msg.derivatives_forward, dtype=dtype, device=device)
            for c in self.callback_connections[quantity_name]:
                with c.lock:
                    c.connected_quantities[quantity_name] = q_value
                    c.connected_quantities_initialized[quantity_name] = True
                    c.connected_quantities_timestamps[quantity_name] = timestamp
                    c.derivatives_forward_mode.update_(forward_derivatives)
        except Exception:
            print("In ROS middleware got exception: ")
            traceback.print_exc()

    def derivative_callback(self, msg, device, dtype):
        try:
            quantity_name = msg.name
            backward_derivatives = derivative_msg_to_derivative_dict(msg, dtype=dtype, device=device)
            for c in self.derivative_callback_connections[quantity_name]:
                with c.lock:
                    c.derivatives_backward_mode.update_(backward_derivatives)
        except Exception:
            print("In ROS middleware got exception: ")
            traceback.print_exc()

    def publish_component_state(self, component: Component):
        try:
            if not self.active or not component.initialized_all_quantities:
                return
            timestamp = component.timestamp
            quantities = component.quantities
            forward_derivatives = component.derivatives_forward_mode
            backward_derivatives = component.derivatives_backward_mode
            for q_name, q_value in quantities.items():
                msg = QuantityAndForwardDerivativesMsg()
                t = timestamp.cpu().numpy()
                ros_time = time_float_to_ros(t)
                msg.header.stamp = ros_time.to_msg()
                msg.name = q_name
                msg.quantity.name = q_name
                msg.quantity.quantity_value = torch_tensor_to_ros_multiarray(q_value, msg.quantity.quantity_value)
                if q_name in forward_derivatives:
                    msg.derivatives_forward = derivative_block_to_derivative_msg(
                        forward_derivatives[q_name], msg.derivatives_forward
                    )
                self.publishers[q_name].publish(msg)
            for q_name in backward_derivatives.keys():
                if q_name in self.backward_derivative_publishers:
                    derivative_msg = DerivativesMsg()
                    derivative_msg = derivative_block_to_derivative_msg(backward_derivatives[q_name], derivative_msg)
                    self.backward_derivative_publishers[q_name].publish(derivative_msg)
            if isinstance(component, EstimationComponent):
                pursued_gradients = component.last_combined_backward
                for q_name in pursued_gradients.keys():
                    if q_name in self.combined_backward_publishers:
                        derivative_msg = DerivativesMsg()
                        derivative_msg = derivative_block_to_derivative_msg(pursued_gradients[q_name], derivative_msg)
                        self.combined_backward_publishers[q_name].publish(derivative_msg)
            if isinstance(component, ActionComponent):
                local_traces, mix_ins = component.used_gradients
                msg = ListOfStringListsMsg()
                for trace in local_traces:
                    submsg = StringListMsg()
                    [submsg.string_list.append(x) for x in trace]
                    msg.mixed_in_at_node.append(submsg)
                self.publishers["used_gradients"].publish(msg)
        except Exception:
            print("Unable to publish " + component.name)
            traceback.print_exc()

    def shutdown(self):
        self.active = False

    def __del__(self):
        self.shutdown()
        all_instantiated_wrappers.remove(self)


def ros_wrap_components(
    components: List[Callable],
    *,
    ros_namespace: str = default_ros_namespace,
    node: Union[rclpy.node.Node, None] = None,
    node_name: Union[str, None] = None,
    spin: bool = False,
    rate: int = 10,
    passive_mode: bool = False,
    component_building_functions: Union[Dict[str, Callable], None] = None,
    connection_building_functions: Union[Dict[str, Callable], None] = None,
    viz_kwargs: Union[Dict, None] = None,
    num_threads: Union[int, None] = None,
):
    if num_threads is not None:
        # before building the components, whose first inference creates torch's thread pool
        torch.set_num_threads(num_threads)
    if node is None:
        if node_name is None:
            node_name = "unnamed_node" + str(random.randint(0, 1000))
        rclpy.init()
        node = rclpy.create_node(node_name, namespace=ros_namespace)
        print("Started Node " + str(node_name))
    built_components = []
    for c in components:
        try:
            built_components.append(c(mockbuild=False))
            print("Done building " + node_name)
        except BaseException as e:
            print("Error in building component for node " + node_name)
            traceback.print_exc()
            raise e
    wrappers = []
    for c in built_components:
        if c.mocked:
            print(c.name + " only mocked -- won't ROS wrap it")
            continue
        try:
            if not isinstance(c, VisualizationComponent):
                wrappers.append(ComponentROSWrapper(component=c, node=node))
            else:
                wrappers.append(ComponentROSWrapper(component=c, node=node, collect_all_derivatives=True))
                if (component_building_functions is not None) and (connection_building_functions is not None):
                    c.start_network_plotting(component_building_functions, connection_building_functions, **viz_kwargs)
            print("ROS wrapped " + str(c.name))
        except Exception as e:
            print("Error in ROS wrapping component for node " + node_name)
            traceback.print_exc()
            raise e

    if len(wrappers) == 0:
        exit(0)

    def shutdown_wrappers():
        wrappers = all_instantiated_wrappers.copy()
        for c in built_components:
            if isinstance(c, ActionComponent):
                if c.active:
                    c.stop()
        for w in wrappers:
            w.shutdown()

    rclpy.utilities.get_default_context().on_shutdown(shutdown_wrappers)

    time.sleep(0.5)
    if spin:
        spin_thread = threading.Thread(target=spin_node_in_thread, args=(node,), daemon=True)
        spin_thread.start()
        clock = node.get_clock()
        rate = node.create_rate(rate)
        try:
            while rclpy.ok():
                for c in built_components:
                    if c.mocked:
                        continue
                    if isinstance(c, ActionComponent):
                        if not c.active and not passive_mode:
                            c.start()
                    c.update(time_ros_to_torch(clock.now()))
                rate.sleep()
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass


def get_all_active_ros_wrappers():
    return all_instantiated_wrappers.copy()


def spin_node_in_thread(node):
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()


def build_components_with_ros_middleware(
    component_building_functions: Dict[str, Callable],
    passive_mode: bool = False,
    connection_building_functions: Union[Dict[str, Callable], None] = None,
    viz_kwargs: Union[Dict, None] = None,
    frame_rates: Union[None, Dict[str, int]] = None,
    num_threads: Union[None, int, Dict[str, int]] = None,
):
    """Start every component in its own (forked) process with its own ROS node, communicating over topics.

    `frame_rates`: update rate [Hz] per component name (default 10).
    `num_threads`: torch threads per component process (an int for all, or a dict per component; None keeps torch's
    default). Limiting it avoids oversubscribing the CPU, since every process would otherwise start one thread per
    core.
    """
    ctx = mp.get_context("fork")
    if frame_rates is None:
        frame_rates = dict()
    for n, f in component_building_functions.items():
        if n in frame_rates:
            rate = frame_rates[n]
        else:
            rate = 10
        threads = num_threads.get(n) if isinstance(num_threads, dict) else num_threads
        p = ctx.Process(
            target=lambda: ros_wrap_components(
                [f],
                spin=True,
                node_name=n,
                passive_mode=passive_mode,
                rate=rate,
                component_building_functions=component_building_functions,
                connection_building_functions=connection_building_functions,
                viz_kwargs=viz_kwargs,
                num_threads=threads,
            ),
            name=n,
        )
        p.start()
        print("Started Process " + str(n))
        all_ros_subprocesses.append(p)

    def shutdown_processes():
        for p in all_ros_subprocesses:
            p.terminate()

    context = rclpy.Context()
    rclpy.init(context=context)
    context.on_shutdown(shutdown_processes)
    return all_ros_subprocesses


def wait_for_ros_wrapped_components():
    try:
        _ = [p.join() for p in all_ros_subprocesses]
    except KeyboardInterrupt:
        [p.terminate() for p in all_ros_subprocesses]
        _ = [p.join() for p in all_ros_subprocesses]
        raise KeyboardInterrupt
