import torch

from aicon.blocksworld_experiment.actions import BlockPuttingAction
from aicon.blocksworld_experiment.connections import build_connections
from aicon.blocksworld_experiment.estimators import ClearEstimator, BelowEstimator
from aicon.blocksworld_experiment.goal import SmartStackAonBonC, StackAonBGoal, StackAonBonCGoal, UnstackAonBGoal
from aicon.blocksworld_experiment.sensors import BlocksBelowSensor


def get_building_functions_basic_blocks_world(init_setup: str = "3towers", goal: str = "StackAonB"):
    connection_builders = build_connections()
    match goal:
        case "StackAonB":
            goal_class = StackAonBGoal
        case "StackAonBonC":
            goal_class = StackAonBonCGoal
        case "SmartStackAonBonC":
            goal_class = SmartStackAonBonC
        case "UnstackAonB":
            goal_class = UnstackAonBGoal
    diff_depth = 12
    component_builders = {"BlocksBelowSensor": lambda mockbuild: BlocksBelowSensor("BlocksBelowSensor", connections={
        k: connection_builders[k] for k in ("BelowLikelihoodDummySensing",)}, device=torch.device("cpu"),
                                                                                   dtype=torch.double,
                                                                                   mockbuild=mockbuild,setup=init_setup),
        "BelowEstimator": lambda mockbuild: BelowEstimator("BelowEstimator",
                                                           connections={k: connection_builders[k] for k in (
                                                           "BelowLikelihoodDummySensing", "ClearLikelihood",
                                                           "BelowLikelihood")}, goals={
                goal: lambda d, t, mockbuild=False: goal_class(is_active=True, dtype=t, device=d,
                    mockbuild=mockbuild)}, device=torch.device("cpu"), dtype=torch.double, mockbuild=mockbuild,
                                                           prevent_loops_in_differentiation=False,
                                                           max_length_differentiation_trace=diff_depth),
        "ClearEstimator": lambda mockbuild: ClearEstimator("ClearEstimator",
                                                           connections={k: connection_builders[k] for k in
                                                                        ("ClearLikelihood", "BelowLikelihood")},
                                                           device=torch.device("cpu"), dtype=torch.double,
                                                           mockbuild=mockbuild, prevent_loops_in_differentiation=False,
                                                           max_length_differentiation_trace=diff_depth),
        "BlockPuttingAction": lambda mockbuild: BlockPuttingAction("BlockPuttingAction",
                                                                   connections={k: connection_builders[k] for k in
                                                                                ("BelowLikelihood","BelowLikelihoodDummySensing")},
                                                                   device=torch.device("cpu"), dtype=torch.double,
                                                                   mockbuild=mockbuild), }
    frame_rates = {"BlocksBelowSensor": 10, "BelowEstimator": 10, "ClearEstimator": 10, "BlockPuttingAction": 10, }
    return component_builders, connection_builders, frame_rates
