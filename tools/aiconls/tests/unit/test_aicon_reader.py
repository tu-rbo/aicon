import sys
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper

from reader.aicon_files.aicon_reader import AICONComponentParser
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentQuantityGeneric,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
    SpecGoalProps,
)


def test_aicon_reader_extracts_goals_and_connections():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={"ReachGoal": SpecGoalProps(name="ReachGoal")},
            connections={
                "testConnection": SpecConnectionProps(
                    components=["TestSensor"],
                )
            },
        )
    )

    source = """
class ReachGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(*_args):
            return (0,), (0,)
        return goal_func

class testConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///goals_and_connections.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert set(parser.goals.keys()) == {"ReachGoal"}
    assert set(parser.connections.keys()) == {"testConnection"}
    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["W202", "E206"]


def test_aicon_reader_validates_complex_fixture_implementations(
    complex_spec_content: str,
):
    parsed_spec = parse_spec_content(complex_spec_content)
    assert parsed_spec.spec is not None

    source = """
class EEPoseSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"pose_ee": (6,), "uncertainty_ee": (6, 6)}

class DrawerPoseEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"position_drawer": (3,), "uncertainty_drawer": (3, 3)}

class VeloEEAction(ActionComponent):
    cls_quantity_names_and_shapes = {"action_velo_ee": (3,)}

class DrawerOpenViaJointGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(*_args):
            return (0,), (0,)
        return goal_func

class KeepEEInWorkspaceGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(*_args):
            return (0,), (0,)
        return goal_func

class eeDrawerDistance(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

class eeControlLoop(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {
                "action_velo_ee": (3,),
                "pose_ee": (6,),
                "uncertainty_ee": (6, 6),
                "position_drawer": (3,),
                "uncertainty_drawer": (3, 3),
            },
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )
""".strip()
    parser = AICONComponentParser(
        spec=parsed_spec.spec, document_uri="file:///complex.py"
    )
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert set(parser.components.keys()) == {
        "EEPoseSensor",
        "DrawerPoseEstimator",
        "VeloEEAction",
    }
    assert set(parser.goals.keys()) == {
        "DrawerOpenViaJointGoal",
        "KeepEEInWorkspaceGoal",
    }
    assert set(parser.connections.keys()) == {
        "eeDrawerDistance",
        "eeControlLoop",
    }
    assert len(parser.diagnostics) == 10
    assert [diagnostic.code for diagnostic in parser.diagnostics] == (["W202"] * 9) + [
        "E206"
    ]


def test_aicon_reader_connection_func_rejects_unknown_parameters():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "testConnection": SpecConnectionProps(
                    components=["TestSensor"],
                )
            },
        )
    )

    source = """
class testConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        def connection_func(test_value, wrong_parameter):
            return test_value
        return connection_func
""".strip()
    parser = AICONComponentParser(
        spec, document_uri="file:///custom_connections_file.py"
    )
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E202", "E206"]
    assert (
        parser.diagnostics[0].message
        == "connection_func parameter 'wrong_parameter' in connection "
        "'testConnection' is not defined in super constructor quantities. Allowed "
        "parameters: test_value."
    )


def test_aicon_reader_connection_warns_about_unused_constructor_quantities():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "used_quantity": SpecComponentQuantityGeneric(shape=(1, 4)),
                        "unused_quantity": SpecComponentQuantityGeneric(shape=(1, 4)),
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "testConnection": SpecConnectionProps(
                    components=["TestSensor"],
                )
            },
        )
    )

    source = """
class testConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {"used_quantity": (1, 4), "unused_quantity": (1, 4)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(used_quantity):
            return used_quantity
        return connection_func
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///connections.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["W202", "E206"]
    assert (
        parser.diagnostics[0].message
        == "Quantity 'unused_quantity' from connection 'testConnection' super "
        "constructor is unused in connection_func parameters."
    )
    assert parser.diagnostics[0].range.start.line == 4


def test_aicon_reader_build_connections_reports_missing_entry():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="KnownConnection",
                ),
                "missingConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="MissingConnection",
                ),
            },
        )
    )
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

def build_connections():
    connections = {
        "knownConnection": lambda device, dtype, mockbuild: KnownConnection(
            "knownConnection", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
    return connections
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///connections.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["W202", "E203"]


def test_aicon_reader_build_connections_reports_wrong_class_and_name():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="KnownConnection",
                ),
            },
        )
    )
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

class OtherConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

def build_connections():
    connections = {
        "knownConnection": lambda device, dtype, mockbuild: OtherConnection(
            "wrongName", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
    return connections
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///connections.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == [
        "W202",
        "W201",
        "E204",
    ]


def test_aicon_reader_build_connections_reports_wrong_constructor_name_argument():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="KnownConnection",
                ),
            },
        )
    )
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

def build_connections():
    connections = {
        "knownConnection": lambda device, dtype, mockbuild: KnownConnection(
            "wrongName", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
    return connections
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///connections.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["W202", "E205"]


def test_aicon_reader_get_building_functions_requires_build_connections_call():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="KnownConnection",
                ),
            },
        )
    )
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        def connection_func(test_value):
            return test_value
        return connection_func

def build_connections():
    connections = {
        "knownConnection": lambda device, dtype, mockbuild: KnownConnection(
            "knownConnection", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
    return connections

def get_building_functions(sim_env_pointer=None):
    connection_builders = {}
    return {}, connection_builders, {}
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///builders.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E207"]


def test_aicon_reader_get_building_functions_allows_build_connections_call():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "test_value": SpecComponentQuantityGeneric(shape=(1, 4))
                    },
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                )
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["TestSensor"],
                    class_name="KnownConnection",
                ),
            },
        )
    )
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"test_value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        def connection_func(test_value):
            return test_value
        return connection_func

def build_connections():
    connections = {
        "knownConnection": lambda device, dtype, mockbuild: KnownConnection(
            "knownConnection", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
    return connections

def get_building_functions(sim_env_pointer=None):
    connection_builders = build_connections()
    return {}, connection_builders, {}
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///builders.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert parser.diagnostics == []


def test_aicon_reader_goal_func_rejects_unknown_parameters():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "known_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=["ReachGoal"],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                )
            },
            goals={"ReachGoal": SpecGoalProps(name="ReachGoal")},
            connections={},
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"known_value": (1,)}

class ReachGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(known_value, unknown_value):
            return (0,), (0,)
        return goal_func
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///goals.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E103"]
    assert (
        parser.diagnostics[0].message
        == "goal_func parameter 'unknown_value' in goal 'ReachGoal' is not available "
        "from quantities of connected components owning this goal. Allowed parameters: "
        "known_value."
    )


def test_aicon_reader_goal_func_rejects_disconnected_component_quantities():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "GoalOwnerEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "owner_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=["ReachGoal"],
                    instance_name="GoalOwnerEstimator",
                    class_name="GoalOwnerEstimator",
                ),
                "DisconnectedSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "disconnected_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="DisconnectedSensor",
                    class_name="DisconnectedSensor",
                ),
            },
            goals={"ReachGoal": SpecGoalProps(name="ReachGoal")},
            connections={
                "ownerConnection": SpecConnectionProps(
                    components=["GoalOwnerEstimator", "ConnectedSensor"],
                )
            },
        )
    )

    source = """
class GoalOwnerEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"owner_value": (1,)}

class DisconnectedSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"disconnected_value": (1,)}

class ReachGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(owner_value, disconnected_value):
            return (0,), (0,)
        return goal_func
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///goals.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E103"]
    assert (
        parser.diagnostics[0].message
        == "goal_func parameter 'disconnected_value' in goal 'ReachGoal' is not "
        "available from quantities of connected components owning this goal. "
        "Allowed parameters: owner_value."
    )


def test_aicon_reader_estimator_f_func_rejects_disconnected_quantities():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
                "ConnectedSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "connected_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="ConnectedSensor",
                    class_name="ConnectedSensor",
                ),
                "DisconnectedSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "disconnected_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="DisconnectedSensor",
                    class_name="DisconnectedSensor",
                ),
            },
            goals={},
            connections={
                "estimatorConnection": SpecConnectionProps(
                    components=["TestEstimator", "ConnectedSensor"],
                )
            },
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value, connected_value, disconnected_value):
            return local_value + connected_value + disconnected_value
        return f_func, [], []

class ConnectedSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"connected_value": (1,)}

class DisconnectedSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"disconnected_value": (1,)}
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E106"]
    assert (
        parser.diagnostics[0].message
        == "f_func parameter 'disconnected_value' in estimator 'TestEstimator' is not "
        "available from local quantities or directly connected components. Allowed "
        "parameters: connected_value, dt, local_value."
    )


def test_aicon_reader_estimator_f_func_accepts_local_and_directly_connected_quantities():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
                "ConnectedSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "connected_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="ConnectedSensor",
                    class_name="ConnectedSensor",
                ),
            },
            goals={},
            connections={
                "estimatorConnection": SpecConnectionProps(
                    components=["TestEstimator", "ConnectedSensor"],
                )
            },
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value, connected_value):
            return local_value + connected_value
        return f_func, [], []

class ConnectedSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"connected_value": (1,)}
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert parser.diagnostics == []


def test_aicon_reader_estimator_f_func_accepts_dt_parameter():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value, dt):
            return local_value + dt
        return f_func, [], []
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert parser.diagnostics == []


def test_aicon_reader_estimator_f_func_rejects_unknown_differentiable_output_name():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value):
            return (local_value,), (local_value,)
        return f_func, ["missing_output"], ["local_value"]
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E109"]
    assert (
        parser.diagnostics[0].message
        == "f_func differentiable output 'missing_output' in estimator "
        "'TestEstimator' is not defined as local quantity in spec. Available "
        "local quantities: local_value."
    )


def test_aicon_reader_estimator_f_func_rejects_unknown_non_diff_output_name():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value):
            return (local_value,), (local_value,)
        return f_func, ["local_value"], ["missing_output"]
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E109"]
    assert (
        parser.diagnostics[0].message
        == "f_func non-differentiable/state output 'missing_output' in estimator "
        "'TestEstimator' is not defined as local quantity in spec. Available "
        "local quantities: local_value."
    )


def test_aicon_reader_estimator_f_func_accepts_different_output_signatures():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "local_value": SpecComponentQuantityGeneric(shape=(1,)),
                        "auxiliary_state": SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class TestEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"local_value": (1,), "auxiliary_state": (1,)}

    def define_estimation_function_f(self):
        def f_func(local_value, auxiliary_state):
            return (local_value,), (local_value, auxiliary_state)
        return f_func, ["local_value"], ["local_value", "auxiliary_state"]
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///estimators.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert parser.diagnostics == []


def test_aicon_reader_component_rejects_invalid_connection_usage():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "KnownSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityGeneric(shape=(1, 4))},
                    goals=[],
                    instance_name="KnownSensor",
                    class_name="KnownSensor",
                ),
                "OtherEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={"estimate": SpecComponentQuantityGeneric(shape=(1, 4))},
                    goals=[],
                    instance_name="OtherEstimator",
                    class_name="OtherEstimator",
                ),
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["KnownSensor", "OtherEstimator"],
                ),
            },
        )
    )

    source = """
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}

    def update(self):
        self.connections["unknownConnection"]
        self.connections["knownConnection"].connected_quantities["missing_quantity"]
        self.connections["knownConnection"].connected_quantities["estimate"]
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///sensors.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E104", "E105"]


def test_aicon_reader_component_rejects_invalid_local_quantity_usage():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "KnownSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityGeneric(shape=(1, 4))},
                    goals=[],
                    instance_name="KnownSensor",
                    class_name="KnownSensor",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}

    def update(self):
        self.quantities["missing_quantity"]
        self.quantities["value"]
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///sensors.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert [diagnostic.code for diagnostic in parser.diagnostics] == ["E107"]


def test_aicon_reader_resolves_component_type_through_inheritance_chain():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "SensorBase": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_base": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="SensorBase",
                    class_name="SensorBase",
                ),
                "ConcreteSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="ConcreteSensor",
                    class_name="ConcreteSensor",
                ),
                "ActionBase": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={
                        "action_base": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="ActionBase",
                    class_name="ActionBase",
                ),
                "ConcreteAction": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={
                        "action_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="ConcreteAction",
                    class_name="ConcreteAction",
                ),
                "EstimatorBase": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "estimator_base": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="EstimatorBase",
                    class_name="EstimatorBase",
                ),
                "ConcreteEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "estimate_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="ConcreteEstimator",
                    class_name="ConcreteEstimator",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = """
class SensorBase(SensorComponent):
    cls_quantity_names_and_shapes = {"sensor_base": (1,)}

class ConcreteSensor(SensorBase):
    cls_quantity_names_and_shapes = {"sensor_value": (1,)}

class ActionBase(ActionComponent):
    cls_quantity_names_and_shapes = {"action_base": (1,)}

class ConcreteAction(ActionBase):
    cls_quantity_names_and_shapes = {"action_value": (1,)}

class EstimatorBase(EstimationComponent):
    cls_quantity_names_and_shapes = {"estimator_base": (1,)}

class ConcreteEstimator(EstimatorBase):
    cls_quantity_names_and_shapes = {"estimate_value": (1,)}
""".strip()
    parser = AICONComponentParser(spec, document_uri="file:///inherited_components.py")
    MetadataWrapper(cst.parse_module(source)).visit(parser)

    assert parser.diagnostics == []
    assert (
        parser.components["ConcreteSensor"].component_type == SpecComponentTypes.Sensor
    )
    assert (
        parser.components["ConcreteAction"].component_type == SpecComponentTypes.Action
    )
    assert (
        parser.components["ConcreteEstimator"].component_type
        == SpecComponentTypes.Estimator
    )


def test_aicon_reader_resolves_component_type_from_imported_runtime_base(
    tmp_path: Path,
) -> None:
    package_name = "runtime_sensor_pkg_for_reader_test"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "camera_sensors.py").write_text(
        """
class SensorComponent:
    pass

class ROSCameraRGBDSensor(SensorComponent):
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "SomeComponent": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="SomeComponent",
                    class_name="SomeComponent",
                ),
            },
            goals={},
            connections={},
        )
    )

    source = f"""
from {package_name}.camera_sensors import ROSCameraRGBDSensor

class SomeComponent(ROSCameraRGBDSensor):
    cls_quantity_names_and_shapes = {{"sensor_value": (1,)}}
""".strip()
    try:
        parser = AICONComponentParser(spec, document_uri="file:///components.py")
        MetadataWrapper(cst.parse_module(source)).visit(parser)

        assert parser.diagnostics == []
        assert (
            parser.components["SomeComponent"].component_type
            == SpecComponentTypes.Sensor
        )
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        sys.modules.pop(f"{package_name}.camera_sensors", None)
        sys.modules.pop(package_name, None)


def test_aicon_reader_resolves_component_type_from_project_local_import(
    tmp_path: Path,
) -> None:
    file_a = tmp_path / "FileA.py"
    file_b = tmp_path / "FileB.py"
    file_a.write_text(
        """
class SensorComponent:
    pass

class DerivedSensor(SensorComponent):
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )

    source = """
from FileA import DerivedSensor

class mySensor(DerivedSensor):
    cls_quantity_names_and_shapes = {"sensor_value": (1,)}
""".strip()
    file_b.write_text(f"{source}\n", encoding="utf-8")

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "mySensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="mySensor",
                    class_name="mySensor",
                ),
            },
            goals={},
            connections={},
        )
    )

    try:
        parser = AICONComponentParser(
            spec,
            document_uri=file_b.as_uri(),
            project_root=tmp_path,
        )
        MetadataWrapper(cst.parse_module(source)).visit(parser)

        assert parser.diagnostics == []
        assert parser.components["mySensor"].component_type == SpecComponentTypes.Sensor
    finally:
        sys.modules.pop("FileA", None)


def test_aicon_reader_resolves_component_type_from_relative_import(
    tmp_path: Path,
) -> None:
    package_name = "relative_sensor_pkg_for_reader_test"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "file_a.py").write_text(
        """
class SensorComponent:
    pass

class DerivedSensor(SensorComponent):
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    file_b = package_dir / "file_b.py"
    source = """
from .file_a import DerivedSensor

class MySensor(DerivedSensor):
    cls_quantity_names_and_shapes = {"sensor_value": (1,)}
""".strip()
    file_b.write_text(f"{source}\n", encoding="utf-8")

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "MySensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="MySensor",
                    class_name="MySensor",
                ),
            },
            goals={},
            connections={},
        )
    )

    try:
        parser = AICONComponentParser(
            spec,
            document_uri=file_b.as_uri(),
            project_root=tmp_path,
        )
        MetadataWrapper(cst.parse_module(source)).visit(parser)

        assert parser.diagnostics == []
        assert parser.components["MySensor"].component_type == SpecComponentTypes.Sensor
    finally:
        sys.modules.pop(f"{package_name}.file_a", None)
        sys.modules.pop(package_name, None)


def test_aicon_reader_resolves_component_type_when_runtime_import_fails(
    tmp_path: Path,
) -> None:
    package_name = "failing_runtime_import_pkg_for_reader_test"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "file_a.py").write_text(
        """
import dependency_that_does_not_exist_for_aiconls_tests

class SensorComponent:
    pass

class DerivedSensor(SensorComponent):
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    file_b = package_dir / "file_b.py"
    source = f"""
from {package_name}.file_a import DerivedSensor

class MySensor(DerivedSensor):
    cls_quantity_names_and_shapes = {{"sensor_value": (1,)}}
""".strip()
    file_b.write_text(f"{source}\n", encoding="utf-8")

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "MySensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "sensor_value": SpecComponentQuantityGeneric(shape=(1,))
                    },
                    goals=[],
                    instance_name="MySensor",
                    class_name="MySensor",
                ),
            },
            goals={},
            connections={},
        )
    )

    try:
        parser = AICONComponentParser(
            spec,
            document_uri=file_b.as_uri(),
            project_root=tmp_path,
        )
        MetadataWrapper(cst.parse_module(source)).visit(parser)

        assert parser.diagnostics == []
        assert parser.components["MySensor"].component_type == SpecComponentTypes.Sensor
    finally:
        sys.modules.pop(f"{package_name}.file_a", None)
        sys.modules.pop(package_name, None)
