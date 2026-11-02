from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lsprotocol import types

import dynamic.loader as dynamic_scanner
from dynamic.loader import (
    AICONRuntimeTypes,
    assert_component_quantity_shapes_match_spec,
    instantiate_network_from_spec,
    run_network_pass_with_random_sensor_values,
)
from reader.aicon_files.reader_types import (
    ParsedComponent,
    ParsedConnection,
    ParsedGoal,
    ParsedQuantity,
)
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
    SpecGoalProps,
)
from reader.spec.spec_types import (
    SpecComponentQuantityGeneric as SpecComponentQuantityProps,
)


@pytest.fixture(autouse=True)
def _patch_runtime_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dynamic_scanner,
        "_load_aicon_runtime_types",
        lambda: AICONRuntimeTypes(
            sensor_component=object,
            action_component=object,
            estimation_component=object,
            goal=object,
            active_interconnection=object,
        ),
    )


def _range() -> types.Range:
    zero = types.Position(line=0, character=0)
    return types.Range(start=zero, end=zero)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_instantiate_network_from_spec_imports_classes_and_creates_instances(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network.py"
    _write_file(
        module_path,
        """
class GoalImpl:
    def __init__(self, is_active=True, dtype=None, device=None, mockbuild=False):
        self.is_active = is_active


class ConnectionImpl:
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        self.name = name


class SensorImpl:
    def __init__(self, name, connections, dtype=None, device=None, mockbuild=False, sim_env_pointer=None):
        self.name = name
        self.connections = connections


class ActionImpl:
    def __init__(self, name, connections, goals=None, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
        self.goals = goals
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=["goal_instance"],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                ),
                "sensor_clone": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_clone",
                    class_name="SensorImpl",
                ),
                "action_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={"cmd": SpecComponentQuantityProps(shape=(1,))},
                    goals=["GoalImpl"],
                    instance_name="action_instance",
                    class_name="ActionImpl",
                ),
            },
            goals={
                "goal_instance": SpecGoalProps(
                    name="GoalImpl",
                    instance_name="goal_instance",
                )
            },
            connections={
                "connection_instance": SpecConnectionProps(
                    components=["sensor_instance", "action_instance"],
                    instance_name="connection_instance",
                    class_name="ConnectionImpl",
                )
            },
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=["goal_instance"],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )
    action = ParsedComponent(
        component_type=SpecComponentTypes.Action,
        quantities={"cmd": ParsedQuantity(shape=(1,))},
        goals=["GoalImpl"],
        instance_name="ActionImpl",
        class_name="ActionImpl",
        position=_range(),
        document_uri=module_uri,
    )
    goal = ParsedGoal(
        name="GoalImpl",
        position=_range(),
        document_uri=module_uri,
        goal_func_params={},
    )
    connection = ParsedConnection(
        name="ConnectionImpl",
        position=_range(),
        document_uri=module_uri,
        quantities={},
        quantities_pos=None,
        connection_func_params={},
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor, "ActionImpl": action},
        parsed_goals={"GoalImpl": goal},
        parsed_connections={"ConnectionImpl": connection},
    )

    assert instantiation.missing_component_instances == ()
    assert instantiation.missing_goal_instances == ()
    assert instantiation.missing_connection_instances == ()
    assert (
        instantiation.components["sensor_instance"].instance.__class__.__name__
        == "SensorImpl"
    )
    assert (
        instantiation.components["sensor_clone"].instance.__class__.__name__
        == "SensorImpl"
    )
    assert (
        instantiation.components["action_instance"].instance.__class__.__name__
        == "ActionImpl"
    )
    assert (
        instantiation.goals["goal_instance"].instance.__class__.__name__ == "GoalImpl"
    )
    assert (
        instantiation.connections["connection_instance"].instance.__class__.__name__
        == "ConnectionImpl"
    )
    assert instantiation.components["action_instance"].goal_instances == (
        "goal_instance",
    )
    assert instantiation.components["sensor_instance"].connection_instances == (
        "connection_instance",
    )

    action_instance = cast(Any, instantiation.components["action_instance"].instance)
    action_goals = action_instance.goals
    assert isinstance(action_goals, dict)
    assert (
        action_goals["goal_instance"](None, None, True)
        is instantiation.goals["goal_instance"].instance
    )

    sensor_instance = cast(Any, instantiation.components["sensor_instance"].instance)
    sensor_connections = sensor_instance.connections
    assert isinstance(sensor_connections, dict)
    assert (
        sensor_connections["connection_instance"](None, None, True)
        is instantiation.connections["connection_instance"].instance
    )


def test_instantiate_network_from_spec_tracks_missing_instances() -> None:
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="MissingSensor",
                )
            },
            goals={
                "goal_instance": SpecGoalProps(
                    name="MissingGoal",
                    instance_name="goal_instance",
                )
            },
            connections={
                "connection_instance": SpecConnectionProps(
                    components=["sensor_instance"],
                    instance_name="connection_instance",
                    class_name="MissingConnection",
                )
            },
        )
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={},
        parsed_goals={},
        parsed_connections={},
    )

    assert instantiation.components == {}
    assert instantiation.goals == {}
    assert instantiation.connections == {}
    assert instantiation.missing_component_instances == ("sensor_instance",)
    assert instantiation.missing_goal_instances == ("goal_instance",)
    assert instantiation.missing_connection_instances == ("connection_instance",)


def test_instantiate_network_from_spec_maps_local_shapes_to_class_shapes(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_shapes.py"
    _write_file(
        module_path,
        """
class ConnectionImpl:
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        self.name = name


class ActionLikeComponent:
    cls_quantity_names_and_shapes = {}

    def __init__(self, name, connections, goals=None, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.local_quantity_names_and_shapes = self.cls_quantity_names_and_shapes


class ActionImpl(ActionLikeComponent):
    local_quantity_names_and_shapes = {"action_velo_ee": (3,)}
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "action_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={
                        "action_velo_ee": SpecComponentQuantityProps(shape=(3,))
                    },
                    goals=[],
                    instance_name="action_instance",
                    class_name="ActionImpl",
                )
            },
            goals={},
            connections={
                "connection_instance": SpecConnectionProps(
                    components=["action_instance"],
                    instance_name="connection_instance",
                    class_name="ConnectionImpl",
                )
            },
        )
    )

    module_uri = module_path.resolve().as_uri()
    action = ParsedComponent(
        component_type=SpecComponentTypes.Action,
        quantities={"action_velo_ee": ParsedQuantity(shape=(3,))},
        goals=[],
        instance_name="ActionImpl",
        class_name="ActionImpl",
        position=_range(),
        document_uri=module_uri,
    )
    connection = ParsedConnection(
        name="ConnectionImpl",
        position=_range(),
        document_uri=module_uri,
        quantities={},
        quantities_pos=None,
        connection_func_params={},
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"ActionImpl": action},
        parsed_goals={},
        parsed_connections={"ConnectionImpl": connection},
    )

    action_instance = cast(Any, instantiation.components["action_instance"].instance)
    assert action_instance.local_quantity_names_and_shapes == {"action_velo_ee": (3,)}


def test_dynamic_pass_uses_random_sensor_values_and_preserves_spec_shapes(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_random_sensor.py"
    _write_file(
        module_path,
        """
import torch
from aicon_base.components.action_components import ActionComponent
from aicon_base.components.sensor_components import SensorComponent
from aicon_base.connections.connections import ActiveInterconnection


class ConnectionImpl(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {"sensor_value": (2,), "action_value": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(sensor_value, action_value):
            return sensor_value, action_value

        return connection_func


class SensorImpl(SensorComponent):
    cls_quantity_names_and_shapes = {"sensor_value": (2,)}

    def __init__(self, name, connections, dtype=None, device=None, mockbuild=False, sim_env_pointer=None):
        super().__init__(name, connections, dtype=dtype, device=device, mockbuild=mockbuild)
        self.non_random_obtain_calls = 0

    def obtain_measurements(self):
        self.non_random_obtain_calls += 1
        self.quantities["sensor_value"] = torch.zeros((1,), dtype=self.dtype, device=self.device)
        return True


class ActionImpl(ActionComponent):
    cls_quantity_names_and_shapes = {"action_value": (3,)}

    def _start(self):
        return None

    def _stop(self):
        return None

    def initialize_quantities(self):
        self.quantities["action_value"] = torch.zeros((3,), dtype=self.dtype, device=self.device)
        return True

    def determine_new_actions(self):
        sensor_value = self.connections["connection_instance"].connected_quantities["sensor_value"]
        action_value = torch.zeros((3,), dtype=self.dtype, device=self.device)
        action_value[:2] = sensor_value
        self.quantities["action_value"] = action_value

    def send_action_values(self):
        return None
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"sensor_value": SpecComponentQuantityProps(shape=(2,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                ),
                "action_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={"action_value": SpecComponentQuantityProps(shape=(3,))},
                    goals=[],
                    instance_name="action_instance",
                    class_name="ActionImpl",
                ),
            },
            goals={},
            connections={
                "connection_instance": SpecConnectionProps(
                    components=["sensor_instance", "action_instance"],
                    instance_name="connection_instance",
                    class_name="ConnectionImpl",
                )
            },
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"sensor_value": ParsedQuantity(shape=(2,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )
    action = ParsedComponent(
        component_type=SpecComponentTypes.Action,
        quantities={"action_value": ParsedQuantity(shape=(3,))},
        goals=[],
        instance_name="ActionImpl",
        class_name="ActionImpl",
        position=_range(),
        document_uri=module_uri,
    )
    connection = ParsedConnection(
        name="ConnectionImpl",
        position=_range(),
        document_uri=module_uri,
        quantities={},
        quantities_pos=None,
        connection_func_params={"sensor_value": _range(), "action_value": _range()},
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor, "ActionImpl": action},
        parsed_goals={},
        parsed_connections={"ConnectionImpl": connection},
    )

    assert_component_quantity_shapes_match_spec(spec, instantiation)

    run_network_pass_with_random_sensor_values(
        spec,
        instantiation,
        new_time=torch.tensor([0.0]),
    )
    run_network_pass_with_random_sensor_values(
        spec,
        instantiation,
        new_time=torch.tensor([1.0]),
    )

    assert_component_quantity_shapes_match_spec(spec, instantiation)

    sensor_instance = cast(Any, instantiation.components["sensor_instance"].instance)
    assert sensor_instance.non_random_obtain_calls == 0
    assert sensor_instance._aiconls_random_sensor_update_count == 2


def test_instantiate_network_from_spec_skips_class_with_missing_constructor_parameter(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_missing_param.py"
    _write_file(
        module_path,
        """
class SensorImpl:
    def __init__(self, name, connections, required_flag, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
        # uses the argument, so a placeholder (None) does not work either
        self.required_flag = required_flag.upper()
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                )
            },
            goals={},
            connections={},
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor},
        parsed_goals={},
        parsed_connections={},
    )

    assert instantiation.components == {}
    assert instantiation.missing_component_instances == ("sensor_instance",)


def test_instantiate_network_from_spec_uses_placeholders_for_missing_constructor_parameters(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_placeholder_param.py"
    _write_file(
        module_path,
        """
class SensorImpl:
    def __init__(self, name, connections, open_value: float, env, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
        self.open_value = open_value
        self.env = env
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                )
            },
            goals={},
            connections={},
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor},
        parsed_goals={},
        parsed_connections={},
    )

    instance = instantiation.components["sensor_instance"].instance
    assert instance.open_value == 0.0 and instance.env is None
    assert instantiation.missing_component_instances == ()
    assert instantiation.instantiation_failures == ()


def test_assert_component_quantity_shapes_match_spec_raises_on_shape_mismatch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_shape_mismatch.py"
    _write_file(
        module_path,
        """
import torch


class SensorImpl:
    def __init__(self, name, connections, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
        self.quantities = {"value": torch.zeros((2,), dtype=torch.float32)}
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                )
            },
            goals={},
            connections={},
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor},
        parsed_goals={},
        parsed_connections={},
    )

    with pytest.raises(AssertionError, match="expected shape"):
        assert_component_quantity_shapes_match_spec(spec, instantiation)


def test_assert_component_quantity_shapes_match_spec_raises_on_missing_quantity(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_missing_quantity.py"
    _write_file(
        module_path,
        """
class SensorImpl:
    def __init__(self, name, connections, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
        self.quantities = {}
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                )
            },
            goals={},
            connections={},
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor},
        parsed_goals={},
        parsed_connections={},
    )

    with pytest.raises(AssertionError, match="missing runtime quantity value"):
        assert_component_quantity_shapes_match_spec(spec, instantiation)


def test_assert_component_quantity_shapes_match_spec_raises_on_missing_quantity_container(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "runtime_network_missing_container.py"
    _write_file(
        module_path,
        """
class SensorImpl:
    def __init__(self, name, connections, dtype=None, device=None, mockbuild=False):
        self.name = name
        self.connections = connections
""".strip(),
    )

    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"value": SpecComponentQuantityProps(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorImpl",
                )
            },
            goals={},
            connections={},
        )
    )

    module_uri = module_path.resolve().as_uri()
    sensor = ParsedComponent(
        component_type=SpecComponentTypes.Sensor,
        quantities={"value": ParsedQuantity(shape=(1,))},
        goals=[],
        instance_name="SensorImpl",
        class_name="SensorImpl",
        position=_range(),
        document_uri=module_uri,
    )

    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={"SensorImpl": sensor},
        parsed_goals={},
        parsed_connections={},
    )

    with pytest.raises(
        AssertionError,
        match="missing runtime quantity container on component instance",
    ):
        assert_component_quantity_shapes_match_spec(spec, instantiation)


def test_run_network_pass_with_random_sensor_values_surfaces_loader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AICON_SPEC(SpecFormat(components={}, goals={}, connections={}))
    instantiation = instantiate_network_from_spec(
        spec=spec,
        parsed_components={},
        parsed_goals={},
        parsed_connections={},
    )

    def _failing_loader() -> Any:
        raise TypeError(
            "Could not import AICON middleware function 'run_component_sequence'."
        )

    monkeypatch.setattr(
        dynamic_scanner, "_load_run_component_sequence", _failing_loader
    )

    with pytest.raises(TypeError, match="run_component_sequence"):
        run_network_pass_with_random_sensor_values(spec, instantiation)
