import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
from urllib.parse import unquote, urlparse

import torch
from lsprotocol.types import Range

from dynamic.randomization import sample_allowed_tensor_values
from reader.aicon_files.packages import module_name_and_import_root
from reader.aicon_files.reader_types import (
    ParsedComponent,
    ParsedComponents,
    ParsedConnection,
    ParsedConnections,
    ParsedGoal,
    ParsedGoals,
)
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import SpecComponentTypes, is_dynamic_shape, shapes_match

if TYPE_CHECKING:
    from aicon_base.components.action_components import ActionComponent
    from aicon_base.components.estimation_components import EstimationComponent
    from aicon_base.components.goals import Goal
    from aicon_base.components.sensor_components import SensorComponent
    from aicon_base.connections.connections import ActiveInterconnection

    ComponentInstance: TypeAlias = (
        SensorComponent | ActionComponent | EstimationComponent
    )
    GoalInstance: TypeAlias = Goal
    ConnectionInstance: TypeAlias = ActiveInterconnection
else:
    ComponentInstance: TypeAlias = object
    GoalInstance: TypeAlias = object
    ConnectionInstance: TypeAlias = object


@dataclass(frozen=True)
class AICONRuntimeTypes:
    sensor_component: type[object]
    action_component: type[object]
    estimation_component: type[object]
    goal: type[object]
    active_interconnection: type[object]


@dataclass(frozen=True)
class InstantiatedComponent:
    instance_name: str
    class_name: str
    parsed_component: ParsedComponent
    instance: ComponentInstance
    goal_instances: tuple[str, ...]
    connection_instances: tuple[str, ...]


@dataclass(frozen=True)
class InstantiatedGoal:
    instance_name: str
    class_name: str
    parsed_goal: ParsedGoal
    instance: GoalInstance


@dataclass(frozen=True)
class InstantiatedConnection:
    instance_name: str
    class_name: str
    parsed_connection: ParsedConnection
    instance: ConnectionInstance
    component_instances: tuple[str, ...]


@dataclass(frozen=True)
class InstantiationFailure:
    kind: Literal["component", "goal", "connection"]
    instance_name: str
    class_name: str
    document_uri: str
    position: Range
    reason: str


@dataclass(frozen=True)
class DynamicNetworkInstantiation:
    components: dict[str, InstantiatedComponent]
    goals: dict[str, InstantiatedGoal]
    connections: dict[str, InstantiatedConnection]
    missing_component_instances: tuple[str, ...]
    missing_goal_instances: tuple[str, ...]
    missing_connection_instances: tuple[str, ...]
    # entities whose class could not be loaded or constructed; they are skipped
    instantiation_failures: tuple[InstantiationFailure, ...] = ()


def _resolve_goal_instance_names(spec: AICON_SPEC, goal_reference: str) -> list[str]:
    if goal_reference in spec.goals:
        return [goal_reference]
    return spec.get_goal_instances_by_class_name(goal_reference)


def _uri_to_file_path(document_uri: str) -> Path:
    uri = urlparse(document_uri)
    if uri.scheme != "file":
        raise RuntimeError(
            f"Dynamic linting only supports file:// URIs, got: {document_uri}"
        )
    return Path(unquote(uri.path)).resolve()


def _load_module_from_file(file_path: Path, module_name: str) -> ModuleType:
    # Inside a package, import the file under its real name, so that its relative imports (also of parent packages,
    # e.g. `from ..util import ...`) resolve.
    package_module = module_name_and_import_root(file_path)
    if package_module is not None:
        dotted_name, import_root = package_module
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
        return importlib.import_module(dotted_name)
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load module spec from file: {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _load_class(
    module_cache: dict[Path, ModuleType], class_name: str, uri: str
) -> Callable[..., object]:
    file_path = _uri_to_file_path(uri)
    module = module_cache.get(file_path)
    if module is None:
        module_name = f"_aiconls_dynamic_{abs(hash(file_path))}"
        module = _load_module_from_file(file_path, module_name)
        module_cache[file_path] = module
    loaded_class = getattr(module, class_name, None)
    if loaded_class is None:
        raise RuntimeError(f"Could not import class '{class_name}' from '{file_path}'")
    if not isinstance(loaded_class, type):
        raise TypeError(
            f"Imported symbol '{class_name}' from '{file_path}' is not a class"
        )
    return cast(Callable[..., object], loaded_class)


def _import_aicon_module(preferred_module: str, fallback_module: str) -> ModuleType:
    try:
        return importlib.import_module(preferred_module)
    except ModuleNotFoundError:
        try:
            return importlib.import_module(fallback_module)
        except ModuleNotFoundError as fallback_error:
            raise RuntimeError(
                "Could not import AICON runtime module. Tried "
                f"'{preferred_module}' and '{fallback_module}'."
            ) from fallback_error


def _load_aicon_runtime_types() -> AICONRuntimeTypes:
    sensor_module = _import_aicon_module(
        "aicon.aicon_base.components.sensor_components",
        "aicon_base.components.sensor_components",
    )
    action_module = _import_aicon_module(
        "aicon.aicon_base.components.action_components",
        "aicon_base.components.action_components",
    )
    estimation_module = _import_aicon_module(
        "aicon.aicon_base.components.estimation_components",
        "aicon_base.components.estimation_components",
    )
    goals_module = _import_aicon_module(
        "aicon.aicon_base.components.goals",
        "aicon_base.components.goals",
    )
    connection_module = _import_aicon_module(
        "aicon.aicon_base.connections.connections",
        "aicon_base.connections.connections",
    )
    return AICONRuntimeTypes(
        sensor_component=cast(type[object], sensor_module.SensorComponent),
        action_component=cast(type[object], action_module.ActionComponent),
        estimation_component=cast(type[object], estimation_module.EstimationComponent),
        goal=cast(type[object], goals_module.Goal),
        active_interconnection=cast(
            type[object], connection_module.ActiveInterconnection
        ),
    )


def _instantiate_class(
    class_type: Callable[..., object],
    *,
    context_name: str,
    available_kwargs: dict[str, object],
) -> object:
    signature = inspect.signature(class_type)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs_to_use: dict[str, object]
    if accepts_kwargs:
        kwargs_to_use = dict(available_kwargs)
    else:
        kwargs_to_use = {}
        for key, value in available_kwargs.items():
            parameter = parameters.get(key)
            if parameter is None:
                continue
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                kwargs_to_use[key] = value

    missing_required_parameters = [
        parameter_name
        for parameter_name, parameter in parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Parameter.empty
        and parameter_name not in kwargs_to_use
    ]
    if not missing_required_parameters:
        return class_type(**kwargs_to_use)

    # Parameters only the experiment can provide (e.g. a simulation environment) get placeholders: the dynamic checks
    # randomise the quantities, so such parameters are mostly not needed. If the class needs them anyway, report them.
    for parameter_name in missing_required_parameters:
        kwargs_to_use[parameter_name] = _placeholder_argument(
            parameters[parameter_name]
        )
    try:
        return class_type(**kwargs_to_use)
    except Exception as error:
        raise RuntimeError(
            f"Could not instantiate {context_name}. Missing constructor parameters: "
            f"{', '.join(missing_required_parameters)} (placeholders failed: {type(error).__name__}: {error})."
        ) from error


def _placeholder_argument(parameter: inspect.Parameter) -> object:
    """A stand-in for a required constructor argument: zero for numbers, False for flags, None otherwise."""
    annotation = parameter.annotation
    if isinstance(annotation, str):
        annotation = {"float": float, "int": int, "bool": bool}.get(
            annotation, annotation
        )
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    return None


def _ensure_component_class_quantity_shapes(class_type: Callable[..., object]) -> None:
    class_object = cast(type[Any], class_type)
    class_level_shapes = getattr(class_object, "cls_quantity_names_and_shapes", None)
    local_shapes = getattr(class_object, "local_quantity_names_and_shapes", None)
    if not isinstance(class_level_shapes, dict) or not isinstance(local_shapes, dict):
        return
    if class_level_shapes:
        return
    normalized_local_shapes = cast(dict[str, tuple[int, ...]], local_shapes)
    class_object.cls_quantity_names_and_shapes = normalized_local_shapes.copy()


def _constant_builder(
    instance: object,
) -> Callable[[object | None, object | None, bool], object]:
    def _builder(
        _device: object | None,
        _dtype: object | None,
        _mockbuild: bool,
    ) -> object:
        return instance

    return _builder


def _load_run_component_sequence() -> Callable[..., object]:
    middleware_module = _import_aicon_module(
        "aicon.aicon_base.middleware.python_sequential",
        "aicon_base.middleware.python_sequential",
    )
    run_component_sequence = getattr(middleware_module, "run_component_sequence", None)
    if not callable(run_component_sequence):
        raise TypeError(
            "Could not import AICON middleware function 'run_component_sequence'."
        )
    return run_component_sequence


def _bind_random_sensor_measurements(
    *, sensor_instance: object, quantity_shapes: dict[str, tuple[int, ...]]
) -> None:
    sensor = cast(Any, sensor_instance)
    if not hasattr(sensor, "_aiconls_random_sensor_update_count"):
        sensor._aiconls_random_sensor_update_count = 0

    def _obtain_measurements(self: Any) -> bool:
        current_count = cast(
            int, getattr(self, "_aiconls_random_sensor_update_count", 0)
        )
        self._aiconls_random_sensor_update_count = current_count + 1
        for quantity_name, expected_shape in quantity_shapes.items():
            try:
                quantity_value = self.quantities[quantity_name]
            except KeyError:
                quantity_value = torch.zeros(
                    expected_shape,
                    dtype=getattr(self, "dtype", None),
                    device=getattr(self, "device", None),
                )
            if not isinstance(quantity_value, torch.Tensor):
                quantity_value = torch.zeros(
                    expected_shape,
                    dtype=getattr(self, "dtype", None),
                    device=getattr(self, "device", None),
                )
            self.quantities[quantity_name] = sample_allowed_tensor_values(
                quantity_name=quantity_name,
                tensor=quantity_value,
            )
        return True

    sensor.obtain_measurements = MethodType(_obtain_measurements, sensor)


def _enable_random_sensor_values(
    spec: AICON_SPEC, instantiation: DynamicNetworkInstantiation
) -> None:
    for instance_name, component_props in spec.components.items():
        if component_props.component_type != SpecComponentTypes.Sensor:
            continue
        instantiated_component = instantiation.components.get(instance_name)
        if instantiated_component is None:
            continue
        # dynamic quantities keep the values the sensor creates itself
        quantity_shapes = {
            quantity_name: tuple(quantity_props.shape)
            for quantity_name, quantity_props in component_props.quantities.items()
            if not is_dynamic_shape(quantity_props.shape)
        }
        _bind_random_sensor_measurements(
            sensor_instance=instantiated_component.instance,
            quantity_shapes=quantity_shapes,
        )


def _extract_tensor_shape(value: object) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    shape = getattr(value, "shape", None)
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    return None


def get_component_quantity_shape_mismatch_details(
    spec: AICON_SPEC, instantiation: DynamicNetworkInstantiation
) -> dict[str, tuple[str, ...]]:
    mismatches_by_component: dict[str, list[str]] = {}
    for instance_name, component_props in spec.components.items():
        instantiated_component = instantiation.components.get(instance_name)
        if instantiated_component is None:
            continue
        quantities = getattr(instantiated_component.instance, "quantities", None)
        if quantities is None:
            mismatches_by_component.setdefault(instance_name, []).append(
                "missing runtime quantity container on component instance"
            )
            continue
        for quantity_name, quantity_props in component_props.quantities.items():
            expected_shape = tuple(quantity_props.shape)
            try:
                quantity_value = quantities[quantity_name]
            except KeyError:
                mismatches_by_component.setdefault(instance_name, []).append(
                    f"{quantity_name}: missing runtime quantity value"
                )
                continue
            actual_shape = _extract_tensor_shape(quantity_value)
            if actual_shape is None:
                mismatches_by_component.setdefault(instance_name, []).append(
                    f"{quantity_name}: quantity value has no tensor-like shape"
                )
                continue
            if not shapes_match(actual_shape, expected_shape):
                mismatches_by_component.setdefault(instance_name, []).append(
                    f"{quantity_name}: expected shape {expected_shape}, got {actual_shape}"
                )
    return {
        component_instance_name: tuple(details)
        for component_instance_name, details in mismatches_by_component.items()
    }


def get_component_quantity_shape_mismatches(
    spec: AICON_SPEC, instantiation: DynamicNetworkInstantiation
) -> tuple[str, ...]:
    mismatches: list[str] = []
    mismatch_details = get_component_quantity_shape_mismatch_details(
        spec, instantiation
    )
    for instance_name, component_mismatches in mismatch_details.items():
        mismatches.extend(
            f"{instance_name}.{mismatch_detail}"
            for mismatch_detail in component_mismatches
        )
    return tuple(mismatches)


def assert_component_quantity_shapes_match_spec(
    spec: AICON_SPEC, instantiation: DynamicNetworkInstantiation
) -> None:
    mismatches = get_component_quantity_shape_mismatches(spec, instantiation)
    if mismatches:
        raise AssertionError(
            "Runtime quantity shapes do not match spec: " + "; ".join(mismatches)
        )


def run_network_pass_with_random_sensor_values(
    spec: AICON_SPEC,
    instantiation: DynamicNetworkInstantiation,
    *,
    new_time: torch.Tensor | None = None,
    passive_mode: bool = False,
) -> None:
    _enable_random_sensor_values(spec, instantiation)
    run_component_sequence = _load_run_component_sequence()
    runtime_components = {
        instance_name: instantiated_component.instance
        for instance_name, instantiated_component in instantiation.components.items()
    }
    timestamp = new_time if new_time is not None else torch.zeros(1)
    run_component_sequence(
        runtime_components,
        timestamp,
        passive_mode=passive_mode,
    )


def _cast_goal_instance(
    instance: object, runtime_types: AICONRuntimeTypes
) -> GoalInstance:
    if not isinstance(instance, runtime_types.goal):
        raise TypeError(
            f"Instantiated object has invalid goal type: {instance.__class__.__name__}"
        )
    return cast(GoalInstance, instance)


def _cast_connection_instance(
    instance: object, runtime_types: AICONRuntimeTypes
) -> ConnectionInstance:
    if not isinstance(instance, runtime_types.active_interconnection):
        raise TypeError(
            "Instantiated object has invalid connection type: "
            f"{instance.__class__.__name__}"
        )
    return cast(ConnectionInstance, instance)


def _cast_component_instance(
    component_type: SpecComponentTypes,
    instance: object,
    runtime_types: AICONRuntimeTypes,
) -> ComponentInstance:
    if component_type == SpecComponentTypes.Sensor:
        expected_type = runtime_types.sensor_component
    elif component_type == SpecComponentTypes.Action:
        expected_type = runtime_types.action_component
    elif component_type == SpecComponentTypes.Estimator:
        expected_type = runtime_types.estimation_component
    else:
        raise RuntimeError(
            f"Unsupported component type for dynamic instantiation: {component_type}"
        )
    if not isinstance(instance, expected_type):
        raise TypeError(
            "Instantiated object has invalid component type for "
            f"{component_type.name}: {instance.__class__.__name__}"
        )
    return cast(ComponentInstance, instance)


def spec_torch_dtype(
    spec: AICON_SPEC, component_instance_names: list[str]
) -> torch.dtype | None:
    """Return the spec dtype shared by all given components, or None for the default."""
    dtype_names = {
        component_props.dtype
        for name in component_instance_names
        if (component_props := spec.components.get(name)) is not None
    }
    if len(dtype_names) != 1:
        return None
    dtype_name = next(iter(dtype_names))
    if dtype_name is None:
        return None
    return cast(torch.dtype, getattr(torch, dtype_name))


def instantiate_network_from_spec(
    spec: AICON_SPEC,
    parsed_components: ParsedComponents,
    parsed_goals: ParsedGoals,
    parsed_connections: ParsedConnections,
) -> DynamicNetworkInstantiation:
    module_cache: dict[Path, ModuleType] = {}
    runtime_types = _load_aicon_runtime_types()
    instantiation_failures: list[InstantiationFailure] = []

    goal_objects: dict[str, object] = {}
    instantiated_goals: dict[str, InstantiatedGoal] = {}
    missing_goal_instances: list[str] = []
    for instance_name, goal_props in spec.goals.items():
        parsed_goal = parsed_goals.get(goal_props.class_name)
        if parsed_goal is None:
            missing_goal_instances.append(instance_name)
            continue
        try:
            goal_class = _load_class(
                module_cache, goal_props.class_name, parsed_goal.document_uri
            )
            goal_instance = _instantiate_class(
                goal_class,
                context_name=f"goal '{instance_name}' ({goal_props.class_name})",
                available_kwargs={
                    "name": instance_name,
                    "is_active": True,
                    "dtype": spec_torch_dtype(
                        spec,
                        [
                            component_name
                            for component_name, component_props in spec.components.items()
                            if instance_name in component_props.goals
                            or goal_props.class_name in component_props.goals
                        ],
                    ),
                    "device": None,
                    "mockbuild": False,
                },
            )
            typed_goal_instance = _cast_goal_instance(goal_instance, runtime_types)
        except Exception as error:  # noqa: BLE001 - user code may raise anything
            instantiation_failures.append(
                InstantiationFailure(
                    kind="goal",
                    instance_name=instance_name,
                    class_name=goal_props.class_name,
                    document_uri=parsed_goal.document_uri,
                    position=parsed_goal.position,
                    reason=f"{error.__class__.__name__}: {error}",
                )
            )
            missing_goal_instances.append(instance_name)
            continue
        goal_objects[instance_name] = typed_goal_instance
        instantiated_goals[instance_name] = InstantiatedGoal(
            instance_name=instance_name,
            class_name=goal_props.class_name,
            parsed_goal=parsed_goal,
            instance=typed_goal_instance,
        )

    connection_instances: dict[str, object] = {}
    instantiated_connections: dict[str, InstantiatedConnection] = {}
    missing_connection_instances: list[str] = []
    for instance_name, connection_props in spec.connections.items():
        parsed_connection = parsed_connections.get(connection_props.class_name)
        if parsed_connection is None:
            missing_connection_instances.append(instance_name)
            continue
        try:
            connection_class = _load_class(
                module_cache,
                connection_props.class_name,
                parsed_connection.document_uri,
            )
            connection_instance = _instantiate_class(
                connection_class,
                context_name=f"connection '{instance_name}' ({connection_props.class_name})",
                available_kwargs={
                    "name": instance_name,
                    "dtype": spec_torch_dtype(spec, connection_props.components),
                    "device": None,
                    "mockbuild": False,
                },
            )
            typed_connection_instance = _cast_connection_instance(
                connection_instance, runtime_types
            )
        except Exception as error:  # noqa: BLE001 - user code may raise anything
            instantiation_failures.append(
                InstantiationFailure(
                    kind="connection",
                    instance_name=instance_name,
                    class_name=connection_props.class_name,
                    document_uri=parsed_connection.document_uri,
                    position=parsed_connection.position,
                    reason=f"{error.__class__.__name__}: {error}",
                )
            )
            missing_connection_instances.append(instance_name)
            continue
        connection_instances[instance_name] = typed_connection_instance
        instantiated_connections[instance_name] = InstantiatedConnection(
            instance_name=instance_name,
            class_name=connection_props.class_name,
            parsed_connection=parsed_connection,
            instance=typed_connection_instance,
            component_instances=tuple(connection_props.components),
        )

    connection_instances_by_component: dict[str, list[str]] = {}
    for connection_instance_name, connection_props in spec.connections.items():
        for component_instance_name in connection_props.components:
            connection_instances_by_component.setdefault(
                component_instance_name, []
            ).append(connection_instance_name)

    instantiated_components: dict[str, InstantiatedComponent] = {}
    missing_component_instances: list[str] = []
    for instance_name, component_props in spec.components.items():
        parsed_component = parsed_components.get(component_props.class_name)
        if parsed_component is None:
            missing_component_instances.append(instance_name)
            continue

        try:
            component_class = _load_class(
                module_cache, component_props.class_name, parsed_component.document_uri
            )
            _ensure_component_class_quantity_shapes(component_class)
            resolved_goal_instances: list[str] = []
            for goal_reference in component_props.goals:
                for goal_instance_name in _resolve_goal_instance_names(
                    spec, goal_reference
                ):
                    if goal_instance_name in instantiated_goals:
                        resolved_goal_instances.append(goal_instance_name)

            goal_builders = {
                goal_instance_name: _constant_builder(goal_objects[goal_instance_name])
                for goal_instance_name in resolved_goal_instances
                if goal_instance_name in goal_objects
            }
            connection_builders = {
                connection_instance_name: _constant_builder(
                    connection_instances[connection_instance_name]
                )
                for connection_instance_name in connection_instances_by_component.get(
                    instance_name, []
                )
                if connection_instance_name in connection_instances
            }
            component_instance = _instantiate_class(
                component_class,
                context_name=f"component '{instance_name}' ({component_props.class_name})",
                available_kwargs={
                    "name": instance_name,
                    "connections": connection_builders,
                    "goals": goal_builders or None,
                    "dtype": spec_torch_dtype(spec, [instance_name]),
                    "device": None,
                    "mockbuild": False,
                },
            )
            typed_component_instance = _cast_component_instance(
                component_props.component_type,
                component_instance,
                runtime_types,
            )
        except Exception as error:  # noqa: BLE001 - user code may raise anything
            instantiation_failures.append(
                InstantiationFailure(
                    kind="component",
                    instance_name=instance_name,
                    class_name=component_props.class_name,
                    document_uri=parsed_component.document_uri,
                    position=parsed_component.position,
                    reason=f"{error.__class__.__name__}: {error}",
                )
            )
            missing_component_instances.append(instance_name)
            continue
        instantiated_components[instance_name] = InstantiatedComponent(
            instance_name=instance_name,
            class_name=component_props.class_name,
            parsed_component=parsed_component,
            instance=typed_component_instance,
            goal_instances=tuple(resolved_goal_instances),
            connection_instances=tuple(
                connection_instances_by_component.get(instance_name, [])
            ),
        )

    return DynamicNetworkInstantiation(
        components=instantiated_components,
        goals=instantiated_goals,
        connections=instantiated_connections,
        missing_component_instances=tuple(missing_component_instances),
        missing_goal_instances=tuple(missing_goal_instances),
        missing_connection_instances=tuple(missing_connection_instances),
        instantiation_failures=tuple(instantiation_failures),
    )
