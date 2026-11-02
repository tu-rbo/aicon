import ast
import logging
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

import yaml
from lsprotocol import types
from lsprotocol.types import Diagnostic
from yaml.error import YAMLError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    DYNAMIC_DIMENSION,
    SpecComponentProps,
    SpecComponentQuantity,
    SpecComponentQuantityCategorical,
    SpecComponentQuantityGaussian,
    SpecComponentQuantityGeneric,
    SpecComponentQuantityParticle,
    SpecComponentQuantityReference,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
    SpecGoalProps,
    SpecQuantityTypes,
)

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    "bool",
    "uint8",
    "int8",
    "int16",
    "short",
    "int32",
    "int",
    "int64",
    "long",
    "float16",
    "half",
    "bfloat16",
    "float32",
    "float",
    "float64",
    "double",
    "complex64",
    "cfloat",
    "complex128",
    "cdouble",
}

SPEC_COMPONENT_TYPE_NAMES = {"Sensor", "Action", "Estimator"}


@dataclass(frozen=True)
class ParsedSpecContent:
    spec: AICON_SPEC | None
    diagnostics: list[Diagnostic]


def _node_to_range(node: Node) -> types.Range:
    return types.Range(
        start=types.Position(
            line=node.start_mark.line,
            character=node.start_mark.column,
        ),
        end=types.Position(
            line=node.end_mark.line,
            character=node.end_mark.column,
        ),
    )


def _default_range() -> types.Range:
    zero = types.Position(line=0, character=0)
    return types.Range(start=zero, end=zero)


def _safe_node_range(node: Node | None) -> types.Range:
    if node is None:
        return _default_range()
    start_mark = getattr(node, "start_mark", None)
    end_mark = getattr(node, "end_mark", None)
    start_line = getattr(start_mark, "line", None)
    start_column = getattr(start_mark, "column", None)
    end_line = getattr(end_mark, "line", None)
    end_column = getattr(end_mark, "column", None)
    if not (
        isinstance(start_line, int)
        and isinstance(start_column, int)
        and isinstance(end_line, int)
        and isinstance(end_column, int)
    ):
        return _default_range()
    return _node_to_range(node)


def _make_spec_diagnostic(
    code: ErrorCode,
    *,
    node: Node | None = None,
    details: str | None = None,
    range_override: types.Range | None = None,
) -> Diagnostic:
    diagnostic_range = range_override or _safe_node_range(node)
    diagnostic = make_diagnostic(code, range=diagnostic_range)
    if details:
        diagnostic.message = f"{diagnostic.message} {details}"
    return diagnostic


def _mapping_items(node: Node | None) -> list[tuple[Node, Node]]:
    if not isinstance(node, MappingNode):
        return []

    return list(node.value)


def _mapping_get(node: Node | None, key: str) -> Node | None:
    for key_node, value_node in _mapping_items(node):
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return value_node

    return None


def _validate_no_duplicate_keys(
    section_node: Node | None,
    error_code: ErrorCode,
    entry_name: str,
    diagnostics: list[Diagnostic],
) -> None:
    if not isinstance(section_node, MappingNode):
        return

    seen_names: set[str] = set()
    for key_node, _ in _mapping_items(section_node):
        if not isinstance(key_node, ScalarNode):
            continue

        if key_node.value in seen_names:
            diagnostics.append(
                _make_spec_diagnostic(
                    error_code,
                    node=key_node,
                    details=f"Duplicate {entry_name} '{key_node.value}' found.",
                )
            )
            continue

        seen_names.add(key_node.value)


def _parse_component_type(
    component_name_node: Node,
    component_node: Node,
    diagnostics: list[Diagnostic],
) -> SpecComponentTypes | None:
    component_type_node = _mapping_get(component_node, "component_type")
    if not isinstance(component_type_node, ScalarNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_COMPONENT_TYPE,
                node=component_name_node,
                details="Missing required scalar field 'component_type'.",
            )
        )
        return None

    # Goals are declared in the top-level 'goals' section, not as components.
    if component_type_node.value in SPEC_COMPONENT_TYPE_NAMES:
        return SpecComponentTypes[component_type_node.value]

    diagnostics.append(
        _make_spec_diagnostic(
            ErrorCode.MALFORMED_COMPONENT_TYPE,
            node=component_type_node,
            details=(
                f"Unsupported component_type '{component_type_node.value}'. Expected one of: Sensor, Action, Estimator."
            ),
        )
    )
    return None


def _parse_dtype(component_node: Node, diagnostics: list[Diagnostic]) -> str | None:
    dtype_node = _mapping_get(component_node, "dtype")
    if dtype_node is None:
        return None

    if not isinstance(dtype_node, ScalarNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_DATA_TYPE,
                node=dtype_node,
                details="Field 'dtype' must be a scalar string.",
            )
        )
        return None

    normalized_dtype = dtype_node.value.removeprefix("torch.")
    if normalized_dtype not in SUPPORTED_DTYPES:
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_DATA_TYPE,
                node=dtype_node,
                details=(
                    f"Unsupported dtype '{dtype_node.value}'. Supported dtypes: {', '.join(sorted(SUPPORTED_DTYPES))}."
                ),
            )
        )
        return None
    return normalized_dtype


def _is_positive_int(node: Node | None) -> TypeGuard[ScalarNode]:
    return isinstance(node, ScalarNode) and node.value.isdigit() and int(node.value) > 0


def _make_duplicate_quantity_diagnostic(name_node: ScalarNode) -> Diagnostic:
    return _make_spec_diagnostic(
        ErrorCode.MALFORMED_QUANTITY_DEFINITION,
        node=name_node,
        details=(
            f"Quantity '{name_node.value}' is defined more than once in this component. "
            "Names given by 'uncertainty_name' and 'particle_weights_name' are quantities as well."
        ),
    )


def _parse_shape(
    shape_node: Node, diagnostics: list[Diagnostic]
) -> tuple[int, ...] | None:
    if not isinstance(shape_node, ScalarNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                node=shape_node,
                details="Field 'shape' must be a scalar tuple-like string, e.g. '(1, 4)'.",
            )
        )
        return None

    try:
        parsed_expr = ast.parse(shape_node.value, mode="eval").body
    except SyntaxError:
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                node=shape_node,
                details=(
                    f"Could not parse shape value '{shape_node.value}'. Expected a tuple of integers, e.g. '(1, 4)'."
                ),
            )
        )
        return None

    if not isinstance(parsed_expr, ast.Tuple) or len(parsed_expr.elts) == 0:
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                node=shape_node,
                details=(
                    f"Invalid shape '{shape_node.value}'. Shape must be a non-empty tuple of integers."
                ),
            )
        )
        return None

    normalized_shape: list[int] = []
    for item in parsed_expr.elts:
        if (
            isinstance(item, ast.UnaryOp)
            and isinstance(item.op, ast.USub)
            and isinstance(item.operand, ast.Constant)
            and item.operand.value == -DYNAMIC_DIMENSION
            and not isinstance(item.operand.value, bool)
        ):
            normalized_shape.append(DYNAMIC_DIMENSION)
            continue
        if not isinstance(item, ast.Constant) or not isinstance(item.value, int):
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                    node=shape_node,
                    details=(
                        f"Invalid shape '{shape_node.value}'. All tuple entries must be "
                        f"non-negative integers or {DYNAMIC_DIMENSION} for a size only known at runtime."
                    ),
                )
            )
            return None
        if isinstance(item.value, bool):
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                    node=shape_node,
                    details=(
                        f"Invalid shape '{shape_node.value}'. Boolean values are not allowed in shape tuples."
                    ),
                )
            )
            return None
        normalized_shape.append(item.value)

    return tuple(normalized_shape)


def _parse_quantities(
    component_node: Node,
    diagnostics: list[Diagnostic],
) -> dict[str, SpecComponentQuantity]:
    quantities: dict[str, SpecComponentQuantity] = {}
    quantities_node = _mapping_get(component_node, "quantities")
    if quantities_node is None:
        return quantities
    if not isinstance(quantities_node, MappingNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_SPEC,
                node=quantities_node,
                details="Field 'quantities' must be a mapping of quantity definitions.",
            )
        )
        return quantities

    for quantity_name_node, quantity_node in _mapping_items(quantities_node):
        if not isinstance(quantity_name_node, ScalarNode):
            continue
        if not isinstance(quantity_node, MappingNode):
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                    node=quantity_name_node,
                    details=(
                        f"Quantity '{quantity_name_node.value}' must be a mapping."
                    ),
                )
            )
            continue
        if quantity_name_node.value in quantities:
            diagnostics.append(_make_duplicate_quantity_diagnostic(quantity_name_node))
            continue
        q_type = _parse_quantity_q_type(quantity_node, diagnostics)

        if q_type is None:
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                    node=quantity_name_node,
                    details=f"q_type '{quantity_name_node.value}' has to be 'Generic', 'Gaussian', 'Categorical' or 'Particle'.",
                )
            )
            continue

        match q_type:
            case SpecQuantityTypes.Generic:
                shape_node = _mapping_get(quantity_node, "shape")
                if shape_node is None:
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' is missing required field 'shape'.",
                        )
                    )
                    continue

                shape = _parse_shape(shape_node, diagnostics)
                if shape is None:
                    continue

                quantities[quantity_name_node.value] = SpecComponentQuantityGeneric(
                    shape=shape
                )
            case SpecQuantityTypes.Gaussian:
                dim_node = _mapping_get(quantity_node, "dim")

                if not _is_positive_int(dim_node):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' requires a positive integer field 'dim'.",
                        )
                    )
                    continue

                uncertainty_name = _mapping_get(quantity_node, "uncertainty_name")

                if uncertainty_name is None or not isinstance(
                    uncertainty_name, ScalarNode
                ):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' is missing required field 'uncertainty_name'.",
                        )
                    )
                    continue
                shape = (int(dim_node.value),)

                quantities[quantity_name_node.value] = SpecComponentQuantityGaussian(
                    shape=shape,
                    dim=int(dim_node.value),
                    uncertainty_name=uncertainty_name.value,
                )
                if uncertainty_name.value in quantities:
                    diagnostics.append(
                        _make_duplicate_quantity_diagnostic(uncertainty_name)
                    )
                    continue
                quantities[uncertainty_name.value] = SpecComponentQuantityReference(
                    shape=(int(dim_node.value), int(dim_node.value)),
                    referenced_quantity_name=quantity_name_node.value,
                )
            case SpecQuantityTypes.Categorical:
                dim_node = _mapping_get(quantity_node, "dim")

                if not _is_positive_int(dim_node):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' requires a positive integer field 'dim'.",
                        )
                    )
                    continue
                shape = (int(dim_node.value),)

                quantities[quantity_name_node.value] = SpecComponentQuantityCategorical(
                    shape=shape, dim=int(dim_node.value)
                )

            case SpecQuantityTypes.Particle:
                dim_node = _mapping_get(quantity_node, "dim")
                particle_count = _mapping_get(quantity_node, "particle_count")
                particle_weights_name = _mapping_get(
                    quantity_node, "particle_weights_name"
                )

                if not _is_positive_int(dim_node):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' requires a positive integer field 'dim'.",
                        )
                    )
                    continue

                if not _is_positive_int(particle_count):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' requires a positive integer field 'particle_count'.",
                        )
                    )
                    continue

                if particle_weights_name is None or not isinstance(
                    particle_weights_name, ScalarNode
                ):
                    diagnostics.append(
                        _make_spec_diagnostic(
                            ErrorCode.MALFORMED_QUANTITY_DEFINITION,
                            node=quantity_name_node,
                            details=f"Quantity '{quantity_name_node.value}' is missing required field 'particle_weights_name'.",
                        )
                    )
                    continue
                shape = (int(particle_count.value), int(dim_node.value))

                quantities[quantity_name_node.value] = SpecComponentQuantityParticle(
                    shape=shape,
                    dim=int(dim_node.value),
                    particle_count=int(particle_count.value),
                    particle_weights_name=particle_weights_name.value,
                )
                if particle_weights_name.value in quantities:
                    diagnostics.append(
                        _make_duplicate_quantity_diagnostic(particle_weights_name)
                    )
                    continue
                quantities[particle_weights_name.value] = (
                    SpecComponentQuantityReference(
                        shape=(int(particle_count.value),),
                        referenced_quantity_name=quantity_name_node.value,
                    )
                )
            case SpecQuantityTypes.Reference:
                pass

    return quantities


def _parse_quantity_q_type(quantity: Node, diagnostics: list[Diagnostic]):
    q_type = _mapping_get(quantity, "q_type")
    if q_type is None:
        return SpecQuantityTypes.Generic

    if not isinstance(q_type, ScalarNode):
        return None

    try:
        return SpecQuantityTypes[q_type.value]
    except KeyError:
        return None


def _validate_allowed_properties(
    entry_node: Node,
    *,
    allowed_properties: set[str],
    error_code: ErrorCode,
    entry_kind: str,
    entry_name: str,
    diagnostics: list[Diagnostic],
) -> None:
    for property_name_node, _ in _mapping_items(entry_node):
        if not isinstance(property_name_node, ScalarNode):
            continue
        if property_name_node.value in allowed_properties:
            continue
        diagnostics.append(
            _make_spec_diagnostic(
                error_code,
                node=property_name_node,
                details=(
                    f"Unexpected property '{property_name_node.value}' in "
                    f"{entry_kind} '{entry_name}'. Allowed properties: "
                    f"{', '.join(sorted(allowed_properties))}."
                ),
            )
        )


def _parse_components(
    components_node: Node | None,
    diagnostics: list[Diagnostic],
) -> dict[str, SpecComponentProps]:
    components: dict[str, SpecComponentProps] = {}
    if components_node is None:
        return components
    if not isinstance(components_node, MappingNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_SPEC,
                node=components_node,
                details="Top-level field 'components' must be a mapping.",
            )
        )
        return components

    for component_name_node, component_node in _mapping_items(components_node):
        if not isinstance(component_name_node, ScalarNode):
            continue
        if not isinstance(component_node, MappingNode):
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_SPEC,
                    node=component_name_node,
                    details=(
                        f"Component '{component_name_node.value}' must be a mapping containing at least 'component_type'."
                    ),
                )
            )
            continue

        _validate_allowed_properties(
            component_node,
            allowed_properties={
                "class_name",
                "component_type",
                "dtype",
                "goals",
                "quantities",
            },
            error_code=ErrorCode.UNEXPECTED_COMPONENT_PROPERTY,
            entry_kind="component",
            entry_name=component_name_node.value,
            diagnostics=diagnostics,
        )

        component_type = _parse_component_type(
            component_name_node,
            component_node,
            diagnostics,
        )
        class_name = component_name_node.value
        class_name_node = _mapping_get(component_node, "class_name")
        if class_name_node is not None:
            if isinstance(class_name_node, ScalarNode):
                class_name = class_name_node.value
            else:
                diagnostics.append(
                    _make_spec_diagnostic(
                        ErrorCode.MALFORMED_SPEC,
                        node=class_name_node,
                        details=(
                            f"Component '{component_name_node.value}' field 'class_name' must be a scalar string."
                        ),
                    )
                )
                continue
        dtype = _parse_dtype(component_node, diagnostics)
        quantities = _parse_quantities(component_node, diagnostics)
        goals: list[str] = []
        goals_node = _mapping_get(component_node, "goals")
        if isinstance(goals_node, SequenceNode):
            for goal_node in goals_node.value:
                if isinstance(goal_node, ScalarNode):
                    goals.append(goal_node.value)

        if component_type is None:
            continue

        components[component_name_node.value] = SpecComponentProps(
            component_type=component_type,
            quantities=quantities,
            goals=goals,
            instance_name=component_name_node.value,
            class_name=class_name,
            dtype=dtype,
        )

    return components


def _parse_connections(
    spec_root: Node, diagnostics: list[Diagnostic]
) -> dict[str, SpecConnectionProps]:
    connections: dict[str, SpecConnectionProps] = {}
    connections_node = _mapping_get(spec_root, "connections")
    for connection_name_node, connection_node in _mapping_items(connections_node):
        if not isinstance(connection_name_node, ScalarNode):
            continue
        if not isinstance(connection_node, MappingNode):
            diagnostics.append(
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_SPEC,
                    node=connection_name_node,
                    details=(
                        f"Connection '{connection_name_node.value}' must be a mapping."
                    ),
                )
            )
            continue

        _validate_allowed_properties(
            connection_node,
            allowed_properties={
                "class_name",
                "components",
            },
            error_code=ErrorCode.UNEXPECTED_CONNECTION_PROPERTY,
            entry_kind="connection",
            entry_name=connection_name_node.value,
            diagnostics=diagnostics,
        )

        class_name = connection_name_node.value
        class_name_node = _mapping_get(connection_node, "class_name")
        if class_name_node is not None:
            if isinstance(class_name_node, ScalarNode):
                class_name = class_name_node.value
            else:
                diagnostics.append(
                    _make_spec_diagnostic(
                        ErrorCode.MALFORMED_SPEC,
                        node=class_name_node,
                        details=(
                            f"Connection '{connection_name_node.value}' field 'class_name' must be a scalar string."
                        ),
                    )
                )
                continue

        components: list[str] = []
        components_node = _mapping_get(connection_node, "components")
        if isinstance(components_node, SequenceNode):
            for component_node in components_node.value:
                if isinstance(component_node, ScalarNode):
                    components.append(component_node.value)

        connections[connection_name_node.value] = SpecConnectionProps(
            components=components,
            instance_name=connection_name_node.value,
            class_name=class_name,
        )

    return connections


def _parse_goals(
    spec_root: Node, diagnostics: list[Diagnostic]
) -> dict[str, SpecGoalProps]:
    goals: dict[str, SpecGoalProps] = {}
    goals_node = _mapping_get(spec_root, "goals")
    for goal_name_node, goal_node in _mapping_items(goals_node):
        if not isinstance(goal_name_node, ScalarNode):
            continue

        _validate_allowed_properties(
            goal_node,
            allowed_properties={
                "class_name",
            },
            error_code=ErrorCode.UNEXPECTED_GOAL_PROPERTY,
            entry_kind="goal",
            entry_name=goal_name_node.value,
            diagnostics=diagnostics,
        )

        class_name = goal_name_node.value
        class_name_node = _mapping_get(goal_node, "class_name")
        if class_name_node is not None:
            if isinstance(class_name_node, ScalarNode):
                class_name = class_name_node.value
            else:
                diagnostics.append(
                    _make_spec_diagnostic(
                        ErrorCode.MALFORMED_SPEC,
                        node=class_name_node,
                        details=(
                            f"Goal '{goal_name_node.value}' field 'class_name' must be a scalar string."
                        ),
                    )
                )
                continue

        goals[goal_name_node.value] = SpecGoalProps(
            name=class_name,
            instance_name=goal_name_node.value,
        )

    return goals


def parse_spec_yaml(spec_root: Node, diagnostics: list[Diagnostic]) -> SpecFormat:
    if not isinstance(spec_root, MappingNode):
        diagnostics.append(
            _make_spec_diagnostic(
                ErrorCode.MALFORMED_SPEC,
                node=spec_root,
                details="Spec root must be a YAML mapping.",
            )
        )
        return SpecFormat(components={}, goals={}, connections={})

    components_node = _mapping_get(spec_root, "components")
    goals_node = _mapping_get(spec_root, "goals")
    connections_node = _mapping_get(spec_root, "connections")
    _validate_no_duplicate_keys(
        components_node,
        ErrorCode.DUPLICATE_COMPONENT,
        "component",
        diagnostics,
    )
    _validate_no_duplicate_keys(
        goals_node,
        ErrorCode.DUPLICATE_GOAL,
        "goal",
        diagnostics,
    )
    _validate_no_duplicate_keys(
        connections_node,
        ErrorCode.DUPLICATE_CONNECTION,
        "connection",
        diagnostics,
    )

    return SpecFormat(
        components=_parse_components(components_node, diagnostics),
        goals=_parse_goals(spec_root, diagnostics),
        connections=_parse_connections(spec_root, diagnostics),
    )


def _yaml_error_range(error: YAMLError) -> types.Range:
    mark = getattr(error, "problem_mark", None) or getattr(error, "context_mark", None)
    line = getattr(mark, "line", None)
    column = getattr(mark, "column", None)
    if isinstance(line, int) and isinstance(column, int):
        position = types.Position(line=line, character=column)
        return types.Range(start=position, end=position)
    return _default_range()


def _yaml_error_details(error: YAMLError) -> str:
    context = getattr(error, "context", None)
    problem = getattr(error, "problem", None)
    if isinstance(context, str) and context and isinstance(problem, str) and problem:
        return f"{context}: {problem}"
    if isinstance(problem, str) and problem:
        return problem
    if isinstance(context, str) and context:
        return context
    message = str(error).strip()
    if message:
        return message
    return "YAML parser failed without additional context."


def parse_spec_content(content: str) -> ParsedSpecContent:
    try:
        yaml_module: Any = yaml
        composed_root = cast(Node | None, yaml_module.compose(content))
        spec_root = composed_root or MappingNode(
            tag="tag:yaml.org,2002:map",
            value=[],
        )
    except YAMLError as error:
        logger.error("Spec yaml malformed")
        return ParsedSpecContent(
            spec=None,
            diagnostics=[
                _make_spec_diagnostic(
                    ErrorCode.MALFORMED_SPEC,
                    details=_yaml_error_details(error),
                    range_override=_yaml_error_range(error),
                )
            ],
        )

    diagnostics: list[Diagnostic] = []
    spec_format = parse_spec_yaml(spec_root, diagnostics)

    if diagnostics:
        logger.error("Spec file contains errors")
        return ParsedSpecContent(spec=None, diagnostics=diagnostics)

    return ParsedSpecContent(spec=AICON_SPEC(spec_format), diagnostics=[])
