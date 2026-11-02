from dataclasses import dataclass, field

from lsprotocol import types

from reader.spec.spec_types import (
    SpecComponentTypes,
    TensorShape,
)


def _str_set_factory() -> set[str]:
    return set()


def _range_dict_factory() -> dict[str, types.Range]:
    return {}


def _connection_quantity_reference_list_factory() -> list[
    "ConnectionQuantityReference"
]:
    return []


def _quantity_reference_list_factory() -> list["QuantityReference"]:
    return []


@dataclass
class ParsedQuantity:
    shape: TensorShape


ParsedQuantities = dict[str, ParsedQuantity]


@dataclass
class ConnectionQuantityReference:
    connection_name: str
    quantity_name: str
    position: types.Range


@dataclass
class QuantityReference:
    quantity_name: str
    position: types.Range


@dataclass
class ParsedComponent:
    component_type: SpecComponentTypes
    goals: list[str]
    instance_name: str
    class_name: str
    position: types.Range
    document_uri: str
    quantities: ParsedQuantities


ParsedComponents = dict[str, ParsedComponent]


@dataclass
class _ParsedQuantifiedComponentNonDefault:
    name: str
    quantities_pos: types.Range | None
    quantity_references: list[QuantityReference] = field(
        default_factory=_quantity_reference_list_factory
    )
    connection_references: dict[str, types.Range] = field(
        default_factory=_range_dict_factory
    )
    connected_quantity_references: list[ConnectionQuantityReference] = field(
        default_factory=_connection_quantity_reference_list_factory
    )


@dataclass
class ParsedSensor(_ParsedQuantifiedComponentNonDefault, ParsedComponent):
    component_type: SpecComponentTypes = field(
        default=SpecComponentTypes["Sensor"], init=False
    )


@dataclass
class ParsedAction(_ParsedQuantifiedComponentNonDefault, ParsedComponent):
    component_type: SpecComponentTypes = field(
        default=SpecComponentTypes["Action"], init=False
    )


@dataclass
class ParsedEstimator(_ParsedQuantifiedComponentNonDefault, ParsedComponent):
    component_type: SpecComponentTypes = field(
        default=SpecComponentTypes["Estimator"], init=False
    )
    f_func_params: dict[str, types.Range] = field(default_factory=_range_dict_factory)
    f_output_signature_diff_names: dict[str, types.Range] = field(
        default_factory=_range_dict_factory
    )
    f_output_signature_non_diff_names: dict[str, types.Range] = field(
        default_factory=_range_dict_factory
    )


@dataclass
class ParsedGoal:
    name: str
    position: types.Range
    document_uri: str
    goal_func_params: dict[str, types.Range]


ParsedGoals = dict[str, ParsedGoal]


@dataclass
class ParsedConnection:
    name: str
    position: types.Range
    document_uri: str
    quantities: ParsedQuantities
    quantities_pos: types.Range | None
    connection_func_params: dict[str, types.Range]
    quantity_positions: dict[str, types.Range] = field(
        default_factory=_range_dict_factory
    )
    # quantities of required_signs_dict: those whose gradient is switched and the switching conditions
    switching_quantities: set[str] = field(default_factory=_str_set_factory)


ParsedConnections = dict[str, ParsedConnection]
