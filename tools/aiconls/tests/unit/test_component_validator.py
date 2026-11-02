from lsprotocol import types

from reader.aicon_files.reader_types import (
    ConnectionQuantityReference,
    ParsedQuantities,
    ParsedQuantity,
    ParsedSensor,
    QuantityReference,
)
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentQuantityGeneric,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
)
from validator.component_validator import ComponentValidator


def _make_spec() -> AICON_SPEC:
    return AICON_SPEC(
        SpecFormat(
            components={
                "KnownSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "value": SpecComponentQuantityGeneric(shape=(1, 4)),
                    },
                    goals=[],
                    instance_name="KnownSensor",
                    class_name="KnownSensor",
                ),
                "OtherEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "estimate": SpecComponentQuantityGeneric(shape=(1, 4)),
                    },
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


def _make_class_named_spec() -> AICON_SPEC:
    return AICON_SPEC(
        SpecFormat(
            components={
                "known_sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        "value": SpecComponentQuantityGeneric(shape=(1, 4)),
                    },
                    goals=[],
                    instance_name="known_sensor_instance",
                    class_name="KnownSensor",
                ),
                "other_estimator_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={
                        "estimate": SpecComponentQuantityGeneric(shape=(1, 4)),
                    },
                    goals=[],
                    instance_name="other_estimator_instance",
                    class_name="OtherEstimator",
                ),
            },
            goals={},
            connections={
                "knownConnection": SpecConnectionProps(
                    components=["known_sensor_instance", "other_estimator_instance"],
                ),
            },
        )
    )


def _make_parsed_sensor(
    name: str,
    quantities: ParsedQuantities,
    quantities_pos: types.Range | None,
    position: types.Range | None = None,
    quantity_references: list[QuantityReference] | None = None,
    connection_references: dict[str, types.Range] | None = None,
    connected_quantity_references: list[ConnectionQuantityReference] | None = None,
) -> ParsedSensor:
    if position is None:
        position = types.Range(
            start=types.Position(line=1, character=0),
            end=types.Position(line=1, character=10),
        )

    return ParsedSensor(
        document_uri="file:///component.py",
        name=name,
        position=position,
        quantities=quantities,
        goals=[],
        instance_name=name,
        class_name=name,
        quantities_pos=quantities_pos,
        quantity_references=quantity_references or [],
        connection_references=connection_references or {},
        connected_quantity_references=connected_quantity_references or [],
    )


def test_component_validator_reports_missing_component():
    unknown_component_range = types.Range(
        start=types.Position(line=3, character=2),
        end=types.Position(line=3, character=15),
    )
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "UnknownSensor",
            {"value": ParsedQuantity(shape=(1, 4))},
            quantities_pos=unknown_component_range,
            position=unknown_component_range,
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["W101"]
    assert (
        validator.diagnostics[0].message
        == "Component 'UnknownSensor' not found in spec."
    )
    assert validator.diagnostics[0].range == unknown_component_range


def test_component_validator_reports_missing_quantities():
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor", {"value": ParsedQuantity(shape=(1, 4))}, None
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["W103"]
    assert validator.diagnostics[0].severity == types.DiagnosticSeverity.Warning
    assert (
        validator.diagnostics[0].message
        == "Component 'KnownSensor' is missing local quantity definitions. "
        "Declare them in the class attribute cls_quantity_names_and_shapes. "
        "Expected quantities from spec: value."
    )


def test_component_validator_reports_shape_mismatch():
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {"value": ParsedQuantity(shape=(2, 4))},
            types.Range(
                start=types.Position(line=4, character=0),
                end=types.Position(line=4, character=10),
            ),
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["E102"]
    assert (
        validator.diagnostics[0].message
        == "Component 'KnownSensor' quantities do not match spec: quantity 'value' has "
        "shape (2, 4), expected (1, 4)."
    )


def test_component_validator_reports_reference_to_unconnected_connection():
    connection_ref_range = types.Range(
        start=types.Position(line=10, character=25),
        end=types.Position(line=10, character=44),
    )
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {"value": ParsedQuantity(shape=(1, 4))},
            types.Range(
                start=types.Position(line=4, character=0),
                end=types.Position(line=4, character=10),
            ),
            connection_references={"unknownConnection": connection_ref_range},
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["E104"]
    assert "connection 'unknownConnection'" in validator.diagnostics[0].message
    assert "knownConnection" in validator.diagnostics[0].message
    assert validator.diagnostics[0].range == connection_ref_range


def test_component_validator_reports_unknown_connected_quantity_access():
    connected_quantity_range = types.Range(
        start=types.Position(line=12, character=50),
        end=types.Position(line=12, character=67),
    )
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {"value": ParsedQuantity(shape=(1, 4))},
            types.Range(
                start=types.Position(line=4, character=0),
                end=types.Position(line=4, character=10),
            ),
            connection_references={
                "knownConnection": types.Range(
                    start=types.Position(line=12, character=25),
                    end=types.Position(line=12, character=42),
                )
            },
            connected_quantity_references=[
                ConnectionQuantityReference(
                    connection_name="knownConnection",
                    quantity_name="missing_quantity",
                    position=connected_quantity_range,
                )
            ],
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["E105"]
    assert "quantity 'missing_quantity'" in validator.diagnostics[0].message
    assert (
        "Available connected quantities: estimate." in validator.diagnostics[0].message
    )
    assert validator.diagnostics[0].range == connected_quantity_range


def test_component_validator_checks_connections_without_quantity_definitions():
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {},
            None,
            connection_references={
                "unknownConnection": types.Range(
                    start=types.Position(line=10, character=25),
                    end=types.Position(line=10, character=44),
                ),
                "knownConnection": types.Range(
                    start=types.Position(line=12, character=25),
                    end=types.Position(line=12, character=42),
                ),
            },
            connected_quantity_references=[
                ConnectionQuantityReference(
                    connection_name="knownConnection",
                    quantity_name="missing_quantity",
                    position=types.Range(
                        start=types.Position(line=12, character=50),
                        end=types.Position(line=12, character=67),
                    ),
                )
            ],
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W103",
        "E104",
        "E105",
    ]


def test_component_validator_reports_unknown_local_quantity_access():
    local_quantity_range = types.Range(
        start=types.Position(line=8, character=24),
        end=types.Position(line=8, character=43),
    )
    validator = ComponentValidator(
        _make_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {"value": ParsedQuantity(shape=(1, 4))},
            types.Range(
                start=types.Position(line=4, character=0),
                end=types.Position(line=4, character=10),
            ),
            quantity_references=[
                QuantityReference(
                    quantity_name="missing_quantity",
                    position=local_quantity_range,
                )
            ],
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["E107"]
    assert "quantity 'missing_quantity'" in validator.diagnostics[0].message
    assert "Available local quantities: value." in validator.diagnostics[0].message
    assert validator.diagnostics[0].range == local_quantity_range


def test_component_validator_matches_spec_component_by_class_name():
    validator = ComponentValidator(
        _make_class_named_spec(),
        _make_parsed_sensor(
            "KnownSensor",
            {"value": ParsedQuantity(shape=(1, 4))},
            types.Range(
                start=types.Position(line=4, character=0),
                end=types.Position(line=4, character=10),
            ),
            connection_references={
                "knownConnection": types.Range(
                    start=types.Position(line=12, character=25),
                    end=types.Position(line=12, character=42),
                )
            },
        ),
    )

    assert validator.diagnostics == []


def test_component_validator_allows_local_quantities_of_all_instances():
    spec = AICON_SPEC(
        SpecFormat(
            components={
                f"sensor_{index}": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={
                        quantity_name: SpecComponentQuantityGeneric(shape=(1,)),
                    },
                    goals=[],
                    instance_name=f"sensor_{index}",
                    class_name="KnownSensor",
                )
                for index, quantity_name in enumerate(("first_value", "second_value"))
            },
            goals={},
            connections={},
        )
    )
    access_range = types.Range(
        start=types.Position(line=6, character=8),
        end=types.Position(line=6, character=22),
    )
    validator = ComponentValidator(
        spec,
        _make_parsed_sensor(
            "KnownSensor",
            {"first_value": ParsedQuantity(shape=(1,))},
            access_range,
            quantity_references=[
                QuantityReference(quantity_name=name, position=access_range)
                for name in ("first_value", "second_value", "third_value")
            ],
        ),
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == ["E107"]
    assert "'third_value'" in validator.diagnostics[0].message
