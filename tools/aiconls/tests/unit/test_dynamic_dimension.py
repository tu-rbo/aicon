from types import SimpleNamespace
from typing import Any, cast

import libcst as cst
import torch
from lsprotocol import types

from dynamic.loader import (
    DynamicNetworkInstantiation,
    _enable_random_sensor_values,  # pyright: ignore[reportPrivateUsage]
    get_component_quantity_shape_mismatch_details,
)
from generator.quantities import gen_init_quantities
from reader.aicon_files.extractor.util import build_shape_tuple, extract_shapes_dict
from reader.aicon_files.reader_types import ParsedQuantity, ParsedSensor
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import SpecComponentQuantityGeneric, shapes_match
from validator.component_validator import ComponentValidator

SPEC = """
components:
  PointSensor:
    component_type: Sensor
    quantities:
      points:
        shape: (-1, 3)
      depth:
        shape: (2,)
""".strip()


def _parse_spec() -> AICON_SPEC:
    parsed = parse_spec_content(SPEC)
    assert parsed.spec is not None
    return parsed.spec


def _make_instantiation(quantities: dict[str, torch.Tensor]) -> Any:
    instance = SimpleNamespace(quantities=quantities)
    return cast(
        DynamicNetworkInstantiation,
        SimpleNamespace(components={"PointSensor": SimpleNamespace(instance=instance)}),
    )


def test_shapes_match_skips_dynamic_shapes() -> None:
    assert shapes_match((5, 3), (-1, 3))
    assert shapes_match((-1,), (2, 2))
    assert shapes_match((2,), (2,))
    assert not shapes_match((2,), (3,))


def test_spec_reader_accepts_dynamic_dimension() -> None:
    parsed = parse_spec_content(SPEC)

    assert parsed.diagnostics == []
    assert parsed.spec is not None
    points = parsed.spec.components["PointSensor"].quantities["points"]
    assert points.shape == (-1, 3)


def test_spec_reader_rejects_other_negative_dimensions() -> None:
    parsed = parse_spec_content(SPEC.replace("(-1, 3)", "(-2, 3)"))

    assert parsed.spec is None
    assert [diagnostic.code for diagnostic in parsed.diagnostics] == ["E003"]


def test_extract_shapes_dict_reads_dynamic_dimension() -> None:
    quantities = extract_shapes_dict(
        cst.parse_expression('{"points": (-1, 3), "depth": (2,)}')
    )

    assert quantities["points"].shape == (-1, 3)
    assert quantities["depth"].shape == (2,)


def test_extract_shapes_dict_treats_named_constants_as_dynamic() -> None:
    quantities = extract_shapes_dict(
        cst.parse_expression('{"obstacle_points": (NUM_OBSTACLE_POINTS, 3)}')
    )

    assert quantities["obstacle_points"].shape == (-1, 3)


def test_build_shape_tuple_round_trips_dynamic_dimension() -> None:
    shape_tuple = build_shape_tuple((-1, 3))

    assert cst.Module(body=[]).code_for_node(shape_tuple) == "(-1, 3)"
    quantities = extract_shapes_dict(
        cst.Dict(
            elements=[
                cst.DictElement(key=cst.SimpleString('"points"'), value=shape_tuple)
            ]
        )
    )
    assert quantities["points"].shape == (-1, 3)


def test_component_validator_skips_dynamic_quantities() -> None:
    quantities_range = types.Range(
        start=types.Position(line=4, character=0),
        end=types.Position(line=4, character=10),
    )
    sensor = ParsedSensor(
        document_uri="file:///component.py",
        name="PointSensor",
        position=quantities_range,
        quantities={
            "points": ParsedQuantity(shape=(-1,)),
            "depth": ParsedQuantity(shape=(2,)),
        },
        goals=[],
        instance_name="PointSensor",
        class_name="PointSensor",
        quantities_pos=quantities_range,
        quantity_references=[],
        connection_references={},
        connected_quantity_references=[],
    )

    assert ComponentValidator(_parse_spec(), sensor).diagnostics == []


def test_runtime_shape_check_skips_dynamic_quantities() -> None:
    instantiation = _make_instantiation(
        {"points": torch.zeros(7, 3), "depth": torch.zeros(3)}
    )

    mismatches = get_component_quantity_shape_mismatch_details(
        _parse_spec(), instantiation
    )

    assert mismatches == {"PointSensor": ("depth: expected shape (2,), got (3,)",)}


def test_random_sensor_values_leave_dynamic_quantities_untouched() -> None:
    points = torch.zeros(0, 3)
    instantiation = _make_instantiation({"points": points, "depth": torch.zeros(2)})

    _enable_random_sensor_values(_parse_spec(), instantiation)
    sensor = instantiation.components["PointSensor"].instance
    assert sensor.obtain_measurements()

    assert sensor.quantities["points"] is points
    assert sensor.quantities["depth"].shape == (2,)


def test_generated_init_starts_dynamic_quantities_empty() -> None:
    generated = gen_init_quantities(
        "points", SpecComponentQuantityGeneric(shape=(-1, 3))
    )

    assert "torch.zeros(\n            (0, 3)," in generated
