import libcst as cst

from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentQuantityGeneric,
    SpecComponentTypes,
)
from updater.sensor_updater import SensorUpdater


def _make_spec_component() -> SpecComponentProps:
    return SpecComponentProps(
        component_type=SpecComponentTypes.Sensor,
        quantities={
            "value": SpecComponentQuantityGeneric(shape=(1, 4)),
            "state": SpecComponentQuantityGeneric(shape=(2,)),
        },
        goals=[],
        instance_name="KnownSensor",
        class_name="KnownSensor",
    )


def test_sensor_updater_updates_cls_quantity_names_and_shapes():
    source = """
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (2, 4)}
"""
    module = cst.parse_module(source)
    updated = module.visit(SensorUpdater("KnownSensor", _make_spec_component()))

    assert (
        updated.code
        == """
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {'value': (1, 4), 'state': (2,)}
"""
    )


def test_sensor_updater_does_not_touch_other_components():
    source = """
class OtherSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (2, 4)}
"""
    module = cst.parse_module(source)
    updated = module.visit(SensorUpdater("KnownSensor", _make_spec_component()))
    assert updated.code == source


def test_sensor_updater_inserts_missing_cls_quantity_names_and_shapes():
    source = """
class KnownSensor(SensorComponent):
    def __init__(self):
        super().__init__()
"""
    module = cst.parse_module(source)
    updated = module.visit(SensorUpdater("KnownSensor", _make_spec_component()))

    assert (
        updated.code
        == """
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {'value': (1, 4), 'state': (2,)}
    def __init__(self):
        super().__init__()
"""
    )
