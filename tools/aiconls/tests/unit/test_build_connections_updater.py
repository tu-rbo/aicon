import libcst as cst

from reader.spec.spec_types import SpecConnectionProps
from updater.build_connections_updater import BuildConnectionsUpdater


def _connections() -> dict[str, SpecConnectionProps]:
    return {
        "knownConnection": SpecConnectionProps(
            components=["TestSensor"],
            class_name="KnownConnection",
        ),
        "otherConnection": SpecConnectionProps(
            components=["TestSensor"],
            class_name="OtherConnection",
        ),
    }


def test_build_connections_updater_replaces_existing_function():
    source = """
class KnownConnection(ActiveInterconnection):
    pass

def build_connections():
    return {}
"""
    updated = cst.parse_module(source).visit(BuildConnectionsUpdater(_connections()))

    assert "def build_connections():" in updated.code
    assert '"knownConnection"' in updated.code
    assert 'KnownConnection("knownConnection"' in updated.code
    assert '"otherConnection"' in updated.code
    assert 'OtherConnection("otherConnection"' in updated.code


def test_build_connections_updater_inserts_function_before_first_class():
    source = """
class KnownConnection(ActiveInterconnection):
    pass
"""
    updated = cst.parse_module(source).visit(BuildConnectionsUpdater(_connections()))

    assert updated.code.index("def build_connections():") < updated.code.index(
        "class KnownConnection"
    )
