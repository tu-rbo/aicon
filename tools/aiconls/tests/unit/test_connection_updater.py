import libcst as cst

from reader.spec.spec_types import SpecComponentQuantityGeneric, SpecQuantities
from updater.connection_updater import ConnectionUpdater


def _make_spec_quantities() -> SpecQuantities:
    return {
        "value": SpecComponentQuantityGeneric(shape=(1, 4)),
        "state": SpecComponentQuantityGeneric(shape=(2,)),
    }


def test_connection_updater_updates_super_init_quantities():
    source = """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {"value": (2, 4)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )
"""
    module = cst.parse_module(source)
    updated = module.visit(
        ConnectionUpdater("KnownConnection", _make_spec_quantities())
    )

    assert (
        updated.code
        == """
class KnownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {'value': (1, 4), 'state': (2,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )
"""
    )


def test_connection_updater_does_not_touch_other_connections():
    source = """
class OtherConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"value": (2, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)
"""
    module = cst.parse_module(source)
    updated = module.visit(
        ConnectionUpdater("KnownConnection", _make_spec_quantities())
    )
    assert updated.code == source
