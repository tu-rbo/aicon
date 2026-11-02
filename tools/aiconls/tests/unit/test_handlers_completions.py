from types import SimpleNamespace
from typing import cast

from lsprotocol import types

from handlers import completions
from server import AICONLanguageServer


class _DummyDocument:
    def __init__(self, source: str):
        self.source = source


class _DummyWorkspace:
    def __init__(self, document: _DummyDocument):
        self.document = document

    def get_text_document(self, _uri: str) -> _DummyDocument:
        return self.document


class _DummyLanguageServer:
    def __init__(
        self,
        source: str,
        component: object | None,
        connection: object | None = None,
    ):
        self.workspace = _DummyWorkspace(_DummyDocument(source))
        self.component = component
        self.connection = connection
        self.components = {"SensorA": component} if component is not None else {}
        self.connections = (
            {"DistDrawerEEConnection": connection} if connection is not None else {}
        )
        self.connection_names: list[str] = []
        self.connection_quantities: dict[tuple[str, str], list[str]] = {}
        self.last_connection_component_name: str | None = None
        self.last_connection_quantity_component_name: str | None = None
        self.parse_called = False
        self.spec: object | None = None

    def get_component_for_position(
        self, _document: _DummyDocument, _position: types.Position
    ) -> object | None:
        return self.component

    def parse(self, _document: _DummyDocument) -> None:
        self.parse_called = True

    def refresh_document_if_stale(self, _document: _DummyDocument) -> None:
        return None

    def get_connection_for_position(
        self, _document: _DummyDocument, _position: types.Position
    ) -> object | None:
        return self.connection

    def get_connection_names_for_component(self, _component_name: str) -> list[str]:
        self.last_connection_component_name = _component_name
        return self.connection_names

    def get_connection_quantities_for_component(
        self, component_name: str, connection_name: str
    ) -> list[str]:
        self.last_connection_quantity_component_name = component_name
        return self.connection_quantities.get((component_name, connection_name), [])


class _DummySpec:
    def get_component_instances_by_class_name(self, class_name: str) -> list[str]:
        if class_name == "SensorA":
            return ["sensor_a_instance"]
        return []


def _completion_params(source: str) -> types.CompletionParams:
    return types.CompletionParams(
        text_document=types.TextDocumentIdentifier(uri="file:///component.py"),
        position=types.Position(line=0, character=len(source)),
    )


def test_completions_suggest_quantities():
    line = 'self.quantities["'
    component = SimpleNamespace(
        name="SensorA", quantities={"q1": object(), "q2": object()}
    )
    ls = _DummyLanguageServer(line, component)

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["q1", "q2"]


def test_completions_suggest_connections_for_component():
    line = 'self.connections["'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.connection_names = ["eeDrawerDistance"]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["eeDrawerDistance"]


def test_completions_suggest_connected_quantities_accessor_for_valid_connection():
    line = 'self.connections["eeDrawerDistance"].'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.connection_names = ["eeDrawerDistance"]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["connected_quantities[]"]
    assert items[0].insert_text == 'connected_quantities[""]'


def test_completions_do_not_suggest_connected_quantities_accessor_for_invalid_connection():
    line = 'self.connections["unknownConnection"].'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.connection_names = ["eeDrawerDistance"]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is None


def test_completions_suggest_external_connected_quantities():
    line = 'self.connections["eeDrawerDistance"].connected_quantities["'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.connection_names = ["eeDrawerDistance"]
    ls.connection_quantities[("SensorA", "eeDrawerDistance")] = [
        "position_drawer",
        "uncertainty_drawer",
    ]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["position_drawer", "uncertainty_drawer"]


def test_completions_use_component_instance_name_for_connection_lookups():
    line = 'self.connections["'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.spec = _DummySpec()
    ls.connection_names = ["eeDrawerDistance"]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert ls.last_connection_component_name == "sensor_a_instance"


def test_completions_use_component_instance_name_for_connected_quantities_lookup():
    line = 'self.connections["eeDrawerDistance"].connected_quantities["'
    component = SimpleNamespace(name="SensorA", quantities={})
    ls = _DummyLanguageServer(line, component)
    ls.spec = _DummySpec()
    ls.connection_names = ["eeDrawerDistance"]
    ls.connection_quantities[("sensor_a_instance", "eeDrawerDistance")] = ["ext_q"]

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["ext_q"]
    assert ls.last_connection_quantity_component_name == "sensor_a_instance"


def test_completions_suggest_connected_quantities_for_connection():
    line = 'self.connected_quantities["'
    connection = SimpleNamespace(
        quantities={
            "pose_ee": object(),
            "position_drawer": object(),
        }
    )
    ls = _DummyLanguageServer(line, component=None, connection=connection)

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["pose_ee", "position_drawer"]


def test_completions_suggest_connected_quantities_for_closed_empty_key():
    line = 'self.connected_quantities[""]'
    connection = SimpleNamespace(quantities={"pose_ee": object()})
    ls = _DummyLanguageServer(line, component=None, connection=connection)

    items = completions(cast(AICONLanguageServer, ls), _completion_params(line))

    assert items is not None
    assert [item.label for item in items] == ["pose_ee"]
