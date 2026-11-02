# pyright: reportPrivateUsage=false

from pathlib import Path

from lsprotocol import types
from pytest import MonkeyPatch

from reader.aicon_files.reader_types import (
    ParsedAction,
    ParsedConnection,
    ParsedEstimator,
    ParsedQuantity,
    ParsedSensor,
)
from reader.spec.spec_reader import parse_spec_content
from server import AICONLanguageServer, _ProjectPythonDocument


def _write(path: Path, source: str) -> str:
    path.write_text(source, encoding="utf-8")
    return path.as_uri()


def _range() -> types.Range:
    return types.Range(
        start=types.Position(line=0, character=0),
        end=types.Position(line=0, character=1),
    )


def test_shape_propagation_updates_spec_and_connection_sources(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    spec_source = """components:
  SensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
connections:
  TestConnection:
    class_name: TestConnection
    components: [SensorA]
"""
    component_source = """class SensorA:
    def __init__(self):
            self.cls_quantity_names_and_shapes = {"value": (2,)}
"""
    connection_source = """class TestConnection:
    def __init__(self):
        super().__init__("TestConnection", {"value": (1,)})

    def define_implicit_connection_function(self):
        def connection_func(value):
            return value
"""
    spec_uri = _write(tmp_path / "spec.yaml", spec_source)
    component_uri = _write(tmp_path / "sensor.py", component_source)
    connection_uri = _write(tmp_path / "connection.py", connection_source)

    ls = AICONLanguageServer("diagnostic-server", "v1")
    ls.components = {
        "SensorA": ParsedSensor(
            name="SensorA",
            quantities_pos=_range(),
            document_uri=component_uri,
            position=_range(),
            quantities={"value": ParsedQuantity(shape=(2,))},
            goals=[],
            instance_name="SensorA",
            class_name="SensorA",
        )
    }
    ls.connections = {
        "TestConnection": ParsedConnection(
            name="TestConnection",
            position=_range(),
            document_uri=connection_uri,
            quantities={"value": ParsedQuantity(shape=(1,))},
            quantities_pos=_range(),
            connection_func_params={"value": _range()},
        )
    }

    updated_spec_source = ls._build_component_quantities_spec_source(
        spec_source,
        "SensorA",
        {"value": ParsedQuantity(shape=(2,))},
    )
    assert updated_spec_source is not None
    assert "shape: (2,)" in updated_spec_source
    assert "shape: (1,)" not in updated_spec_source

    parsed_spec = parse_spec_content(updated_spec_source)
    assert parsed_spec.spec is not None
    project_edit = ls._build_project_quantity_sync_edit(parsed_spec.spec)
    assert project_edit is not None
    assert project_edit.changes is not None
    assert connection_uri in project_edit.changes

    connection_edit = project_edit.changes[connection_uri][0]
    assert "(2,)" in connection_edit.new_text
    assert "(1,)" not in connection_edit.new_text
    assert spec_uri not in project_edit.changes


def test_shape_propagation_scopes_updates_to_changed_component_quantities(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    spec_source = """components:
  SensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (2,)
  SensorB:
    class_name: SensorB
    component_type: Sensor
    quantities:
      other:
        shape: (4,)
connections:
  RelatedConnection:
    class_name: RelatedConnection
    components: [SensorA]
  UnrelatedConnection:
    class_name: UnrelatedConnection
    components: [SensorB]
"""
    related_connection_source = """class RelatedConnection:
    def __init__(self):
        super().__init__("RelatedConnection", {"value": (1,)})
"""
    unrelated_connection_source = """class UnrelatedConnection:
    def __init__(self):
        super().__init__("UnrelatedConnection", {"other": (1,)})
"""
    related_connection_uri = _write(
        tmp_path / "related_connection.py", related_connection_source
    )
    unrelated_connection_uri = _write(
        tmp_path / "unrelated_connection.py", unrelated_connection_source
    )

    ls = AICONLanguageServer("diagnostic-server", "v1")
    ls.connections = {
        "RelatedConnection": ParsedConnection(
            name="RelatedConnection",
            position=_range(),
            document_uri=related_connection_uri,
            quantities={"value": ParsedQuantity(shape=(1,))},
            quantities_pos=_range(),
            connection_func_params={"value": _range()},
        ),
        "UnrelatedConnection": ParsedConnection(
            name="UnrelatedConnection",
            position=_range(),
            document_uri=unrelated_connection_uri,
            quantities={"other": ParsedQuantity(shape=(1,))},
            quantities_pos=_range(),
            connection_func_params={"other": _range()},
        ),
    }

    parsed_spec = parse_spec_content(spec_source)
    assert parsed_spec.spec is not None
    project_edit = ls._build_quantity_scoped_project_sync_edit(
        parsed_spec.spec,
        changed_component_instance="SensorA",
        changed_quantity_names={"value"},
    )

    assert project_edit is not None
    assert project_edit.changes is not None
    assert related_connection_uri in project_edit.changes
    assert unrelated_connection_uri not in project_edit.changes
    related_connection_edit = project_edit.changes[related_connection_uri][0]
    assert "(2,)" in related_connection_edit.new_text
    assert "(1,)" not in related_connection_edit.new_text


def test_component_spec_update_allows_arbitrary_quantity_changes(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    spec_source = """components:
  SensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      old_value:
        shape: (1,)
      to_remove:
        shape: (3,)
"""
    _write(tmp_path / "spec.yaml", spec_source)

    ls = AICONLanguageServer("diagnostic-server", "v1")
    updated_spec_source = ls._build_component_quantities_spec_source(
        spec_source,
        "SensorA",
        {
            "renamed_value": ParsedQuantity(shape=(2,)),
            "added_value": ParsedQuantity(shape=(4,)),
        },
    )
    assert updated_spec_source is not None
    assert "old_value" not in updated_spec_source
    assert "to_remove" not in updated_spec_source
    assert "renamed_value" in updated_spec_source
    assert "added_value" in updated_spec_source
    assert "shape: (2,)" in updated_spec_source
    assert "shape: (4,)" in updated_spec_source


def test_update_document_component_quantities_matches_spec_by_class_name(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    spec_source = """components:
  sensor_instance:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (2,)
"""
    component_source = """class SensorA:
    def __init__(self):
            self.cls_quantity_names_and_shapes = {"value": (1,)}
"""
    _write(tmp_path / "spec.yaml", spec_source)
    component_uri = _write(tmp_path / "sensor.py", component_source)

    ls = AICONLanguageServer("diagnostic-server", "v1")
    parsed_spec = parse_spec_content(spec_source)
    assert parsed_spec.spec is not None
    ls.spec = parsed_spec.spec

    components: list[ParsedSensor | ParsedAction | ParsedEstimator] = [
        ParsedSensor(
            name="SensorA",
            quantities_pos=_range(),
            document_uri=component_uri,
            position=_range(),
            quantities={"value": ParsedQuantity(shape=(1,))},
            goals=[],
            instance_name="SensorA",
            class_name="SensorA",
        )
    ]

    updated_source = ls._update_document_component_quantities(
        _ProjectPythonDocument(path=tmp_path / "sensor.py", source=component_source),
        components,
        source=component_source,
    )

    assert updated_source is not None
    assert "cls_quantity_names_and_shapes = {'value': (2,)}" in updated_source


def test_update_document_component_quantities_reports_parse_errors(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    spec_source = """components:
  sensor_instance:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (2,)
"""
    component_source = "class SensorA(:\n    pass\n"
    _write(tmp_path / "spec.yaml", spec_source)
    component_uri = _write(tmp_path / "sensor.py", component_source)

    ls = AICONLanguageServer("diagnostic-server", "v1")
    parsed_spec = parse_spec_content(spec_source)
    assert parsed_spec.spec is not None
    ls.spec = parsed_spec.spec

    messages: list[str] = []

    def _window_log_message(params: types.LogMessageParams) -> None:
        messages.append(params.message)

    monkeypatch.setattr(ls, "window_log_message", _window_log_message)

    components: list[ParsedSensor | ParsedAction | ParsedEstimator] = [
        ParsedSensor(
            name="SensorA",
            quantities_pos=_range(),
            document_uri=component_uri,
            position=_range(),
            quantities={"value": ParsedQuantity(shape=(1,))},
            goals=[],
            instance_name="SensorA",
            class_name="SensorA",
        )
    ]

    updated_source = ls._update_document_component_quantities(
        _ProjectPythonDocument(path=tmp_path / "sensor.py", source=component_source),
        components,
        source=component_source,
    )

    assert updated_source is None
    assert messages
    assert "Could not parse component source for quantity update" in messages[0]
