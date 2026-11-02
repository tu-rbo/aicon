from collections.abc import Sequence
from pathlib import Path

from lsprotocol import types
from pytest import MonkeyPatch

from server import AICONLanguageServer


def _write(path: Path, source: str) -> str:
    path.write_text(source, encoding="utf-8")
    return path.as_uri()


def _position_for_token(source: str, token: str) -> types.Position:
    for line_index, line in enumerate(source.splitlines()):
        character = line.find(token)
        if character >= 0:
            return types.Position(line=line_index, character=character)
    raise AssertionError(f"Token {token!r} not found")


def test_build_quantity_rename_edit_updates_occurrences_across_project(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    component_source = """class SensorA:
    def __init__(self):
        self.cls_quantity_names_and_shapes = {"value": (1,)}
        self.quantities["value"]
        self.connections["testConnection"].connected_quantities["value"]
"""
    connection_source = """class TestConnection:
    def __init__(self):
        super().__init__("TestConnection", {"value": (1,)})

    def define_implicit_connection_function(self):
        def connection_func(value):
            return value
"""
    goal_source = """class TestGoal:
    def define_goal_cost_function(self):
        def goal_func(value):
            return value
"""
    spec_source = """components:
  sensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
"""

    component_uri = _write(tmp_path / "sensor.py", component_source)
    connection_uri = _write(tmp_path / "connection.py", connection_source)
    goal_uri = _write(tmp_path / "goal.py", goal_source)
    spec_uri = _write(tmp_path / "spec.yaml", spec_source)
    ls = AICONLanguageServer("diagnostic-server", "v1")

    edit = ls.build_quantity_rename_edit(
        component_uri,
        _position_for_token(component_source, "value"),
        "renamed_value",
    )

    assert edit is not None
    assert edit.changes is not None
    assert set(edit.changes.keys()) == {
        component_uri,
        connection_uri,
        goal_uri,
        spec_uri,
    }
    assert len(edit.changes[component_uri]) == 3
    assert len(edit.changes[connection_uri]) == 3
    assert len(edit.changes[goal_uri]) == 2
    assert len(edit.changes[spec_uri]) == 1
    assert {text_edit.new_text for text_edit in edit.changes[component_uri]} == {
        '"renamed_value"'
    }
    assert {text_edit.new_text for text_edit in edit.changes[connection_uri]} == {
        '"renamed_value"',
        "renamed_value",
    }
    assert {text_edit.new_text for text_edit in edit.changes[goal_uri]} == {
        "renamed_value"
    }
    assert {text_edit.new_text for text_edit in edit.changes[spec_uri]} == {
        "renamed_value"
    }


def test_build_quantity_rename_edit_rejects_invalid_identifier_for_parameter_renames(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    source = """class SensorA:
    def __init__(self):
        self.cls_quantity_names_and_shapes = {"value": (1,)}
"""
    connection_source = """class TestConnection:
    def __init__(self):
        super().__init__("TestConnection", {"value": (1,)})

    def define_implicit_connection_function(self):
        def connection_func(value):
            return value
"""
    _write(tmp_path / "sensor.py", source)
    _write(tmp_path / "connection.py", connection_source)
    _write(
        tmp_path / "spec.yaml",
        """components:
  sensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
""",
    )
    ls = AICONLanguageServer("diagnostic-server", "v1")

    edit = ls.build_quantity_rename_edit(
        (tmp_path / "sensor.py").as_uri(),
        _position_for_token(source, "value"),
        "invalid-name",
    )

    assert edit is None


def test_build_quantity_rename_edit_can_start_from_spec_quantity(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    component_source = """class SensorA:
    def __init__(self):
        self.cls_quantity_names_and_shapes = {"value": (1,)}
"""
    spec_source = """components:
  sensorA:
    class_name: SensorA
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
"""
    component_uri = _write(tmp_path / "sensor.py", component_source)
    spec_uri = _write(tmp_path / "spec.yaml", spec_source)
    ls = AICONLanguageServer("diagnostic-server", "v1")

    edit = ls.build_quantity_rename_edit(
        spec_uri,
        _position_for_token(spec_source, "value"),
        "renamed_value",
    )

    assert edit is not None
    assert edit.changes is not None
    assert set(edit.changes.keys()) == {component_uri, spec_uri}
    assert len(edit.changes[spec_uri]) == 1
    assert len(edit.changes[component_uri]) == 1
    assert edit.changes[component_uri][0].new_text == '"renamed_value"'
    assert edit.changes[spec_uri][0].new_text == "renamed_value"


def _apply_edits(source: str, edits: Sequence[types.TextEdit]) -> str:
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def _offset(position: types.Position) -> int:
        return offsets[position.line] + position.character

    result = source
    for edit in sorted(edits, key=lambda e: _offset(e.range.start), reverse=True):
        start = _offset(edit.range.start)
        end = _offset(edit.range.end)
        result = result[:start] + edit.new_text + result[end:]
    return result


def test_build_quantity_rename_edit_updates_estimator_f_func_and_output_signatures(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    estimator_source = """class EstimatorA:
    cls_quantity_names_and_shapes = {"belief": (1,)}

    def initialize_quantities(self):
        self.quantities["belief"] = torch.ones(1)
        return True

    def define_estimation_function_f(self):
        def f_func(belief, measurement, dt):
            new_belief = belief + measurement * dt
            return (new_belief,), (new_belief,)

        return f_func, ["belief"], ["belief"]
"""
    spec_source = """components:
  estimatorA:
    class_name: EstimatorA
    component_type: Estimator
    quantities:
      belief:
        shape: (1,)
"""
    estimator_uri = _write(tmp_path / "estimators.py", estimator_source)
    spec_uri = _write(tmp_path / "spec.yaml", spec_source)
    ls = AICONLanguageServer("diagnostic-server", "v1")

    edit = ls.build_quantity_rename_edit(
        spec_uri,
        _position_for_token(spec_source, "belief"),
        "state",
    )

    assert edit is not None
    assert edit.changes is not None
    assert _apply_edits(estimator_source, edit.changes[estimator_uri]) == (
        """class EstimatorA:
    cls_quantity_names_and_shapes = {"state": (1,)}

    def initialize_quantities(self):
        self.quantities["state"] = torch.ones(1)
        return True

    def define_estimation_function_f(self):
        def f_func(state, measurement, dt):
            new_belief = state + measurement * dt
            return (new_belief,), (new_belief,)

        return f_func, ["state"], ["state"]
"""
    )
    assert _apply_edits(spec_source, edit.changes[spec_uri]) == spec_source.replace(
        "belief", "state"
    )


def test_build_quantity_rename_edit_ignores_attribute_and_keyword_names(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))

    goal_source = """class TestGoal:
    def define_goal_cost_function(self):
        def goal_func(value):
            cost = torch.sum(value.value, dim=0) + scale(value=value)
            return (cost,), (cost,)

        return goal_func
"""
    goal_uri = _write(tmp_path / "goals.py", goal_source)
    _write(tmp_path / "spec.yaml", "goals:\n  TestGoal:\n")
    ls = AICONLanguageServer("diagnostic-server", "v1")

    edit = ls.build_quantity_rename_edit(
        goal_uri,
        _position_for_token(goal_source, "value"),
        "renamed",
    )

    assert edit is not None
    assert edit.changes is not None
    assert _apply_edits(goal_source, edit.changes[goal_uri]) == (
        """class TestGoal:
    def define_goal_cost_function(self):
        def goal_func(renamed):
            cost = torch.sum(renamed.value, dim=0) + scale(value=renamed)
            return (cost,), (cost,)

        return goal_func
"""
    )
