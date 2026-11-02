from typing import cast

from lsprotocol import types

from handlers import code_action_provider, fix_all_spec_mismatches, run_dynamic_check
from server import AICONLanguageServer


class _DummyDocument:
    def __init__(self, source: str):
        self.source = source

    def offset_at_position(self, position: types.Position) -> int:
        lines = self.source.splitlines(keepends=True)
        return sum(len(line) for line in lines[: position.line]) + position.character


class _DummyWorkspace:
    def __init__(self, document: _DummyDocument):
        self.document = document

    def get_text_document(self, _uri: str) -> _DummyDocument:
        return self.document


class _DummyLanguageServer:
    FIX_ALL_SPEC_MISMATCHES_COMMAND = "aiconls.fixAllSpecMismatches"

    def __init__(self, document: _DummyDocument):
        self.workspace = _DummyWorkspace(document)
        self.called = None
        self.spec: object | None = None

    def generate_component(self, name: str, diagnostic: types.Diagnostic):
        self.called = ("component", name, diagnostic.code)
        return "component-action"

    def generate_goal(self, name: str, diagnostic: types.Diagnostic):
        self.called = ("goal", name, diagnostic.code)
        return "goal-action"

    def generate_connection(self, name: str, diagnostic: types.Diagnostic):
        self.called = ("connection", name, diagnostic.code)
        return "connection-action"

    def generate_component_update(self, _uri: str, diagnostic: types.Diagnostic):
        self.called = ("component-update", diagnostic.code)
        return "component-update-action"

    def generate_spec_from_component_update(
        self, _uri: str, diagnostic: types.Diagnostic
    ):
        self.called = ("spec-only-component-shape", diagnostic.code)
        return "spec-only-component-shape-action"

    def generate_spec_from_component_update_and_propagate(
        self, _uri: str, diagnostic: types.Diagnostic
    ):
        self.called = ("propagate-component-shape", diagnostic.code)
        return "propagate-component-shape-action"

    def generate_connection_update(self, _uri: str, diagnostic: types.Diagnostic):
        self.called = ("connection-update", diagnostic.code)
        return "connection-update-action"

    def generate_build_connections_update(
        self, _uri: str, diagnostic: types.Diagnostic
    ):
        self.called = ("build-connections-update", diagnostic.code)
        return "build-connections-update-action"

    def fix_all_spec_mismatches(self, uri: str | None = None):
        self.called = ("fix-all", uri)

    def run_dynamic_check(self):
        self.called = ("run-dynamic-check", None)

    def refresh_document_if_stale(self, _document: _DummyDocument) -> None:
        return None


class _DummySpec:
    def get_component_instances_by_class_name(self, class_name: str) -> list[str]:
        if class_name == "TestComponent":
            return ["test_component_instance"]
        return []

    def get_goal_instances_by_class_name(self, class_name: str) -> list[str]:
        if class_name == "ReachGoal":
            return ["reach_goal_instance"]
        return []

    def get_connection_instances_by_class_name(self, class_name: str) -> list[str]:
        if class_name == "TestConnection":
            return ["test_connection_instance"]
        return []


def _make_params(code: str, line: str, name: str) -> types.CodeActionParams:
    start = line.index(name)
    end = start + len(name)
    severity = (
        types.DiagnosticSeverity.Warning
        if code.startswith("W")
        else types.DiagnosticSeverity.Error
    )
    diagnostic = types.Diagnostic(
        range=types.Range(
            start=types.Position(line=0, character=start),
            end=types.Position(line=0, character=end),
        ),
        message="missing implementation",
        code=code,
        severity=severity,
    )
    return types.CodeActionParams(
        text_document=types.TextDocumentIdentifier(uri="file:///spec.yaml"),
        range=diagnostic.range,
        context=types.CodeActionContext(diagnostics=[diagnostic]),
    )


def test_code_action_provider_generates_goal_quickfix():
    line = "  ReachGoal:"
    params = _make_params("W012", line, "ReachGoal")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "goal-action"
    assert actions[1].title == "Ignore this warning (W012)"
    assert ls.called == ("goal", "ReachGoal", "W012")


def test_code_action_provider_generates_connection_quickfix():
    line = "  testConnection:"
    params = _make_params("W013", line, "testConnection")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "connection-action"
    assert actions[1].title == "Ignore this warning (W013)"
    assert ls.called == ("connection", "testConnection", "W013")


def test_code_action_provider_generates_spec_mismatch_actions():
    line = '    self.cls_quantity_names_and_shapes = {"value": (2, 4)}'
    params = _make_params("E102", line, "cls_quantity_names_and_shapes")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 4
    assert actions[0] == "component-update-action"
    assert actions[1] == "spec-only-component-shape-action"
    assert actions[2] == "propagate-component-shape-action"

    fix_all_action = actions[3]
    assert fix_all_action.kind == types.CodeActionKind.SourceFixAll
    assert fix_all_action.command is not None
    assert (
        fix_all_action.command.command
        == AICONLanguageServer.FIX_ALL_SPEC_MISMATCHES_COMMAND
    )
    assert ls.called == ("propagate-component-shape", "E102")


def test_code_action_provider_generates_missing_quantities_action():
    line = "class KnownSensor(SensorComponent):"
    params = _make_params("W103", line, "KnownSensor")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "component-update-action"
    assert actions[1].title == "Ignore this warning (W103)"
    assert ls.called == ("component-update", "W103")


def test_code_action_provider_generates_connection_spec_mismatch_actions():
    line = '            {"value": (2, 4)},'
    params = _make_params("E201", line, "value")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "connection-update-action"

    fix_all_action = actions[1]
    assert fix_all_action.kind == types.CodeActionKind.SourceFixAll
    assert fix_all_action.command is not None
    assert (
        fix_all_action.command.command
        == AICONLanguageServer.FIX_ALL_SPEC_MISMATCHES_COMMAND
    )
    assert ls.called == ("connection-update", "E201")


def test_code_action_provider_generates_build_connections_action():
    line = "def build_connections():"
    params = _make_params("E203", line, "build_connections")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 1
    assert actions[0] == "build-connections-update-action"
    assert ls.called == ("build-connections-update", "E203")


def test_fix_all_spec_mismatches_handler_passes_uri_argument():
    ls = _DummyLanguageServer(_DummyDocument(""))
    fix_all_spec_mismatches(cast(AICONLanguageServer, ls), ["file:///component.py"])
    assert ls.called == ("fix-all", "file:///component.py")


def test_run_dynamic_check_handler_triggers_server_dynamic_check():
    ls = _DummyLanguageServer(_DummyDocument(""))
    run_dynamic_check(cast(AICONLanguageServer, ls))
    assert ls.called == ("run-dynamic-check", None)


def test_code_action_provider_adds_inline_ignore_warning_edit_at_line_end():
    line = "  ReachGoal: # existing comment"
    params = _make_params("W012", line, "ReachGoal")
    ls = _DummyLanguageServer(_DummyDocument(line))

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    ignore_action = actions[1]
    assert ignore_action.edit is not None
    assert ignore_action.edit.changes is not None
    edits = ignore_action.edit.changes["file:///spec.yaml"]
    assert len(edits) == 1
    edit = edits[0]
    assert edit.new_text == " # aiconls: ignore-warning W012"
    assert edit.range.start.line == 0
    assert edit.range.start.character == len(line)
    assert edit.range.start == edit.range.end


def test_code_action_provider_resolves_component_class_name_to_instance_key():
    line = "    class_name: TestComponent"
    params = _make_params("W011", line, "TestComponent")
    ls = _DummyLanguageServer(_DummyDocument(line))
    ls.spec = _DummySpec()

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "component-action"
    assert actions[1].title == "Ignore this warning (W011)"
    assert ls.called == ("component", "test_component_instance", "W011")


def test_code_action_provider_resolves_goal_class_name_to_instance_key():
    line = "    class_name: ReachGoal"
    params = _make_params("W012", line, "ReachGoal")
    ls = _DummyLanguageServer(_DummyDocument(line))
    ls.spec = _DummySpec()

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "goal-action"
    assert actions[1].title == "Ignore this warning (W012)"
    assert ls.called == ("goal", "reach_goal_instance", "W012")


def test_code_action_provider_resolves_connection_class_name_to_instance_key():
    line = "    class_name: TestConnection"
    params = _make_params("W013", line, "TestConnection")
    ls = _DummyLanguageServer(_DummyDocument(line))
    ls.spec = _DummySpec()

    actions = code_action_provider(cast(AICONLanguageServer, ls), params)

    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == "connection-action"
    assert actions[1].title == "Ignore this warning (W013)"
    assert ls.called == ("connection", "test_connection_instance", "W013")
