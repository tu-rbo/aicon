from pathlib import Path

from lsprotocol import types
from pytest import MonkeyPatch

from server import AICONLanguageServer


class _FailingQuickFixSpec:
    def generate_component(self, _project_root: str, _component_name: str):
        raise RuntimeError("component generation failed")

    def generate_goal(self, _project_root: str, _goal_name: str):
        raise RuntimeError("goal generation failed")

    def generate_connection(self, _project_root: str, _connection_name: str):
        raise RuntimeError("connection generation failed")


def _diagnostic(code: str) -> types.Diagnostic:
    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=1),
        ),
        message="missing implementation",
        code=code,
        severity=types.DiagnosticSeverity.Warning,
    )


def _spec_has_no_errors(_sink: object, _uri: str) -> bool:
    return False


def test_quickfix_generation_failures_are_logged_without_user_error_popup(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")
    ls.spec = _FailingQuickFixSpec()  # type: ignore[assignment]

    monkeypatch.setattr("server.spec_has_errors", _spec_has_no_errors)

    shown_messages: list[types.ShowMessageParams] = []
    logged_messages: list[types.LogMessageParams] = []

    monkeypatch.setattr(ls, "window_show_message", shown_messages.append)
    monkeypatch.setattr(ls, "window_log_message", logged_messages.append)

    assert ls.generate_component("component", _diagnostic("W011")) is None
    assert ls.generate_goal("goal", _diagnostic("W012")) is None
    assert ls.generate_connection("connection", _diagnostic("W013")) is None

    assert shown_messages == []
    assert len(logged_messages) == 3
    logged_text = [entry.message for entry in logged_messages]
    assert "Failed to generate component code action" in logged_text[0]
    assert "Failed to generate goal code action" in logged_text[1]
    assert "Failed to generate connection code action" in logged_text[2]
