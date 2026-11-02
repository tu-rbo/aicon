# pyright: reportPrivateUsage=false

from pathlib import Path
from typing import cast

from lsprotocol import types
from pytest import MonkeyPatch

import server as server_module
from server import AICONLanguageServer


class _FakePythonDocument:
    def __init__(self, path: Path, source: str):
        self.path = str(path)
        self.uri = path.as_uri()
        self.source = source
        self.version: int | None = 7


class _FakeWorkspace:
    def __init__(self, python_document: _FakePythonDocument):
        self._python_document = python_document

    def get_text_document(self, _uri: str) -> _FakePythonDocument:
        return self._python_document


def _make_diagnostic(
    *,
    message: str,
    code: str,
    source: str,
    severity: types.DiagnosticSeverity = types.DiagnosticSeverity.Error,
) -> types.Diagnostic:
    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=1),
        ),
        message=message,
        code=code,
        severity=severity,
        source=source,
    )


def test_parse_all_project_python_files_prefers_open_editor_document(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    python_file = tmp_path / "component.py"
    python_file.write_text("class OnDisk:\n    pass\n", encoding="utf-8")
    ls = AICONLanguageServer("diagnostic-server", "v1")

    open_document = _FakePythonDocument(
        python_file,
        source="class UnsavedEditorVersion:\n    pass\n",
    )
    monkeypatch.setattr(ls.protocol, "_workspace", _FakeWorkspace(open_document))
    ls.opened_files = {python_file.as_uri()}

    parse_calls: list[object] = []

    def _parse(document: object, **kwargs: object) -> None:
        parse_calls.append((document, kwargs))

    monkeypatch.setattr(ls, "parse", _parse)

    ls.parse_all_project_python_files()

    assert len(parse_calls) == 1
    parsed_document, parsed_kwargs = cast(
        tuple[object, dict[str, object]], parse_calls[0]
    )
    assert parsed_document is open_document
    assert parsed_kwargs == {
        "validate_spec_after": False,
        "track_open_document": False,
    }


def test_run_dynamic_check_publishes_snapshot_diagnostics_and_summary(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "spec.yaml").write_text("components: {}\n", encoding="utf-8")

    ls = AICONLanguageServer("diagnostic-server", "v1")
    dynamic_target_uri = (tmp_path / "estimator.py").as_uri()
    stale_uri = (tmp_path / "stale_dynamic.py").as_uri()

    published: list[types.PublishDiagnosticsParams] = []
    shown_messages: list[types.ShowMessageParams] = []
    logged_messages: list[types.LogMessageParams] = []

    monkeypatch.setattr(ls, "text_document_publish_diagnostics", published.append)
    monkeypatch.setattr(ls, "window_show_message", shown_messages.append)
    monkeypatch.setattr(ls, "window_log_message", logged_messages.append)
    monkeypatch.setattr(ls, "parse_all_project_python_files", lambda: None)

    def _validate_spec(document: object | None = None) -> None:
        del document

    monkeypatch.setattr(ls, "validate_spec", _validate_spec)

    stale_static = _make_diagnostic(
        message="Static warning",
        code="W101",
        source="aiconls",
        severity=types.DiagnosticSeverity.Warning,
    )
    stale_dynamic = _make_diagnostic(
        message="Stale dynamic issue",
        code="E108",
        source=AICONLanguageServer.DYNAMIC_DIAGNOSTIC_SOURCE,
    )
    ls.set_diagnostics(stale_uri, 1, [stale_static, stale_dynamic])
    ls._dynamic_diagnostics_by_uri = {stale_uri: [stale_dynamic]}

    static_target = _make_diagnostic(
        message="Static baseline issue",
        code="E201",
        source="aiconls",
    )
    ls.set_diagnostics(dynamic_target_uri, 3, [static_target])

    def _fake_dynamic_analysis(
        _spec: object,
        _components: object,
        _goals: object,
        _connections: object,
    ) -> dict[str, list[types.Diagnostic]]:
        return {
            dynamic_target_uri: [
                _make_diagnostic(
                    message="f_func is not differentiable.",
                    code="E108",
                    source="aiconls",
                )
            ]
        }

    monkeypatch.setattr(server_module, "dynamic_analysis", _fake_dynamic_analysis)

    ls.run_dynamic_check()

    stale_current = ls.diagnostic_sink.diagnostics[stale_uri][1]
    assert stale_current == [stale_static]

    target_current = ls.diagnostic_sink.diagnostics[dynamic_target_uri][1]
    assert len(target_current) == 2
    assert static_target in target_current
    dynamic_entries = [
        diagnostic
        for diagnostic in target_current
        if diagnostic.source == AICONLanguageServer.DYNAMIC_DIAGNOSTIC_SOURCE
    ]
    assert len(dynamic_entries) == 1
    assert dynamic_entries[0].code == "E108"

    assert list(ls._dynamic_diagnostics_by_uri.keys()) == [dynamic_target_uri]
    assert len(ls._dynamic_diagnostics_by_uri[dynamic_target_uri]) == 1

    assert any(
        "Dynamic check completed: 1 issue(s) in 1 file(s)." in message.message
        for message in shown_messages
    )
    assert any(
        "estimator.py:1:1 [E108] f_func is not differentiable." in message.message
        for message in logged_messages
    )
    assert published
