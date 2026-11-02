from pathlib import Path
from typing import cast

from lsprotocol import types
from pygls.workspace import TextDocument
from pytest import MonkeyPatch

from server import AICONLanguageServer


class _FakeSpecDocument:
    def __init__(self, path: Path, source: str = ""):
        self.path = str(path)
        self.uri = path.as_uri()
        self.source = source
        self.version: int | None = 1


class _FakePythonDocument:
    def __init__(self, path: Path, source: str = "class Example:\n    pass\n"):
        self.path = str(path)
        self.uri = path.as_uri()
        self.source = source
        self.version: int | None = 1


def test_server_init_does_not_raise_if_project_root_does_not_exist(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing-project-root"
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(missing_path))

    ls = AICONLanguageServer("diagnostic-server", "v1")
    assert ls.is_project_root_available() is False


def test_server_project_root_error_notification_is_user_visible(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing-project-root"
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(missing_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    shown_messages: list[str] = []
    logged_messages: list[str] = []

    def _show_message(params: types.ShowMessageParams) -> None:
        shown_messages.append(params.message)

    def _log_message(params: types.LogMessageParams) -> None:
        logged_messages.append(params.message)

    monkeypatch.setattr(ls, "window_show_message", _show_message)
    monkeypatch.setattr(ls, "window_log_message", _log_message)

    ls.notify_project_root_unavailable()

    assert len(shown_messages) == 1
    assert len(logged_messages) == 1
    assert "does not exist" in shown_messages[0]


def test_parse_uses_same_editor_spec_document_for_update_and_validate(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    calls: list[tuple[str, object | None]] = []
    monkeypatch.setattr(ls, "_mark_open_python_files_stale", lambda: None)

    def _update_spec(document: TextDocument | None = None) -> None:
        calls.append(("update", document))

    def _validate_spec(document: TextDocument | None = None) -> None:
        calls.append(("validate", document))

    monkeypatch.setattr(ls, "update_spec", _update_spec)
    monkeypatch.setattr(ls, "validate_spec", _validate_spec)

    spec_path = tmp_path / "spec.yaml"
    spec_document = _FakeSpecDocument(spec_path)

    ls.parse(cast(TextDocument, spec_document))

    assert calls == [("update", spec_document), ("validate", spec_document)]


def test_parse_updates_spec_before_marking_python_files_stale(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    call_order: list[str] = []

    def _update_spec(document: TextDocument | None = None) -> None:
        call_order.append("update")

    def _mark_open_python_files_stale() -> None:
        call_order.append("mark")

    def _validate_spec(document: TextDocument | None = None) -> None:
        call_order.append("validate")

    monkeypatch.setattr(ls, "update_spec", _update_spec)
    monkeypatch.setattr(
        ls, "_mark_open_python_files_stale", _mark_open_python_files_stale
    )
    monkeypatch.setattr(ls, "validate_spec", _validate_spec)

    spec_path = tmp_path / "spec.yaml"
    spec_document = _FakeSpecDocument(spec_path)

    ls.parse(cast(TextDocument, spec_document))

    assert call_order == ["update", "mark", "validate"]


def test_parse_does_not_reparse_opened_files_when_updated_spec_has_errors(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    call_order: list[str] = []

    def _update_spec(document: TextDocument | None = None) -> None:
        del document
        call_order.append("update")
        ls.spec_parse_diagnostics = [
            types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=0, character=0),
                    end=types.Position(line=0, character=1),
                ),
                severity=types.DiagnosticSeverity.Error,
                message="Malformed spec",
            )
        ]

    def _mark_open_python_files_stale() -> None:
        call_order.append("mark")

    def _validate_spec(document: TextDocument | None = None) -> None:
        del document
        call_order.append("validate")

    monkeypatch.setattr(ls, "update_spec", _update_spec)
    monkeypatch.setattr(
        ls, "_mark_open_python_files_stale", _mark_open_python_files_stale
    )
    monkeypatch.setattr(ls, "validate_spec", _validate_spec)

    spec_document = _FakeSpecDocument(tmp_path / "spec.yaml")
    ls.parse(cast(TextDocument, spec_document))

    assert call_order == ["update", "validate"]


def test_validate_spec_refreshes_from_current_editor_spec(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    spec_path = tmp_path / "spec.yaml"
    spec_document = _FakeSpecDocument(spec_path, source="components:\n")

    def _set_diagnostics(
        _uri: str, _version: int, _diagnostics: list[types.Diagnostic]
    ) -> None:
        return None

    monkeypatch.setattr(ls, "set_diagnostics", _set_diagnostics)

    updated: list[object] = []

    def _update_spec(document: TextDocument | None = None) -> None:
        updated.append(document)

    monkeypatch.setattr(ls, "update_spec", _update_spec)

    ls.validate_spec(cast(TextDocument, spec_document))

    assert updated == [spec_document]


def test_parse_tracks_open_python_documents_by_uri(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    python_document = _FakePythonDocument(tmp_path / "sensor.py")

    def _set_diagnostics(
        _uri: str, _version: int, _diagnostics: list[types.Diagnostic]
    ) -> None:
        return None

    monkeypatch.setattr(ls, "set_diagnostics", _set_diagnostics)

    ls.parse(cast(TextDocument, python_document))

    assert python_document.uri in ls.opened_files
    assert python_document.path not in ls.opened_files


def test_mark_open_python_files_stale_marks_only_open_python_documents(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    python_document = _FakePythonDocument(tmp_path / "sensor.py")
    spec_document = _FakeSpecDocument(tmp_path / "spec.yaml")
    ls.opened_files = {python_document.uri, spec_document.uri, "file:///note.txt"}

    def _update_spec(document: TextDocument | None = None) -> None:
        del document

    def _validate_spec(document: TextDocument | None = None) -> None:
        del document

    monkeypatch.setattr(ls, "update_spec", _update_spec)
    monkeypatch.setattr(ls, "validate_spec", _validate_spec)

    ls.parse(cast(TextDocument, spec_document))

    assert ls.stale_files == {python_document.uri}


def test_refresh_document_if_stale_reparses_document_once(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    python_document = _FakePythonDocument(tmp_path / "python" / "sensor.py")
    ls.stale_files = {python_document.uri}
    reparsed: list[str] = []

    def _parse(document: TextDocument) -> None:
        reparsed.append(document.uri)

    monkeypatch.setattr(ls, "parse", _parse)

    ls.refresh_document_if_stale(cast(TextDocument, python_document))
    ls.stale_files.discard(python_document.uri)
    ls.refresh_document_if_stale(cast(TextDocument, python_document))

    assert reparsed == [python_document.uri]


def test_update_spec_notifies_once_when_spec_is_missing(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    shown_messages: list[str] = []
    logged_messages: list[str] = []

    def _show_message(params: types.ShowMessageParams) -> None:
        shown_messages.append(params.message)

    def _log_message(params: types.LogMessageParams) -> None:
        logged_messages.append(params.message)

    monkeypatch.setattr(ls, "window_show_message", _show_message)
    monkeypatch.setattr(ls, "window_log_message", _log_message)

    ls.update_spec()
    ls.update_spec()

    assert ls.spec is None
    assert len(shown_messages) == 1
    assert len(logged_messages) == 1
    assert "Waiting for spec.yaml to be created." in shown_messages[0]


def test_parse_reports_spec_errors_instead_of_missing_spec(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "components:\n"
        "  Example:\n"
        "    component_type: Sensor\n"
        "    quantities:\n"
        "      value:\n"
        "        shape: (-2,)\n",
        encoding="utf-8",
    )
    python_document = _FakePythonDocument(tmp_path / "component.py")

    shown_messages: list[str] = []
    published: dict[str, list[types.Diagnostic]] = {}

    def _show_message(params: types.ShowMessageParams) -> None:
        shown_messages.append(params.message)

    def _log_message(_params: types.LogMessageParams) -> None:
        return None

    def _set_diagnostics(
        uri: str, _version: int, diagnostics: list[types.Diagnostic]
    ) -> None:
        published[uri] = diagnostics

    monkeypatch.setattr(ls, "window_show_message", _show_message)
    monkeypatch.setattr(ls, "window_log_message", _log_message)
    monkeypatch.setattr(ls, "set_diagnostics", _set_diagnostics)

    ls.parse(cast(TextDocument, python_document))
    ls.parse(cast(TextDocument, python_document))

    assert ls.spec is None
    assert len(shown_messages) == 1
    assert "contains errors" in shown_messages[0]
    assert published[spec_path.as_uri()]


def test_parse_retries_spec_load_after_spec_file_is_created(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")

    spec_path = tmp_path / "spec.yaml"
    python_document = _FakePythonDocument(tmp_path / "component.py")

    def _set_diagnostics(
        _uri: str, _version: int, _diagnostics: list[types.Diagnostic]
    ) -> None:
        return None

    monkeypatch.setattr(ls, "set_diagnostics", _set_diagnostics)

    ls.parse(cast(TextDocument, python_document))
    assert ls.spec is None

    spec_path.write_text(
        "components: {}\ngoals: {}\nconnections: {}\n",
        encoding="utf-8",
    )
    ls.parse(cast(TextDocument, python_document))

    assert ls.spec is not None
