from typing import Any, cast

from lsprotocol import types

from handlers import did_change, did_open, initialize
from server import AICONLanguageServer


class _DummyDocument:
    def __init__(self, uri: str):
        self.uri = uri
        self.path = uri.replace("file://", "")


class _DummyWorkspace:
    def __init__(self, document: _DummyDocument):
        self.document = document

    def get_text_document(self, _uri: str) -> _DummyDocument:
        return self.document


class _DummyLanguageServer:
    def __init__(self, document: _DummyDocument) -> None:
        self.workspace = _DummyWorkspace(document)
        self.calls: list[tuple[str, Any]] = []
        self.project_root_available = True
        self.spec: object | None = object()

    def parse(self, document: _DummyDocument, **kwargs: Any) -> None:
        self.calls.append(("parse", (document, kwargs)))

    def clear_dynamic_diagnostics(self) -> None:
        self.calls.append(("clear-dynamic", None))

    def parse_all_project_python_files(self) -> None:
        self.calls.append(("parse-all", None))

    def validate_spec(self, document: _DummyDocument | None = None) -> None:
        self.calls.append(("validate", document))

    def is_project_root_available(self) -> bool:
        return self.project_root_available

    def notify_project_root_unavailable(self) -> None:
        self.calls.append(("notify-project-root-unavailable", None))

    def update_spec(self) -> None:
        self.calls.append(("update-spec", None))


def _open_params(uri: str) -> types.DidOpenTextDocumentParams:
    return types.DidOpenTextDocumentParams(
        text_document=types.TextDocumentItem(
            uri=uri,
            language_id="python",
            version=1,
            text="",
        )
    )


def test_did_open_spec_parses_all_project_python_files_before_validation() -> None:
    spec_uri = "file:///project/spec.yaml"
    document = _DummyDocument(spec_uri)
    ls = _DummyLanguageServer(document)

    did_open(cast(AICONLanguageServer, ls), _open_params(spec_uri))

    assert ls.calls == [
        ("parse", (document, {"validate_spec_after": False})),
        ("parse-all", None),
        ("validate", document),
    ]


def test_did_open_python_file_parses_only_opened_document() -> None:
    python_uri = "file:///project/component.py"
    document = _DummyDocument(python_uri)
    ls = _DummyLanguageServer(document)

    did_open(cast(AICONLanguageServer, ls), _open_params(python_uri))

    assert ls.calls == [("parse", (document, {}))]


def test_did_change_clears_dynamic_diagnostics_before_parsing() -> None:
    python_uri = "file:///project/component.py"
    document = _DummyDocument(python_uri)
    ls = _DummyLanguageServer(document)

    did_change(cast(AICONLanguageServer, ls), _open_params(python_uri))

    assert ls.calls == [
        ("clear-dynamic", None),
        ("parse", (document, {})),
    ]


def test_initialize_notifies_user_when_project_root_is_unavailable() -> None:
    document = _DummyDocument("file:///project/spec.yaml")
    ls = _DummyLanguageServer(document)
    ls.project_root_available = False

    initialize(cast(AICONLanguageServer, ls), types.InitializedParams())

    assert ls.calls == [("notify-project-root-unavailable", None)]


def test_initialize_scans_project_after_loading_spec() -> None:
    document = _DummyDocument("file:///project/spec.yaml")
    ls = _DummyLanguageServer(document)

    initialize(cast(AICONLanguageServer, ls), types.InitializedParams())

    assert ls.calls == [
        ("update-spec", None),
        ("parse-all", None),
        ("validate", None),
    ]


def test_initialize_skips_project_scan_without_spec() -> None:
    document = _DummyDocument("file:///project/spec.yaml")
    ls = _DummyLanguageServer(document)
    ls.spec = None

    initialize(cast(AICONLanguageServer, ls), types.InitializedParams())

    assert ls.calls == [("update-spec", None)]
