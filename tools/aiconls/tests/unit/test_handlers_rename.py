from typing import cast

from lsprotocol import types

from handlers import rename_quantity
from server import AICONLanguageServer


class _DummyDocument:
    def __init__(self, uri: str):
        self.uri = uri


class _DummyWorkspace:
    def __init__(self, document: _DummyDocument):
        self.document = document

    def get_text_document(self, _uri: str) -> _DummyDocument:
        return self.document


class _DummyLanguageServer:
    def __init__(self, uri: str):
        self.workspace = _DummyWorkspace(_DummyDocument(uri))
        self.refreshed = False
        self.last_call: tuple[str, types.Position, str] | None = None

    def refresh_document_if_stale(self, _document: _DummyDocument) -> None:
        self.refreshed = True

    def build_quantity_rename_edit(
        self, uri: str, position: types.Position, new_name: str
    ) -> types.WorkspaceEdit:
        self.last_call = (uri, position, new_name)
        return types.WorkspaceEdit(changes={})


def test_rename_quantity_handler_uses_server_builder() -> None:
    uri = "file:///component.py"
    params = types.RenameParams(
        text_document=types.TextDocumentIdentifier(uri=uri),
        position=types.Position(line=1, character=2),
        new_name="renamed_value",
    )
    ls = _DummyLanguageServer(uri)

    edit = rename_quantity(cast(AICONLanguageServer, ls), params)

    assert edit is not None
    assert ls.refreshed
    assert ls.last_call == (uri, types.Position(line=1, character=2), "renamed_value")
