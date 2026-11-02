import asyncio
from pathlib import Path

from lsprotocol import types
from pytest import MonkeyPatch

from server import AICONLanguageServer


class _FakeSpecDocument:
    def __init__(self, path: Path, source: str = ""):
        self.path = str(path)
        self.uri = path.as_uri()
        self.source = source
        self.version: int | None = 1


class _FakeWorkspace:
    def __init__(self, spec_document: _FakeSpecDocument):
        self._spec_document = spec_document

    def get_text_document(self, _uri: str) -> _FakeSpecDocument:
        return self._spec_document


class _SkeletonSpec:
    def __init__(self, skeleton_files: list[Path]):
        self.skeleton_files = skeleton_files
        self.generate_skeleton_calls = 0

    def get_skeleton_file_paths(self, _project_directory: str) -> list[Path]:
        return self.skeleton_files

    def generate_skeleton(self, _project_directory: str) -> list[str]:
        self.generate_skeleton_calls += 1
        return [str(path) for path in self.skeleton_files]


def _spec_has_no_errors(_sink: object, _uri: str) -> bool:
    return False


def test_generate_files_requests_confirmation_and_cancels_on_decline(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")
    spec_document = _FakeSpecDocument(tmp_path / "spec.yaml", source="components: {}\n")
    monkeypatch.setattr(ls.protocol, "_workspace", _FakeWorkspace(spec_document))

    existing_file = tmp_path / "actions.py"
    existing_file.write_text("# existing", encoding="utf-8")
    skeleton_spec = _SkeletonSpec([existing_file])
    ls.spec = skeleton_spec  # type: ignore[assignment]

    monkeypatch.setattr("server.spec_has_errors", _spec_has_no_errors)

    shown_messages: list[types.ShowMessageParams] = []

    def _show_message(params: types.ShowMessageParams) -> None:
        shown_messages.append(params)

    async def _show_message_request(
        _params: types.ShowMessageRequestParams,
    ) -> types.MessageActionItem:
        return types.MessageActionItem(title="Cancel")

    monkeypatch.setattr(ls, "window_show_message", _show_message)
    monkeypatch.setattr(ls, "window_show_message_request_async", _show_message_request)

    asyncio.run(ls.generate_files())

    assert skeleton_spec.generate_skeleton_calls == 0
    assert any("cancelled" in message.message.lower() for message in shown_messages)


def test_generate_files_overwrites_when_user_confirms(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(tmp_path))
    ls = AICONLanguageServer("diagnostic-server", "v1")
    spec_document = _FakeSpecDocument(tmp_path / "spec.yaml", source="components: {}\n")
    monkeypatch.setattr(ls.protocol, "_workspace", _FakeWorkspace(spec_document))

    existing_file = tmp_path / "actions.py"
    existing_file.write_text("# existing", encoding="utf-8")
    skeleton_spec = _SkeletonSpec([existing_file])
    ls.spec = skeleton_spec  # type: ignore[assignment]

    monkeypatch.setattr("server.spec_has_errors", _spec_has_no_errors)

    async def _show_message_request(
        _params: types.ShowMessageRequestParams,
    ) -> types.MessageActionItem:
        return types.MessageActionItem(title="Overwrite")

    monkeypatch.setattr(ls, "window_show_message_request_async", _show_message_request)

    asyncio.run(ls.generate_files())

    assert skeleton_spec.generate_skeleton_calls == 1
