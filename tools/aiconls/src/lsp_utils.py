import importlib
import logging
import os
import re
from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import libcst as cst
from libcst.metadata import PositionProvider
from lsprotocol import types
from pygls.workspace import TextDocument

from reader.aicon_files.util import get_node_position

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
_LOGGED_EXCEPTION_IDS: set[int] = set()


def configure_logging() -> None:
    log_file = Path(
        os.getenv(
            "AICONLS_LOG_FILE",
            str(Path.home() / "Library" / "Logs" / "aiconls" / "server.log"),
        )
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s:%(lineno)d %(funcName)s: %(message)s"
        )
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("server: %(levelname)s: %(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stderr_handler)
    logger.error("Server log file: %s", log_file)


class _DebugpyModule(Protocol):
    def listen(self, __address: tuple[str, int], /) -> None: ...
    def wait_for_client(self) -> None: ...


def configure_debugpy_from_env() -> None:
    if os.getenv("AICONLS_DEBUG", "0") != "1":
        return

    debugpy = cast(_DebugpyModule, importlib.import_module("debugpy"))

    host = os.getenv("AICONLS_DEBUG_HOST", "127.0.0.1")
    port = int(os.getenv("AICONLS_DEBUG_PORT", "5678"))
    wait_for_client = os.getenv("AICONLS_DEBUG_WAIT_FOR_CLIENT", "1") == "1"

    debugpy.listen((host, port))
    if wait_for_client:
        debugpy.wait_for_client()


class DiagnosticSink:
    def __init__(self) -> None:
        self.diagnostics: dict[str, tuple[int, list[types.Diagnostic]]] = {}

    def report_diagnostic(
        self, diagnostic: types.Diagnostic, uri: str, version: int
    ) -> None:
        if diagnostic in self.diagnostics:
            self.diagnostics[uri][1].append(diagnostic)
        else:
            self.diagnostics[uri] = (version, [diagnostic])

    def set_diagnostics(
        self, diagnostics: list[types.Diagnostic], uri: str, version: int
    ):
        self.diagnostics[uri] = (version, diagnostics)


def is_spec_yaml(path: str) -> bool:
    return path.split(".")[-1] == "yaml"


def is_python_file(path: str) -> bool:
    return path.split(".")[-1] == "py"


def get_document_content(document: TextDocument) -> str:
    source = getattr(document, "source", None)
    if source is None:
        source = getattr(document, "text", None)
    if source is None:
        raise RuntimeError("Unable to access document contents")
    return str(source)


def mark_exception_logged(exc: BaseException) -> None:
    _LOGGED_EXCEPTION_IDS.add(id(exc))


def take_exception_logged(exc: BaseException) -> bool:
    exception_id = id(exc)
    if exception_id not in _LOGGED_EXCEPTION_IDS:
        return False
    _LOGGED_EXCEPTION_IDS.remove(exception_id)
    return True


def is_quantity_completion_line(line: str, character: int) -> bool:
    prefix = line[:character]
    return re.search(r'self\.quantities\[\s*["\']?[^"\']*$', prefix) is not None


def is_connection_completion_line(line: str, character: int) -> bool:
    prefix = line[:character]
    return re.search(r'self\.connections\[\s*["\']?[^"\']*$', prefix) is not None


def _match_connection_access(line: str, character: int) -> re.Match[str] | None:
    prefix = line[:character]
    return re.search(r'self\.connections\[\s*["\']([^"\']+)["\']\]\.', prefix)


def is_connection_member_completion_line(line: str, character: int) -> bool:
    prefix = line[:character]
    return (
        re.search(r'self\.connections\[\s*["\']([^"\']+)["\']\]\.[A-Za-z_]*$', prefix)
        is not None
    )


def get_connection_name_from_member_completion(line: str, character: int) -> str | None:
    match = _match_connection_access(line, character)
    if match is None:
        return None
    return match.group(1)


def is_connected_quantities_completion_line(line: str, character: int) -> bool:
    prefix = line[:character]
    return (
        re.search(
            r'self\.connections\[\s*["\']([^"\']+)["\']\]\.connected_quantities\[\s*["\']?[^"\']*$',
            prefix,
        )
        is not None
    )


def is_self_connected_quantities_completion_line(line: str, character: int) -> bool:
    prefix = line[:character]
    return (
        re.search(
            r'self\.connected_quantities\[\s*["\']?[^"\']*["\']?\]?$',
            prefix,
        )
        is not None
    )


def get_connection_name_from_connected_quantities_completion(
    line: str, character: int
) -> str | None:
    match = re.search(
        r'self\.connections\[\s*["\']([^"\']+)["\']\]\.connected_quantities\[',
        line[:character],
    )
    if match is None:
        return None
    return match.group(1)


def position_in_range(position: types.Position, location: types.Range) -> bool:
    if position.line < location.start.line or position.line > location.end.line:
        return False

    if (
        position.line == location.start.line
        and position.character < location.start.character
    ):
        return False

    return not (
        position.line == location.end.line
        and position.character > location.end.character
    )


def get_diagnostic_from_Code_Action(params: types.CodeActionParams, error_code: str):
    for diagnostic in params.context.diagnostics:
        if diagnostic.code == error_code:
            return diagnostic


class EnclosingComponentFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, position: types.Position, component_names: set[str]) -> None:
        self.position = position
        self.component_names = component_names
        self.component_name: str | None = None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if node.name.value not in self.component_names:
            return

        class_range = get_node_position(self, node)
        if position_in_range(self.position, class_range):
            self.component_name = node.name.value


def log_lsp_exceptions(handler: F) -> F:
    if iscoroutinefunction(handler):

        @wraps(handler)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await handler(*args, **kwargs)
            except Exception as exc:
                if take_exception_logged(exc):
                    raise
                logger.exception("Unhandled exception in %s", handler.__name__)
                raise

        return async_wrapper  # type: ignore[return-value]

    @wraps(handler)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return handler(*args, **kwargs)
        except Exception as exc:
            if take_exception_logged(exc):
                raise
            logger.exception("Unhandled exception in %s", handler.__name__)
            raise

    return wrapper  # type: ignore[return-value]
