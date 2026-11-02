import keyword
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import unquote, urlparse

import libcst as cst
import libcst.matchers as m
import yaml
from libcst.metadata import MetadataWrapper, PositionProvider
from lsprotocol import types
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument
from yaml.nodes import MappingNode, Node, ScalarNode

from diagnostics.diagnostics import ErrorCode
from diagnostics.warning_ignore_annotations import (
    filter_diagnostics,
    parse_global_diagnostic_ignore_annotations,
)
from dynamic.analyzer import dynamic_analysis
from generator.connection import gen_connection_head
from generator.goal import gen_goal_head
from lsp_utils import (
    DiagnosticSink,
    EnclosingComponentFinder,
    get_document_content,
    is_python_file,
    is_spec_yaml,
    mark_exception_logged,
    position_in_range,
)
from project_scanner import (
    add_imported_classes,
    drop_unused_quantity_warnings_used_by_components,
)
from reader.aicon_files.aicon_reader import AICONComponentParser
from reader.aicon_files.reader_types import (
    ParsedAction,
    ParsedComponent,
    ParsedConnection,
    ParsedEstimator,
    ParsedGoal,
    ParsedQuantities,
    ParsedSensor,
)
from reader.aicon_files.util import get_node_position, get_simple_string
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import SpecComponentQuantityGeneric, SpecQuantities
from server_helpers import (
    build_full_document_range,
    build_generation_workspace_edit,
    create_component_updater,
    get_component_header,
    notify_spec_malformed,
    spec_has_errors,
)
from updater.build_connections_updater import BuildConnectionsUpdater
from updater.connection_updater import ConnectionUpdater
from validator.spec_validator import SpecReaderValidator

logger = logging.getLogger(__name__)
EXCLUDED_PROJECT_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


# Parameters the framework passes to f_func/connection_func/goal_func besides quantities.
_NON_QUANTITY_PARAMETERS = frozenset({"dt"})


@dataclass(frozen=True)
class _QuantityOccurrence:
    name: str
    position: types.Range
    is_identifier: bool
    python_quote: str | None = None


class _QuantityOccurrenceCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.occurrences: list[_QuantityOccurrence] = []
        self._inside_define_goal_cost_function = False
        self._inside_define_implicit_connection_function = False
        self._inside_define_estimation_function_f = False
        self._inside_goal_func = False
        self._inside_connection_func = False
        self._inside_f_func = False
        self._goal_func_quantity_names: set[str] = set()
        self._connection_func_quantity_names: set[str] = set()
        self._f_func_quantity_names: set[str] = set()

    def _record(
        self,
        node: cst.CSTNode,
        quantity_name: str,
        *,
        is_identifier: bool,
        python_quote: str | None = None,
    ) -> None:
        self.occurrences.append(
            _QuantityOccurrence(
                name=quantity_name,
                position=get_node_position(self, node),
                is_identifier=is_identifier,
                python_quote=python_quote,
            )
        )

    def _extract_python_quote(self, value: str) -> str | None:
        for index, char in enumerate(value):
            if char not in {"'", '"'}:
                continue
            if value[index : index + 3] == char * 3:
                return char
            return char
        return None

    def _extract_simple_string_subscript_key(
        self, node: cst.Subscript
    ) -> tuple[str, cst.SimpleString] | None:
        if len(node.slice) != 1:
            return None
        index = node.slice[0].slice
        if not isinstance(index, cst.Index):
            return None
        if not isinstance(index.value, cst.SimpleString):
            return None
        return get_simple_string(index.value), index.value

    def _record_dict_keys(self, dict_node: cst.Dict) -> None:
        for element in dict_node.elements:
            dict_element = cst.ensure_type(element, cst.DictElement)
            if not isinstance(dict_element.key, cst.SimpleString):
                continue
            self._record(
                dict_element.key,
                get_simple_string(dict_element.key),
                is_identifier=False,
                python_quote=self._extract_python_quote(dict_element.key.value),
            )

    def _record_string_elements(self, sequence_node: cst.BaseExpression) -> None:
        if not isinstance(sequence_node, (cst.List, cst.Tuple)):
            return
        for element in sequence_node.elements:
            if not isinstance(element, cst.Element):
                continue
            if not isinstance(element.value, cst.SimpleString):
                continue
            self._record(
                element.value,
                get_simple_string(element.value),
                is_identifier=False,
                python_quote=self._extract_python_quote(element.value.value),
            )

    def _record_function_parameters(self, node: cst.FunctionDef) -> set[str]:
        quantity_names: set[str] = set()
        for parameter in (
            list(node.params.posonly_params)
            + list(node.params.params)
            + list(node.params.kwonly_params)
        ):
            quantity_name = parameter.name.value
            if quantity_name in _NON_QUANTITY_PARAMETERS:
                continue
            quantity_names.add(quantity_name)
            self._record(parameter.name, quantity_name, is_identifier=True)
        return quantity_names

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        target = node.targets[0].target
        if (
            m.matches(target, m.Name("cls_quantity_names_and_shapes"))
            or m.matches(
                target,
                m.Attribute(
                    value=m.Name("self"),
                    attr=m.Name("cls_quantity_names_and_shapes"),
                ),
            )
        ) and isinstance(node.value, cst.Dict):
            self._record_dict_keys(node.value)
        return None

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        # the annotated form: cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {...}
        if m.matches(
            node.target, m.Name("cls_quantity_names_and_shapes")
        ) and isinstance(node.value, cst.Dict):
            self._record_dict_keys(node.value)
        return None

    def visit_Call(self, node: cst.Call) -> bool | None:
        if not m.matches(
            node.func,
            m.Attribute(
                value=m.Call(func=m.Name("super")),
                attr=m.Name("__init__"),
            ),
        ):
            return None
        if len(node.args) < 2:
            return None
        quantities_node = node.args[1].value
        if isinstance(quantities_node, cst.Dict):
            self._record_dict_keys(quantities_node)
        return None

    def visit_Subscript(self, node: cst.Subscript) -> bool | None:
        key = self._extract_simple_string_subscript_key(node)
        if key is None:
            return None
        quantity_name, quantity_name_node = key

        if m.matches(
            node.value,
            m.Attribute(
                value=m.Name("self"),
                attr=m.Name("quantities"),
            ),
        ):
            self._record(
                quantity_name_node,
                quantity_name,
                is_identifier=False,
                python_quote=self._extract_python_quote(quantity_name_node.value),
            )
            return None

        if not m.matches(
            node.value,
            m.Attribute(
                value=m.Subscript(),
                attr=m.Name("connected_quantities"),
            ),
        ):
            return None

        connected_quantities_base = cst.ensure_type(node.value, cst.Attribute).value
        if not m.matches(
            connected_quantities_base,
            m.Subscript(
                value=m.Attribute(
                    value=m.Name("self"),
                    attr=m.Name("connections"),
                )
            ),
        ):
            return None

        self._record(
            quantity_name_node,
            quantity_name,
            is_identifier=False,
            python_quote=self._extract_python_quote(quantity_name_node.value),
        )
        return None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value == "define_goal_cost_function":
            self._inside_define_goal_cost_function = True
            return None

        if node.name.value == "define_implicit_connection_function":
            self._inside_define_implicit_connection_function = True
            return None

        if node.name.value == "define_estimation_function_f":
            self._inside_define_estimation_function_f = True
            return None

        if self._inside_define_goal_cost_function and node.name.value == "goal_func":
            self._inside_goal_func = True
            self._goal_func_quantity_names = self._record_function_parameters(node)
            return None

        if (
            self._inside_define_implicit_connection_function
            and node.name.value == "connection_func"
        ):
            self._inside_connection_func = True
            self._connection_func_quantity_names = self._record_function_parameters(
                node
            )
            return None

        if self._inside_define_estimation_function_f and node.name.value == "f_func":
            self._inside_f_func = True
            self._f_func_quantity_names = self._record_function_parameters(node)
            return None

        return None

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if original_node.name.value == "goal_func":
            self._inside_goal_func = False
            self._goal_func_quantity_names = set()

        if original_node.name.value == "connection_func":
            self._inside_connection_func = False
            self._connection_func_quantity_names = set()

        if original_node.name.value == "f_func":
            self._inside_f_func = False
            self._f_func_quantity_names = set()

        if original_node.name.value == "define_goal_cost_function":
            self._inside_define_goal_cost_function = False

        if original_node.name.value == "define_implicit_connection_function":
            self._inside_define_implicit_connection_function = False

        if original_node.name.value == "define_estimation_function_f":
            self._inside_define_estimation_function_f = False

    def visit_Return(self, node: cst.Return) -> bool | None:
        # define_estimation_function_f returns (f_func, diff_names, state_names)
        if not self._inside_define_estimation_function_f or self._inside_f_func:
            return None
        if not isinstance(node.value, cst.Tuple) or len(node.value.elements) < 3:
            return None
        for output_names_element in node.value.elements[1:3]:
            self._record_string_elements(output_names_element.value)
        return None

    def visit_Attribute(self, node: cst.Attribute) -> bool | None:
        # Only the object can refer to a quantity, never the attribute name.
        node.value.visit(self)
        return False

    def visit_Arg(self, node: cst.Arg) -> bool | None:
        # Keyword names of calls are not quantity references.
        node.value.visit(self)
        return False

    def visit_Name(self, node: cst.Name) -> bool | None:
        if self._inside_goal_func and node.value in self._goal_func_quantity_names:
            self._record(node, node.value, is_identifier=True)
            return None
        if (
            self._inside_connection_func
            and node.value in self._connection_func_quantity_names
        ):
            self._record(node, node.value, is_identifier=True)
            return None
        if self._inside_f_func and node.value in self._f_func_quantity_names:
            self._record(node, node.value, is_identifier=True)
        return None


def _yaml_node_to_range(node: Node) -> types.Range:
    return types.Range(
        start=types.Position(
            line=node.start_mark.line,
            character=node.start_mark.column,
        ),
        end=types.Position(
            line=node.end_mark.line,
            character=node.end_mark.column,
        ),
    )


def _yaml_mapping_items(node: Node | None) -> list[tuple[Node, Node]]:
    if not isinstance(node, MappingNode):
        return []
    return list(node.value)


def _yaml_mapping_get(node: Node | None, key: str) -> Node | None:
    for key_node, value_node in _yaml_mapping_items(node):
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return value_node
    return None


class _ProjectPythonDocument:
    def __init__(self, path: pathlib.Path, source: str) -> None:
        self.path = str(path)
        self.uri = path.as_uri()
        self.source = source
        self.version: int | None = 1


_ServerDocument = TextDocument | _ProjectPythonDocument


class AICONLanguageServer(LanguageServer):
    """The AICON Language server"""

    GENERATE_FILES_COMMAND = "aiconls.generateFiles"
    GENERATE_EXPERIMENT_SPECIFICATION_COMMAND = (
        "aiconls.generateExperimentSpecification"
    )
    REVALIDATE_GENERATED_COMPONENT_COMMAND = "aiconls.revalidateGeneratedComponent"
    FIX_ALL_SPEC_MISMATCHES_COMMAND = "aiconls.fixAllSpecMismatches"
    RUN_DYNAMIC_CHECK_COMMAND = "aiconls.runDynamicCheck"
    DYNAMIC_DIAGNOSTIC_SOURCE = "aiconls.dynamic"

    components: dict[str, ParsedComponent]
    goals: dict[str, ParsedGoal]
    connections: dict[str, ParsedConnection]

    def __init__(self, name: str, version: str):
        language_server_cls: Any = LanguageServer
        language_server_cls.__init__(self, name, version)

        self.diagnostic_sink = DiagnosticSink()
        self.opened_files: set[str] = set()
        self.stale_files: set[str] = set()
        self._project_root_error_message: str | None = None
        self._project_root_error_notified = False
        self._spec_missing_notified = False
        self._spec_errors_notified = False
        project_root = os.getenv("AICONLS_PROJECT_ROOT")

        project_root_path: pathlib.Path | None = None
        if not project_root:
            self._project_root_error_message = (
                "AICON project root was not specified (AICONLS_PROJECT_ROOT)."
            )
            logger.error(self._project_root_error_message)
        else:
            project_root_path = pathlib.Path(project_root).expanduser().resolve()
            if not project_root_path.exists() or not project_root_path.is_dir():
                self._project_root_error_message = (
                    f"AICON project root folder does not exist: {project_root_path}"
                )
                logger.error(self._project_root_error_message)

        if project_root_path is None:
            project_root_path = pathlib.Path.cwd().resolve()

        self.AICON_PROJECT_PATH = str(project_root_path)

        assert self.AICON_PROJECT_PATH
        self.spec_path = os.path.join(self.AICON_PROJECT_PATH, "spec.yaml")
        self.spec_uri = pathlib.Path(self.spec_path).as_uri()

        self.components = {}
        self.goals = {}
        self.connections = {}
        self.spec = None
        self.spec_parse_diagnostics = []
        self._spec_global_ignore_annotations = (
            parse_global_diagnostic_ignore_annotations("")
        )
        self._python_parse_diagnostics_by_uri: dict[str, list[types.Diagnostic]] = {}
        self._has_build_connections_by_uri: dict[str, bool] = {}
        self._dynamic_diagnostics_by_uri: dict[str, list[types.Diagnostic]] = {}

    def is_project_root_available(self) -> bool:
        return self._project_root_error_message is None

    def notify_project_root_unavailable(self) -> None:
        if self.is_project_root_available() or self._project_root_error_notified:
            return
        assert self._project_root_error_message is not None
        message = self._project_root_error_message
        self.window_log_message(
            types.LogMessageParams(
                message=message,
                type=types.MessageType.Error,
            )
        )
        self.window_show_message(
            types.ShowMessageParams(
                message=message,
                type=types.MessageType.Error,
            )
        )
        self._project_root_error_notified = True

    def _get_spec_file_path(self) -> pathlib.Path:
        return pathlib.Path(self.spec_path)

    def _is_spec_file_available(self) -> bool:
        return self._get_spec_file_path().exists()

    def notify_spec_missing(self) -> None:
        if self._spec_missing_notified:
            return
        spec_path = self._get_spec_file_path()
        message = (
            f"spec.yaml was not found at {spec_path}. "
            "Waiting for spec.yaml to be created."
        )
        self.window_log_message(
            types.LogMessageParams(
                message=message,
                type=types.MessageType.Warning,
            )
        )
        self.window_show_message(
            types.ShowMessageParams(
                message=message,
                type=types.MessageType.Warning,
            )
        )
        self._spec_missing_notified = True

    def notify_spec_has_errors(self) -> None:
        if self._spec_errors_notified:
            return
        message = (
            f"spec.yaml at {self._get_spec_file_path()} contains errors. "
            "Python files are checked once they are fixed."
        )
        self.window_log_message(
            types.LogMessageParams(message=message, type=types.MessageType.Warning)
        )
        self.window_show_message(
            types.ShowMessageParams(message=message, type=types.MessageType.Warning)
        )
        self._spec_errors_notified = True

    def set_diagnostics(
        self,
        uri: str,
        version: int,
        diagnostics: list[types.Diagnostic],
    ) -> None:
        self.diagnostic_sink.set_diagnostics(diagnostics, uri, version)
        self.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri,
                diagnostics=diagnostics,
                version=version,
            )
        )

    def _filter_build_connections_function_missing(
        self, diagnostics: list[types.Diagnostic]
    ) -> list[types.Diagnostic]:
        has_build_connections_anywhere = any(
            self._has_build_connections_by_uri.values()
        )
        if not has_build_connections_anywhere:
            return diagnostics
        missing_build_connections_code = (
            ErrorCode.BUILD_CONNECTIONS_FUNCTION_MISSING.value.code
        )
        return [
            diagnostic
            for diagnostic in diagnostics
            if str(diagnostic.code) != missing_build_connections_code
        ]

    def _effective_python_diagnostics(self, uri: str) -> list[types.Diagnostic]:
        """The parse diagnostics of a file, without those that depend on other files and do not apply."""
        effective_diagnostics = {
            uri: self._filter_build_connections_function_missing(
                list(self._python_parse_diagnostics_by_uri.get(uri, []))
            )
        }
        if self.spec is not None:
            drop_unused_quantity_warnings_used_by_components(
                effective_diagnostics, self.spec, self.components
            )
        return effective_diagnostics[uri]

    def _publish_python_diagnostics_for_uri(
        self, uri: str, only_if_changed: bool = False
    ) -> None:
        if uri not in self._python_parse_diagnostics_by_uri:
            return
        version_entry = self.diagnostic_sink.diagnostics.get(uri)
        if version_entry is None:
            return
        version, previous = version_entry
        effective_diagnostics = self._effective_python_diagnostics(uri)
        if only_if_changed and effective_diagnostics == previous:
            return
        self.set_diagnostics(uri, version, effective_diagnostics)

    def _refresh_all_python_diagnostics(self, only_if_changed: bool = False) -> None:
        for uri in self._python_parse_diagnostics_by_uri:
            self._publish_python_diagnostics_for_uri(uri, only_if_changed)

    def _clone_diagnostic_with_source(
        self, diagnostic: types.Diagnostic, source: str
    ) -> types.Diagnostic:
        return types.Diagnostic(
            range=diagnostic.range,
            message=diagnostic.message,
            severity=diagnostic.severity,
            code=diagnostic.code,
            code_description=diagnostic.code_description,
            source=source,
            tags=diagnostic.tags,
            related_information=diagnostic.related_information,
            data=diagnostic.data,
        )

    def _strip_dynamic_diagnostics(
        self, diagnostics: list[types.Diagnostic]
    ) -> list[types.Diagnostic]:
        return [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.source != self.DYNAMIC_DIAGNOSTIC_SOURCE
        ]

    def clear_dynamic_diagnostics(self) -> None:
        if not self._dynamic_diagnostics_by_uri:
            return

        previous_dynamic_uris = list(self._dynamic_diagnostics_by_uri.keys())
        self._dynamic_diagnostics_by_uri = {}
        for uri in previous_dynamic_uris:
            version, current_diagnostics = self.diagnostic_sink.diagnostics.get(
                uri, (1, [])
            )
            diagnostics_without_dynamic = self._strip_dynamic_diagnostics(
                list(current_diagnostics)
            )
            self.set_diagnostics(uri, version, diagnostics_without_dynamic)

    def _publish_dynamic_diagnostics(
        self, diagnostics_by_uri: dict[str, list[types.Diagnostic]]
    ) -> None:
        self._dynamic_diagnostics_by_uri = diagnostics_by_uri
        for uri, dynamic_diagnostics in diagnostics_by_uri.items():
            version, current_diagnostics = self.diagnostic_sink.diagnostics.get(
                uri, (1, [])
            )
            static_diagnostics = self._strip_dynamic_diagnostics(
                list(current_diagnostics)
            )
            self.set_diagnostics(
                uri,
                version,
                [*static_diagnostics, *dynamic_diagnostics],
            )

    def _log_dynamic_check_results(
        self, diagnostics_by_uri: dict[str, list[types.Diagnostic]]
    ) -> None:
        file_count = len(diagnostics_by_uri)
        diagnostic_count = sum(
            len(diagnostics) for diagnostics in diagnostics_by_uri.values()
        )
        if diagnostic_count == 0:
            summary = "Dynamic check completed: no issues found."
            self.window_show_message(
                types.ShowMessageParams(
                    message=summary,
                    type=types.MessageType.Info,
                )
            )
            self.window_log_message(
                types.LogMessageParams(
                    message=summary,
                    type=types.MessageType.Info,
                )
            )
            return

        summary = (
            "Dynamic check completed: "
            f"{diagnostic_count} issue(s) in {file_count} file(s). "
            "Snapshot diagnostics were published with source 'aiconls.dynamic'."
        )
        self.window_show_message(
            types.ShowMessageParams(
                message=summary,
                type=types.MessageType.Warning,
            )
        )
        self.window_log_message(
            types.LogMessageParams(
                message=summary,
                type=types.MessageType.Warning,
            )
        )
        self.window_log_message(
            types.LogMessageParams(
                message="Dynamic check details:",
                type=types.MessageType.Warning,
            )
        )

        entries: list[tuple[str, int, int, str, str]] = []
        for uri, diagnostics in diagnostics_by_uri.items():
            file_path = pathlib.Path(unquote(urlparse(uri).path))
            for diagnostic in diagnostics:
                code = str(diagnostic.code) if diagnostic.code is not None else "-"
                entries.append(
                    (
                        str(file_path),
                        diagnostic.range.start.line,
                        diagnostic.range.start.character,
                        code,
                        diagnostic.message,
                    )
                )

        for file_path, line, character, code, message in sorted(entries):
            self.window_log_message(
                types.LogMessageParams(
                    message=f"{file_path}:{line + 1}:{character + 1} [{code}] {message}",
                    type=types.MessageType.Warning,
                )
            )

    def validate_spec(self, document: TextDocument | None = None):
        if document is None:
            if self.spec_uri in self.opened_files:
                document = self.workspace.get_text_document(self.spec_uri)
            elif not self._is_spec_file_available():
                self.notify_spec_missing()
                return
            else:
                spec_source = self._get_spec_file_path().read_text(encoding="utf-8")
                self.update_spec()
                diagnostics = list(self.spec_parse_diagnostics)
                if self.spec is not None:
                    spec_validator = SpecReaderValidator(
                        self.spec_uri,
                        self.spec,
                        self.components,
                        self.goals,
                        self.connections,
                        spec_source,
                    )
                    diagnostics.extend(spec_validator.diagnostics)
                diagnostics = filter_diagnostics(diagnostics, spec_source)
                self.set_diagnostics(
                    self.spec_uri,
                    1,
                    diagnostics,
                )
                logger.debug("validated spec")
                return
        # Keep parsed spec and validated content in sync with the latest editor text.
        self.update_spec(document)
        diagnostics = list(self.spec_parse_diagnostics)
        if self.spec is not None:
            spec_validator = SpecReaderValidator(
                document.uri,
                self.spec,
                self.components,
                self.goals,
                self.connections,
                get_document_content(document),
            )
            diagnostics.extend(spec_validator.diagnostics)
        diagnostics = filter_diagnostics(diagnostics, get_document_content(document))

        if document.version is None:
            logger.debug("Spec version is None")
            document.version = 1

        self.set_diagnostics(
            document.uri,
            document.version,
            diagnostics,
        )

        logger.debug("validated spec")

    def update_spec(self, document: TextDocument | None = None):
        if document is None:
            if self.spec_uri in self.opened_files:
                content = get_document_content(
                    self.workspace.get_text_document(self.spec_uri)
                )
            else:
                spec_file = self._get_spec_file_path()
                if not spec_file.exists():
                    self.spec = None
                    self.spec_parse_diagnostics = []
                    self._spec_global_ignore_annotations = (
                        parse_global_diagnostic_ignore_annotations("")
                    )
                    self.notify_spec_missing()
                    return
                content = spec_file.read_text(encoding="utf-8")
        else:
            content = get_document_content(document)
        self._spec_missing_notified = False
        self._spec_global_ignore_annotations = (
            parse_global_diagnostic_ignore_annotations(content)
        )
        parsed_spec = parse_spec_content(content)
        self.spec_parse_diagnostics = parsed_spec.diagnostics

        if parsed_spec.spec is None:
            logger.error("Cant have new_spec")
            self.window_log_message(
                types.LogMessageParams(
                    message="Error while parsing spec", type=types.MessageType.Warning
                )
            )
            self.spec = None
        else:
            logger.debug("updated spec")
            self.spec = parsed_spec.spec
            self._spec_errors_notified = False

    async def _confirm_skeleton_file_overwrite(
        self, existing_files: list[pathlib.Path]
    ) -> bool:
        if not existing_files:
            return True

        overwrite_action = types.MessageActionItem(title="Overwrite")
        cancel_action = types.MessageActionItem(title="Cancel")
        project_root = pathlib.Path(self.AICON_PROJECT_PATH)
        formatted_files: list[str] = []
        for file_path in existing_files:
            try:
                formatted_files.append(str(file_path.relative_to(project_root)))
            except ValueError:
                formatted_files.append(str(file_path))
        file_list = "\n".join(f"- {file_path}" for file_path in formatted_files)
        response = await self.window_show_message_request_async(
            types.ShowMessageRequestParams(
                type=types.MessageType.Warning,
                message=(
                    "The following generated skeleton files already exist and will be "
                    f"overwritten:\n{file_list}\n\nContinue?"
                ),
                actions=[overwrite_action, cancel_action],
            )
        )
        return response is not None and response.title == overwrite_action.title

    async def generate_files(self) -> None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return
        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return

        target_directory = pathlib.Path(self.AICON_PROJECT_PATH)
        target_directory.mkdir(parents=True, exist_ok=True)
        skeleton_files = self.spec.get_skeleton_file_paths(str(target_directory))
        existing_skeleton_files = [
            skeleton_file for skeleton_file in skeleton_files if skeleton_file.exists()
        ]
        should_continue = await self._confirm_skeleton_file_overwrite(
            existing_skeleton_files
        )
        if not should_continue:
            self.window_show_message(
                types.ShowMessageParams(
                    message="AICON skeleton generation cancelled.",
                    type=types.MessageType.Info,
                )
            )
            return
        self.spec.generate_skeleton(str(target_directory))

    def generate_experiment_specification(self) -> None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return
        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return

        target_directory = pathlib.Path(self.AICON_PROJECT_PATH)
        target_directory.mkdir(parents=True, exist_ok=True)
        self.spec.generate_experiment_specification(str(target_directory))

    def generate_component(
        self, component_name: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return None
        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return None

        try:
            target_file, component_content = self.spec.generate_component(
                self.AICON_PROJECT_PATH, component_name
            )
            component_type = self.spec.components[component_name].component_type
        except (RuntimeError, KeyError) as exc:
            self._report_code_action_generation_error(
                f"Failed to generate component code action for '{component_name}': {exc}"
            )
            return None
        header = get_component_header(component_type)

        return self._build_generation_code_action(
            diagnostic=diagnostic,
            title="Auto-generate component skeleton",
            target_file=target_file,
            generated_content=component_content,
            header=header,
        )

    def generate_goal(
        self, goal_name: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return None
        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return None

        try:
            target_file, goal_content = self.spec.generate_goal(
                self.AICON_PROJECT_PATH, goal_name
            )
        except RuntimeError as exc:
            self._report_code_action_generation_error(
                f"Failed to generate goal code action for '{goal_name}': {exc}"
            )
            return None
        return self._build_generation_code_action(
            diagnostic=diagnostic,
            title="Auto-generate goal skeleton",
            target_file=target_file,
            generated_content=goal_content,
            header=gen_goal_head(),
        )

    def generate_connection(
        self, connection_name: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return None
        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return None

        try:
            target_file, connection_content = self.spec.generate_connection(
                self.AICON_PROJECT_PATH, connection_name
            )
        except RuntimeError as exc:
            self._report_code_action_generation_error(
                f"Failed to generate connection code action for '{connection_name}': {exc}"
            )
            return None
        return self._build_generation_code_action(
            diagnostic=diagnostic,
            title="Auto-generate connection skeleton",
            target_file=target_file,
            generated_content=connection_content,
            header=gen_connection_head(),
        )

    def _build_generation_code_action(
        self,
        diagnostic: types.Diagnostic,
        title: str,
        target_file: pathlib.Path,
        generated_content: str,
        header: str,
    ) -> types.CodeAction:
        workspace_edit, target_file_uri = build_generation_workspace_edit(
            self, target_file, generated_content, header
        )

        return types.CodeAction(
            title=title,
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            is_preferred=True,
            edit=workspace_edit,
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[target_file_uri],
            ),
        )

    def _report_code_action_generation_error(
        self, message: str, *, notify_user: bool = False
    ) -> None:
        logger.error(message)
        self.window_log_message(
            types.LogMessageParams(
                message=message,
                type=types.MessageType.Error,
            )
        )
        if notify_user:
            self.window_show_message(
                types.ShowMessageParams(
                    message=message,
                    type=types.MessageType.Error,
                )
            )

    def _update_document_component_quantities(
        self,
        document: _ServerDocument,
        components: list[ParsedSensor | ParsedAction | ParsedEstimator],
        source: str | None = None,
        spec_override: AICON_SPEC | None = None,
    ) -> str | None:
        active_spec = spec_override or self.spec
        if active_spec is None or not components:
            return None

        if source is None:
            if isinstance(document, _ProjectPythonDocument):
                source = document.source
            else:
                source = get_document_content(document)
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            self._report_code_action_generation_error(
                f"Could not parse component source for quantity update in {document.uri}: {exc}"
            )
            return None

        updated_module = module
        for component in components:
            matching_component_instances = (
                active_spec.get_component_instances_by_class_name(component.class_name)
            )
            if not matching_component_instances:
                continue

            spec_component_instance = matching_component_instances[0]
            spec_component = active_spec.components.get(spec_component_instance)
            if spec_component is None:
                continue

            updater = create_component_updater(
                component.component_type, component.class_name, spec_component
            )
            if updater is None:
                continue

            updated_module = updated_module.visit(updater)

        updated_source = updated_module.code
        if updated_source == source:
            return None
        return updated_source

    def _shape_to_spec_literal(self, shape: tuple[int, ...]) -> str:
        if len(shape) == 1:
            return f"({shape[0]},)"
        return f"({', '.join(str(value) for value in shape)})"

    def _format_spec_quantity_name(self, quantity_name: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", quantity_name):
            return quantity_name
        escaped = quantity_name.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _position_to_offset(self, source: str, position: types.Position) -> int:
        lines = source.splitlines(keepends=True)
        return sum(len(line) for line in lines[: position.line]) + position.character

    def _apply_text_edits(self, source: str, edits: list[types.TextEdit]) -> str:
        with_offsets: list[tuple[int, int, str]] = []
        for edit in edits:
            start = self._position_to_offset(source, edit.range.start)
            end = self._position_to_offset(source, edit.range.end)
            with_offsets.append((start, end, edit.new_text))

        updated = source
        for start, end, new_text in sorted(
            with_offsets, key=lambda item: item[0], reverse=True
        ):
            updated = updated[:start] + new_text + updated[end:]
        return updated

    def _build_component_quantities_spec_source(
        self,
        spec_source: str,
        component_instance_name: str,
        quantities: ParsedQuantities,
    ) -> str | None:
        yaml_module: Any = yaml
        composed_root = cast(Node | None, yaml_module.compose(spec_source))
        if composed_root is None:
            return None

        components_node = _yaml_mapping_get(composed_root, "components")
        component_node: Node | None = None
        for component_name_node, candidate_component_node in _yaml_mapping_items(
            components_node
        ):
            if (
                isinstance(component_name_node, ScalarNode)
                and component_name_node.value == component_instance_name
            ):
                component_node = candidate_component_node
                break
        if component_node is None:
            return None

        quantities_key_node: ScalarNode | None = None
        quantities_node: Node | None = None
        for property_name_node, property_node in _yaml_mapping_items(component_node):
            if (
                isinstance(property_name_node, ScalarNode)
                and property_name_node.value == "quantities"
            ):
                quantities_key_node = property_name_node
                quantities_node = property_node
                break

        if quantities_key_node is None or not isinstance(quantities_node, MappingNode):
            return None

        base_indent = " " * quantities_key_node.start_mark.column
        entry_indent = f"{base_indent}  "
        shape_indent = f"{base_indent}    "
        if not quantities:
            new_quantities_block = f"{base_indent}quantities: {{}}"
        else:
            lines = [f"{base_indent}quantities:"]
            for quantity_name, quantity_props in quantities.items():
                lines.append(
                    f"{entry_indent}{self._format_spec_quantity_name(quantity_name)}:"
                )
                lines.append(
                    f"{shape_indent}shape: {self._shape_to_spec_literal(quantity_props.shape)}"
                )
            new_quantities_block = "\n".join(lines)

        source_lines = spec_source.splitlines(keepends=True)
        start_line = quantities_key_node.start_mark.line
        start_indent = quantities_key_node.start_mark.column
        end_line = len(source_lines)
        for line_index in range(start_line + 1, len(source_lines)):
            candidate = source_lines[line_index]
            stripped = candidate.lstrip(" \t")
            if not stripped or stripped == "\n":
                continue
            indent = len(candidate) - len(stripped)
            if indent <= start_indent:
                end_line = line_index
                break

        start_offset = sum(len(line) for line in source_lines[:start_line])
        if end_line < len(source_lines):
            end_offset = sum(len(line) for line in source_lines[:end_line])
            replacement_block = f"{new_quantities_block}\n"
        else:
            end_offset = len(spec_source)
            replacement_block = new_quantities_block

        replacement = (
            spec_source[:start_offset] + replacement_block + spec_source[end_offset:]
        )
        if replacement == spec_source:
            return None
        return replacement

    def _build_project_quantity_sync_edit(
        self, spec_override: AICON_SPEC
    ) -> types.WorkspaceEdit | None:
        sources_by_uri = self._build_project_python_sources()
        components_by_uri: dict[
            str, list[ParsedSensor | ParsedAction | ParsedEstimator]
        ] = {}
        connections_by_uri: dict[str, list[ParsedConnection]] = {}

        for component in self.components.values():
            if not isinstance(component, (ParsedSensor, ParsedAction, ParsedEstimator)):
                continue
            components_by_uri.setdefault(component.document_uri, []).append(component)

        for connection in self.connections.values():
            connections_by_uri.setdefault(connection.document_uri, []).append(
                connection
            )

        changes: dict[str, list[types.TextEdit]] = {}
        for uri in set(components_by_uri.keys()) | set(connections_by_uri.keys()):
            source = sources_by_uri.get(uri)
            if source is None:
                continue
            uri_path = pathlib.Path(unquote(urlparse(uri).path))
            document = _ProjectPythonDocument(uri_path, source)
            updated_source = source
            has_update = False

            component_source = self._update_document_component_quantities(
                document,
                components_by_uri.get(uri, []),
                source=updated_source,
                spec_override=spec_override,
            )
            if component_source is not None:
                updated_source = component_source
                has_update = True

            connection_source = self._update_document_connection_quantities(
                document,
                connections_by_uri.get(uri, []),
                source=updated_source,
                spec_override=spec_override,
            )
            if connection_source is not None:
                updated_source = connection_source
                has_update = True

            if not has_update:
                continue
            changes[uri] = [
                types.TextEdit(
                    range=build_full_document_range(source),
                    new_text=updated_source,
                )
            ]

        if not changes:
            return None
        return types.WorkspaceEdit(changes=changes)

    def _build_spec_update_from_component(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> tuple[str, str, str, AICON_SPEC, str, set[str]] | None:
        if self.spec is None:
            return None

        document = self.workspace.get_text_document(document_uri)
        component = self.get_component_for_position(document, diagnostic.range.start)
        if component is None:
            self.parse(document)
            component = self.get_component_for_position(
                document, diagnostic.range.start
            )
        if component is None:
            return None
        if not isinstance(component, (ParsedSensor, ParsedAction, ParsedEstimator)):
            return None

        component_instances = self.spec.get_component_instances_by_class_name(
            component.name
        )
        if not component_instances:
            return None
        component_instance = component_instances[0]
        spec_component = self.spec.components.get(component_instance)
        if spec_component is None:
            return None
        previous_quantities = spec_component.quantities

        spec_uri, spec_source = self._build_spec_source()
        updated_spec_source = self._build_component_quantities_spec_source(
            spec_source,
            component_instance,
            component.quantities,
        )
        if updated_spec_source is None:
            return None

        parsed_updated_spec = parse_spec_content(updated_spec_source)
        if parsed_updated_spec.spec is None:
            self._report_code_action_generation_error(
                "Failed to generate spec propagation code action because the updated spec became invalid."
            )
            return None
        changed_quantity_names = {
            quantity_name
            for quantity_name in set(previous_quantities) | set(component.quantities)
            if previous_quantities.get(quantity_name)
            != component.quantities.get(quantity_name)
        }
        return (
            spec_uri,
            spec_source,
            updated_spec_source,
            parsed_updated_spec.spec,
            component_instance,
            changed_quantity_names,
        )

    def generate_spec_from_component_update(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        update_result = self._build_spec_update_from_component(document_uri, diagnostic)
        if update_result is None:
            return None
        spec_uri, spec_source, updated_spec_source, _, _, _ = update_result
        return types.CodeAction(
            title="Update spec quantities from component",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            edit=types.WorkspaceEdit(
                changes={
                    spec_uri: [
                        types.TextEdit(
                            range=build_full_document_range(spec_source),
                            new_text=updated_spec_source,
                        )
                    ]
                }
            ),
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[spec_uri],
            ),
        )

    def generate_spec_from_component_update_and_propagate(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        update_result = self._build_spec_update_from_component(document_uri, diagnostic)
        if update_result is None:
            return None
        (
            spec_uri,
            spec_source,
            updated_spec_source,
            updated_spec,
            component_instance,
            changed_quantity_names,
        ) = update_result

        project_edit = self._build_quantity_scoped_project_sync_edit(
            updated_spec,
            changed_component_instance=component_instance,
            changed_quantity_names=changed_quantity_names,
        )
        changes: dict[str, list[types.TextEdit]] = {}
        if project_edit and project_edit.changes:
            for uri, edit_list in project_edit.changes.items():
                changes[uri] = list(edit_list)
        changes[spec_uri] = [
            types.TextEdit(
                range=build_full_document_range(spec_source),
                new_text=updated_spec_source,
            )
        ]

        return types.CodeAction(
            title="Update spec quantities from component and propagate",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            edit=types.WorkspaceEdit(changes=changes),
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[spec_uri],
            ),
        )

    def _build_quantity_scoped_project_sync_edit(
        self,
        spec_override: AICON_SPEC,
        *,
        changed_component_instance: str,
        changed_quantity_names: set[str],
    ) -> types.WorkspaceEdit | None:
        if not changed_quantity_names:
            return None

        sources_by_uri = self._build_project_python_sources()
        connections_by_uri: dict[str, list[ParsedConnection]] = {}
        expected_quantities_overrides: dict[str, SpecQuantities] = {}

        for connection in self.connections.values():
            connection_instances = spec_override.get_connection_instances_by_class_name(
                connection.name
            )
            if not connection_instances:
                continue
            if not any(
                changed_component_instance
                in spec_override.connections[connection_instance].components
                for connection_instance in connection_instances
                if connection_instance in spec_override.connections
            ):
                continue

            expected_quantities = self._build_expected_connection_quantities(
                connection,
                spec_override=spec_override,
            )
            if expected_quantities is None:
                continue

            original_connection_quantities: SpecQuantities = {
                quantity_name: SpecComponentQuantityGeneric(shape=quantity_props.shape)
                for quantity_name, quantity_props in connection.quantities.items()
            }
            scoped_quantities: SpecQuantities = dict(original_connection_quantities)
            for quantity_name in sorted(changed_quantity_names):
                if quantity_name in expected_quantities:
                    scoped_quantities[quantity_name] = expected_quantities[
                        quantity_name
                    ]
                else:
                    scoped_quantities.pop(quantity_name, None)

            if scoped_quantities == original_connection_quantities:
                continue

            connections_by_uri.setdefault(connection.document_uri, []).append(
                connection
            )
            expected_quantities_overrides[connection.name] = scoped_quantities

        if not connections_by_uri:
            return None

        changes: dict[str, list[types.TextEdit]] = {}
        for uri, connections in connections_by_uri.items():
            source = sources_by_uri.get(uri)
            if source is None:
                continue
            uri_path = pathlib.Path(unquote(urlparse(uri).path))
            document = _ProjectPythonDocument(uri_path, source)
            updated_source = self._update_document_connection_quantities(
                document,
                connections,
                source=source,
                spec_override=spec_override,
                expected_quantities_overrides=expected_quantities_overrides,
            )
            if updated_source is None:
                continue

            changes[uri] = [
                types.TextEdit(
                    range=build_full_document_range(source),
                    new_text=updated_source,
                )
            ]

        if not changes:
            return None
        return types.WorkspaceEdit(changes=changes)

    def _build_expected_connection_quantities(
        self,
        parsed_connection: ParsedConnection,
        spec_override: AICON_SPEC | None = None,
    ) -> SpecQuantities | None:
        active_spec = spec_override or self.spec
        if active_spec is None:
            return None

        connection_instances = active_spec.get_connection_instances_by_class_name(
            parsed_connection.name
        )
        if not connection_instances:
            return None

        quantities: SpecQuantities = {}
        for connection_instance in connection_instances:
            connection = active_spec.connections[connection_instance]
            for component_name in connection.components:
                spec_component = active_spec.components.get(component_name)
                if spec_component is None:
                    continue
                for quantity_name, quantity_props in spec_component.quantities.items():
                    quantities[quantity_name] = quantity_props

        return quantities

    def _update_document_connection_quantities(
        self,
        document: _ServerDocument,
        connections: list[ParsedConnection],
        source: str | None = None,
        spec_override: AICON_SPEC | None = None,
        expected_quantities_overrides: dict[str, SpecQuantities] | None = None,
    ) -> str | None:
        active_spec = spec_override or self.spec
        if active_spec is None or not connections:
            return None

        if source is None:
            if isinstance(document, _ProjectPythonDocument):
                source = document.source
            else:
                source = get_document_content(document)
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            self._report_code_action_generation_error(
                f"Could not parse connection source for quantity update in {document.uri}: {exc}"
            )
            return None

        updated_module = module
        for connection in connections:
            expected_quantities = None
            if expected_quantities_overrides is not None:
                expected_quantities = expected_quantities_overrides.get(connection.name)
            if expected_quantities is None:
                expected_quantities = self._build_expected_connection_quantities(
                    connection, spec_override=active_spec
                )
            if expected_quantities is None:
                continue

            updater = ConnectionUpdater(connection.name, expected_quantities)
            updated_module = updated_module.visit(updater)

        updated_source = updated_module.code
        if updated_source == source:
            return None
        return updated_source

    def generate_component_update(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        document = self.workspace.get_text_document(document_uri)
        component = self.get_component_for_position(document, diagnostic.range.start)
        if component is None:
            self.parse(document)
            component = self.get_component_for_position(
                document, diagnostic.range.start
            )
        if component is None:
            return None
        if not isinstance(component, (ParsedSensor, ParsedAction, ParsedEstimator)):
            return None

        updated_source = self._update_document_component_quantities(
            document, [component]
        )
        if updated_source is None:
            return None

        source = get_document_content(document)
        edit = types.TextEdit(
            range=build_full_document_range(source),
            new_text=updated_source,
        )
        return types.CodeAction(
            title="Update component quantities to match spec",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            is_preferred=True,
            edit=types.WorkspaceEdit(changes={document_uri: [edit]}),
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[document_uri],
            ),
        )

    def generate_connection_update(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        document = self.workspace.get_text_document(document_uri)
        connection = self.get_connection_for_position(document, diagnostic.range.start)
        if connection is None:
            self.parse(document)
            connection = self.get_connection_for_position(
                document, diagnostic.range.start
            )
        if connection is None:
            return None

        updated_source = self._update_document_connection_quantities(
            document, [connection]
        )
        if updated_source is None:
            return None

        source = get_document_content(document)
        edit = types.TextEdit(
            range=build_full_document_range(source),
            new_text=updated_source,
        )
        return types.CodeAction(
            title="Update connection quantities to match spec",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            is_preferred=True,
            edit=types.WorkspaceEdit(changes={document_uri: [edit]}),
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[document_uri],
            ),
        )

    def generate_build_connections_update(
        self, document_uri: str, diagnostic: types.Diagnostic
    ) -> types.CodeAction | None:
        if self.spec is None:
            return None

        document = self.workspace.get_text_document(document_uri)
        source = get_document_content(document)
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            self._report_code_action_generation_error(
                f"Could not parse source for build_connections update in {document_uri}: {exc}"
            )
            return None

        updated_source = module.visit(
            BuildConnectionsUpdater(self.spec.connections)
        ).code
        if updated_source == source:
            return None

        edit = types.TextEdit(
            range=build_full_document_range(source),
            new_text=updated_source,
        )
        return types.CodeAction(
            title="Generate/update build_connections to match spec",
            kind=types.CodeActionKind.QuickFix,
            diagnostics=[diagnostic],
            is_preferred=True,
            edit=types.WorkspaceEdit(changes={document_uri: [edit]}),
            command=types.Command(
                title="Revalidate generated component",
                command=self.REVALIDATE_GENERATED_COMPONENT_COMMAND,
                arguments=[document_uri],
            ),
        )

    def build_fix_all_spec_mismatches_edit(
        self, document_uri: str | None = None
    ) -> types.WorkspaceEdit | None:
        if self.spec is None:
            return None

        component_mismatch_codes = {
            ErrorCode.SHAPE_SPEC_MISMATCH.value.code,
            ErrorCode.QUANTITIES_NOT_IMPLEMENTED.value.code,
        }
        connection_mismatch_code = ErrorCode.QUANTITY_NOT_AVAILABLE.value.code
        components_by_uri: dict[
            str, list[ParsedSensor | ParsedAction | ParsedEstimator]
        ] = {}
        connections_by_uri: dict[str, list[ParsedConnection]] = {}

        for uri, (_, diagnostics) in self.diagnostic_sink.diagnostics.items():
            if document_uri is not None and uri != document_uri:
                continue

            if not is_python_file(uri):
                continue

            target_document = self.workspace.get_text_document(uri)
            matched_components: dict[
                str, ParsedSensor | ParsedAction | ParsedEstimator
            ] = {}
            matched_connections: dict[str, ParsedConnection] = {}
            for diagnostic in diagnostics:
                if diagnostic.code in component_mismatch_codes:
                    component = self.get_component_for_position(
                        target_document, diagnostic.range.start
                    )
                    if component is None:
                        continue
                    if isinstance(
                        component, (ParsedSensor, ParsedAction, ParsedEstimator)
                    ):
                        matched_components[component.name] = component

                if diagnostic.code == connection_mismatch_code:
                    connection = self.get_connection_for_position(
                        target_document, diagnostic.range.start
                    )
                    if connection is None:
                        continue
                    matched_connections[connection.name] = connection

            if matched_components:
                components_by_uri[uri] = list(matched_components.values())
            if matched_connections:
                connections_by_uri[uri] = list(matched_connections.values())

        if not components_by_uri and not connections_by_uri:
            return None

        changes: dict[str, list[types.TextEdit]] = {}
        for uri in set(components_by_uri.keys()) | set(connections_by_uri.keys()):
            document = self.workspace.get_text_document(uri)
            source = get_document_content(document)
            updated_source = source
            has_update = False

            component_source = self._update_document_component_quantities(
                document,
                components_by_uri.get(uri, []),
                source=updated_source,
            )
            if component_source is not None:
                updated_source = component_source
                has_update = True

            connection_source = self._update_document_connection_quantities(
                document,
                connections_by_uri.get(uri, []),
                source=updated_source,
            )
            if connection_source is not None:
                updated_source = connection_source
                has_update = True

            if not has_update:
                continue

            changes[uri] = [
                types.TextEdit(
                    range=build_full_document_range(source),
                    new_text=updated_source,
                )
            ]

        if not changes:
            return None

        return types.WorkspaceEdit(changes=changes)

    def fix_all_spec_mismatches(self, document_uri: str | None = None) -> None:
        workspace_edit = self.build_fix_all_spec_mismatches_edit(document_uri)
        if workspace_edit is None:
            return

        self.workspace_apply_edit(
            types.ApplyWorkspaceEditParams(edit=workspace_edit),
        )

        if workspace_edit.changes is None:
            return

        for uri in workspace_edit.changes:
            target_document = self.workspace.get_text_document(uri)
            self.parse(target_document)
        self.validate_spec()

    def _mark_open_python_files_stale(self):
        for uri in self.opened_files:
            if uri == self.spec_uri or not is_python_file(uri):
                continue
            self.stale_files.add(uri)

    def refresh_document_if_stale(self, document: TextDocument) -> None:
        if document.uri not in self.stale_files:
            return
        self.parse(document)

    def _file_in_workspace(self, doc: _ServerDocument):
        project_path = pathlib.Path(self.AICON_PROJECT_PATH).resolve()
        doc_path = pathlib.Path(doc.path).resolve()
        return doc_path.is_relative_to(project_path)

    def _spec_contains_errors(self) -> bool:
        return any(
            diagnostic.severity == types.DiagnosticSeverity.Error
            for diagnostic in self.spec_parse_diagnostics
        )

    def parse_all_project_python_files(self) -> None:
        project_path = pathlib.Path(self.AICON_PROJECT_PATH).resolve()
        for python_file in sorted(project_path.rglob("*.py")):
            relative_parts = python_file.relative_to(project_path).parts
            if any(part in EXCLUDED_PROJECT_DIRECTORIES for part in relative_parts):
                continue
            file_uri = python_file.as_uri()
            if file_uri in self.opened_files:
                open_document = self.workspace.get_text_document(file_uri)
                self.parse(
                    open_document,
                    validate_spec_after=False,
                    track_open_document=False,
                )
                continue

            source = python_file.read_text(encoding="utf-8")
            self.parse(
                _ProjectPythonDocument(path=python_file, source=source),
                validate_spec_after=False,
                track_open_document=False,
            )

        if self.spec is not None:
            add_imported_classes(
                self.spec,
                self._iter_project_python_files(),
                project_path,
                self._spec_global_ignore_annotations,
                self.components,
                self.goals,
                self.connections,
            )
            self._refresh_all_python_diagnostics(only_if_changed=True)

    def run_dynamic_check(self) -> None:
        if not self._is_spec_file_available() and self.spec is None:
            self.notify_spec_missing()
            return

        if self.spec_uri in self.opened_files:
            spec_document = self.workspace.get_text_document(self.spec_uri)
            self.parse(
                spec_document,
                validate_spec_after=False,
                track_open_document=False,
            )
        else:
            self.update_spec()
            self.validate_spec()

        if self.spec is None or spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return

        self.parse_all_project_python_files()
        self.validate_spec()

        if spec_has_errors(self.diagnostic_sink, self.spec_uri):
            notify_spec_malformed(self)
            return

        self.clear_dynamic_diagnostics()

        try:
            raw_dynamic_diagnostics = dynamic_analysis(
                self.spec,
                self.components,
                self.goals,
                self.connections,
            )
        except Exception as exc:
            message = (
                "Dynamic check failed: "
                f"{exc.__class__.__name__}: {exc}. See logs for details."
            )
            self.window_log_message(
                types.LogMessageParams(
                    message=message,
                    type=types.MessageType.Error,
                )
            )
            self.window_show_message(
                types.ShowMessageParams(
                    message=message,
                    type=types.MessageType.Error,
                )
            )
            mark_exception_logged(exc)
            logger.exception("Dynamic check failed")
            raise

        dynamic_diagnostics_by_uri: dict[str, list[types.Diagnostic]] = {}
        for uri, diagnostics in raw_dynamic_diagnostics.items():
            diagnostics_with_dynamic_source = [
                self._clone_diagnostic_with_source(
                    diagnostic, self.DYNAMIC_DIAGNOSTIC_SOURCE
                )
                for diagnostic in diagnostics
            ]
            if not diagnostics_with_dynamic_source:
                continue
            dynamic_diagnostics_by_uri[uri] = diagnostics_with_dynamic_source

        self._publish_dynamic_diagnostics(dynamic_diagnostics_by_uri)
        self._log_dynamic_check_results(dynamic_diagnostics_by_uri)

    def parse(
        self,
        document: _ServerDocument,
        *,
        validate_spec_after: bool = True,
        track_open_document: bool = True,
    ):
        if not self._file_in_workspace(document):
            return
        try:
            if is_spec_yaml(document.path):
                if isinstance(document, _ProjectPythonDocument):
                    return
                if track_open_document:
                    self.opened_files.add(document.uri)
                self.update_spec(document)
                if not self._spec_contains_errors():
                    self._mark_open_python_files_stale()
                self.validate_spec(document)
                return

            _, previous = self.diagnostic_sink.diagnostics.get(document.uri, (0, []))

            if track_open_document:
                self.opened_files.add(document.uri)
            self.stale_files.discard(document.uri)

            if not document.version:
                logger.debug("No document version for: %s", document.uri)
                document.version = 1

            if self.spec is None and self._is_spec_file_available():
                # also publishes the spec's own diagnostics so the errors are visible
                self.validate_spec()

            if self.spec is None:
                if self._is_spec_file_available():
                    self.notify_spec_has_errors()
                else:
                    self.notify_spec_missing()
                self.set_diagnostics(document.uri, document.version, [])
                return

            had_build_connections_anywhere = any(
                self._has_build_connections_by_uri.values()
            )

            component_parser = AICONComponentParser(
                self.spec,
                document_uri=document.uri,
                project_root=pathlib.Path(self.AICON_PROJECT_PATH),
            )

            if isinstance(document, _ProjectPythonDocument):
                source = document.source
            else:
                source = get_document_content(document)

            try:
                wrapper = MetadataWrapper(cst.parse_module(source))
            except cst.ParserSyntaxError:
                self._python_parse_diagnostics_by_uri[document.uri] = []
                self._has_build_connections_by_uri[document.uri] = False
                self.set_diagnostics(document.uri, document.version, [])
                if had_build_connections_anywhere != any(
                    self._has_build_connections_by_uri.values()
                ):
                    self._refresh_all_python_diagnostics()
                return

            logger.debug("Parsing %s", document.uri)
            wrapper.visit(component_parser)
            filtered_diagnostics = filter_diagnostics(
                component_parser.diagnostics,
                source,
                global_annotations=self._spec_global_ignore_annotations,
            )
            self._python_parse_diagnostics_by_uri[document.uri] = list(
                filtered_diagnostics
            )
            self._has_build_connections_by_uri[document.uri] = (
                component_parser.has_build_connections_function
            )
            has_build_connections_anywhere_changed = (
                had_build_connections_anywhere
                != any(self._has_build_connections_by_uri.values())
            )

            for component, component_props in list(self.components.items()):
                if component_props.document_uri == document.uri:
                    del self.components[component]
            for goal, goal_props in list(self.goals.items()):
                if goal_props.document_uri == document.uri:
                    del self.goals[goal]
            for connection, connection_props in list(self.connections.items()):
                if connection_props.document_uri == document.uri:
                    del self.connections[connection]

            self.components.update(component_parser.components)
            self.goals.update(component_parser.goals)
            self.connections.update(component_parser.connections)
            effective_diagnostics = self._effective_python_diagnostics(document.uri)

            if previous != effective_diagnostics:
                logger.debug("%s: Changing diagnostics", document.path)

                self.set_diagnostics(
                    document.uri,
                    document.version,
                    effective_diagnostics,
                )
            else:
                logger.debug("%s: Non Changing diagnostics", document.path)

            # the diagnostics of the other files can depend on this one (e.g. which quantities its components use)
            self._refresh_all_python_diagnostics(
                only_if_changed=not has_build_connections_anywhere_changed
            )

            if validate_spec_after:
                self.validate_spec()
        except Exception as exc:
            mark_exception_logged(exc)
            logger.exception(
                "Failed to parse document %s (%s)",
                document.path,
                document.uri,
            )
            raise

    def get_component_for_position(
        self, document: TextDocument, position: types.Position
    ) -> ParsedComponent | None:
        document_components = {
            name: component
            for name, component in self.components.items()
            if component.document_uri == document.uri
        }
        if not document_components:
            return None

        source = get_document_content(document)

        try:
            wrapper = MetadataWrapper(cst.parse_module(source))
        except cst.ParserSyntaxError:
            return None

        finder = EnclosingComponentFinder(position, set(document_components.keys()))
        wrapper.visit(finder)
        if finder.component_name is None:
            return None

        return document_components.get(finder.component_name)

    def get_connection_for_position(
        self, document: TextDocument, position: types.Position
    ) -> ParsedConnection | None:
        document_connections = {
            name: connection
            for name, connection in self.connections.items()
            if connection.document_uri == document.uri
        }
        if not document_connections:
            return None

        source = get_document_content(document)

        try:
            wrapper = MetadataWrapper(cst.parse_module(source))
        except cst.ParserSyntaxError:
            return None

        finder = EnclosingComponentFinder(position, set(document_connections.keys()))
        wrapper.visit(finder)
        if finder.component_name is None:
            return None

        return document_connections.get(finder.component_name)

    def get_connection_names_for_component(self, component_name: str) -> list[str]:
        if self.spec is None:
            return []

        if component_name in self.spec.components:
            component_instances = {component_name}
        else:
            component_instances = set(
                self.spec.get_component_instances_by_class_name(component_name)
            )
        if not component_instances:
            return []
        return [
            spec_name
            for spec_name, connection_props in self.spec.connections.items()
            if component_instances.intersection(connection_props.components)
        ]

    def get_connection_quantities_for_component(
        self, component_name: str, connection_name: str
    ) -> list[str]:
        if self.spec is None:
            return []

        connection_props = self.spec.connections.get(connection_name)
        if component_name in self.spec.components:
            component_instances = {component_name}
        else:
            component_instances = set(
                self.spec.get_component_instances_by_class_name(component_name)
            )
        if (
            connection_props is None
            or not component_instances
            or not component_instances.intersection(connection_props.components)
        ):
            return []

        own_quantity_names: set[str] = set()
        for component_instance in component_instances:
            own_component_props = self.spec.components.get(component_instance)
            if own_component_props is None:
                continue
            own_quantity_names.update(own_component_props.quantities.keys())

        quantity_names: list[str] = []
        for linked_component in connection_props.components:
            if linked_component in component_instances:
                continue
            linked_component_props = self.spec.components.get(linked_component)
            if linked_component_props is None:
                continue
            for quantity_name in linked_component_props.quantities:
                if quantity_name in own_quantity_names:
                    continue
                quantity_names.append(quantity_name)

        return list(dict.fromkeys(quantity_names))

    def _iter_project_python_files(self) -> list[pathlib.Path]:
        project_path = pathlib.Path(self.AICON_PROJECT_PATH).resolve()
        files: list[pathlib.Path] = []
        for python_file in sorted(project_path.rglob("*.py")):
            relative_parts = python_file.relative_to(project_path).parts
            if any(part in EXCLUDED_PROJECT_DIRECTORIES for part in relative_parts):
                continue
            files.append(python_file)
        return files

    def _collect_quantity_occurrences(self, source: str) -> list[_QuantityOccurrence]:
        try:
            wrapper = MetadataWrapper(cst.parse_module(source))
        except cst.ParserSyntaxError:
            return []
        collector = _QuantityOccurrenceCollector()
        wrapper.visit(collector)
        return collector.occurrences

    def _build_project_python_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        for uri in self.opened_files:
            if not is_python_file(uri):
                continue
            document = self.workspace.get_text_document(uri)
            sources[uri] = get_document_content(document)

        for python_file in self._iter_project_python_files():
            uri = python_file.as_uri()
            if uri in sources:
                continue
            sources[uri] = python_file.read_text(encoding="utf-8")

        return sources

    def _build_spec_source(self) -> tuple[str, str]:
        if self.spec_uri in self.opened_files:
            spec_document = self.workspace.get_text_document(self.spec_uri)
            return self.spec_uri, get_document_content(spec_document)
        return pathlib.Path(self.spec_path).as_uri(), pathlib.Path(
            self.spec_path
        ).read_text(encoding="utf-8")

    def _collect_spec_quantity_occurrences(
        self, source: str
    ) -> list[_QuantityOccurrence]:
        try:
            yaml_module: Any = yaml
            composed_root = cast(Node | None, yaml_module.compose(source))
        except yaml.YAMLError:
            return []

        if composed_root is None:
            return []

        components_node = _yaml_mapping_get(composed_root, "components")
        occurrences: list[_QuantityOccurrence] = []
        for _, component_node in _yaml_mapping_items(components_node):
            quantities_node = _yaml_mapping_get(component_node, "quantities")
            for quantity_name_node, _ in _yaml_mapping_items(quantities_node):
                if not isinstance(quantity_name_node, ScalarNode):
                    continue
                occurrences.append(
                    _QuantityOccurrence(
                        name=quantity_name_node.value,
                        position=_yaml_node_to_range(quantity_name_node),
                        is_identifier=False,
                    )
                )
        return occurrences

    def _find_quantity_name_at_position(
        self,
        occurrences: list[_QuantityOccurrence],
        position: types.Position,
    ) -> str | None:
        matches = [
            occurrence
            for occurrence in occurrences
            if position_in_range(position, occurrence.position)
        ]
        if not matches:
            return None

        def _sort_key(occurrence: _QuantityOccurrence) -> tuple[int, int, int]:
            return (
                occurrence.position.end.line - occurrence.position.start.line,
                occurrence.position.end.character - occurrence.position.start.character,
                occurrence.position.start.character,
            )

        return min(matches, key=_sort_key).name

    def _format_python_string_replacement(self, new_name: str, quote: str) -> str:
        escaped = new_name.replace("\\", "\\\\")
        if quote == "'":
            escaped = escaped.replace("'", "\\'")
        else:
            escaped = escaped.replace('"', '\\"')
        return f"{quote}{escaped}{quote}"

    def build_quantity_rename_edit(
        self,
        document_uri: str,
        position: types.Position,
        new_name: str,
    ) -> types.WorkspaceEdit | None:
        python_sources_by_uri = self._build_project_python_sources()
        spec_uri, spec_source = self._build_spec_source()
        source_uris = set(python_sources_by_uri.keys()) | {spec_uri}
        if document_uri not in source_uris:
            return None

        python_occurrences_by_uri: dict[str, list[_QuantityOccurrence]] = {}
        for uri, document_source in python_sources_by_uri.items():
            occurrences = self._collect_quantity_occurrences(document_source)
            if occurrences:
                python_occurrences_by_uri[uri] = occurrences

        spec_occurrences = self._collect_spec_quantity_occurrences(spec_source)

        source_occurrences = (
            python_occurrences_by_uri.get(document_uri, [])
            if document_uri != spec_uri
            else spec_occurrences
        )
        old_name = self._find_quantity_name_at_position(source_occurrences, position)
        if old_name is None or old_name == new_name:
            return None

        old_name_occurrences = [
            occurrence
            for occurrences in python_occurrences_by_uri.values()
            for occurrence in occurrences
            if occurrence.name == old_name
        ]
        old_name_occurrences.extend(
            occurrence for occurrence in spec_occurrences if occurrence.name == old_name
        )
        if not old_name_occurrences:
            return None

        if any(occurrence.is_identifier for occurrence in old_name_occurrences) and (
            not new_name.isidentifier() or keyword.iskeyword(new_name)
        ):
            self.window_log_message(
                types.LogMessageParams(
                    type=types.MessageType.Warning,
                    message=(
                        f"Cannot rename quantity '{old_name}' to '{new_name}': name must be a valid Python identifier."
                    ),
                )
            )
            return None

        changes: dict[str, list[types.TextEdit]] = {}
        seen_ranges_by_uri: dict[str, set[tuple[int, int, int, int]]] = {}
        all_occurrences_by_uri = dict(python_occurrences_by_uri)
        all_occurrences_by_uri[spec_uri] = spec_occurrences
        for uri, occurrences in all_occurrences_by_uri.items():
            edits: list[types.TextEdit] = []
            seen_ranges = seen_ranges_by_uri.setdefault(uri, set())
            for occurrence in occurrences:
                if occurrence.name != old_name:
                    continue
                range_key = (
                    occurrence.position.start.line,
                    occurrence.position.start.character,
                    occurrence.position.end.line,
                    occurrence.position.end.character,
                )
                if range_key in seen_ranges:
                    continue
                seen_ranges.add(range_key)
                new_text = (
                    self._format_python_string_replacement(
                        new_name, occurrence.python_quote
                    )
                    if occurrence.python_quote is not None
                    else new_name
                )
                edits.append(
                    types.TextEdit(
                        range=occurrence.position,
                        new_text=new_text,
                    )
                )
            if edits:
                changes[uri] = edits

        if not changes:
            return None
        return types.WorkspaceEdit(changes=changes)


def create_server() -> AICONLanguageServer:
    return AICONLanguageServer("diagnostic-server", "v1")
