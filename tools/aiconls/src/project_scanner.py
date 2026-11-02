import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import libcst as cst
from libcst.metadata import MetadataWrapper
from lsprotocol import types
from lsprotocol.types import Diagnostic, DiagnosticSeverity

from diagnostics.diagnostics import ErrorCode
from diagnostics.warning_ignore_annotations import (
    DiagnosticIgnoreAnnotations,
    filter_diagnostics,
    parse_global_diagnostic_ignore_annotations,
)
from reader.aicon_files.aicon_reader import AICONComponentParser
from reader.aicon_files.packages import find_imported_class_files
from reader.aicon_files.reader_types import (
    ParsedAction,
    ParsedComponents,
    ParsedConnections,
    ParsedEstimator,
    ParsedGoals,
    ParsedSensor,
)
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import SpecComponentProps, SpecComponentTypes
from updater.action_updater import ActionUpdater
from updater.estimator_updater import EstimatorUpdater
from updater.sensor_updater import SensorUpdater
from validator.spec_validator import SpecReaderValidator

SPEC_NOT_FOUND_CODE = "E900"
SPEC_MALFORMED_CODE = "E901"
PYTHON_SYNTAX_ERROR_CODE = "E902"
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"


class ReportPosition(TypedDict):
    line: int
    character: int


class ReportRange(TypedDict):
    start: ReportPosition
    end: ReportPosition


class ReportDiagnostic(TypedDict):
    path: str
    code: str | None
    message: str
    source: str | None
    severity: str
    range: ReportRange


class ReportFileDiagnostics(TypedDict):
    path: str
    uri: str
    diagnostics: list[ReportDiagnostic]


class ReportSummary(TypedDict):
    python_files_scanned: int
    files_with_diagnostics: int
    diagnostic_count: int
    error_count: int
    warning_count: int


class ScanReport(TypedDict):
    project_root: str
    spec_path: str
    status: str
    summary: ReportSummary
    diagnostics_by_file: list[ReportFileDiagnostics]
    diagnostics: list[ReportDiagnostic]


@dataclass(frozen=True)
class ParsedPythonFile:
    components: ParsedComponents
    goals: ParsedGoals
    connections: ParsedConnections
    diagnostics: list[Diagnostic]
    has_build_connections_function: bool


def _filter_build_connections_function_missing(
    diagnostics: list[Diagnostic], has_build_connections_anywhere: bool
) -> list[Diagnostic]:
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


def _make_range(line: int = 0, character: int = 0) -> types.Range:
    position = types.Position(line=line, character=character)
    return types.Range(start=position, end=position)


def _make_cli_diagnostic(
    code: str,
    message: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.Error,
) -> Diagnostic:
    return types.Diagnostic(
        range=_make_range(),
        severity=severity,
        code=code,
        source="aiconls-cli",
        message=message,
    )


def _severity_to_string(severity: DiagnosticSeverity | None) -> str:
    if severity == DiagnosticSeverity.Error:
        return "error"
    if severity == DiagnosticSeverity.Warning:
        return "warning"
    if severity == DiagnosticSeverity.Information:
        return "information"
    if severity == DiagnosticSeverity.Hint:
        return "hint"
    return "unknown"


def _iter_python_files(project_root: Path) -> list[Path]:
    python_files: list[Path] = []
    for candidate in project_root.rglob("*.py"):
        relative_parts = candidate.relative_to(project_root).parts
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts):
            continue
        python_files.append(candidate)
    return sorted(python_files)


def _parse_python_file(
    spec: Any,
    file_path: Path,
    global_ignore_annotations: DiagnosticIgnoreAnnotations,
    project_root: Path,
) -> ParsedPythonFile:
    source = file_path.read_text(encoding="utf-8")
    uri = file_path.resolve().as_uri()
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        diagnostic = _make_cli_diagnostic(
            PYTHON_SYNTAX_ERROR_CODE,
            f"Python syntax error: {exc}",
        )
        return ParsedPythonFile({}, {}, {}, [diagnostic], False)

    parser = AICONComponentParser(
        spec,
        document_uri=uri,
        project_root=project_root,
    )
    MetadataWrapper(module).visit(parser)
    return ParsedPythonFile(
        components=parser.components,
        goals=parser.goals,
        connections=parser.connections,
        diagnostics=filter_diagnostics(
            parser.diagnostics,
            source,
            global_annotations=global_ignore_annotations,
        ),
        has_build_connections_function=parser.has_build_connections_function,
    )


_UNUSED_CONNECTION_QUANTITY = re.compile(
    r"Quantity '([^']+)' from connection '([^']+)' super constructor is unused"
)


def drop_unused_quantity_warnings_used_by_components(
    diagnostics_by_uri: dict[str, list[Diagnostic]],
    spec: Any,
    components: ParsedComponents,
) -> None:
    """Drop the warnings (W202) about connection quantities that connection_func does not use but a component on that
    connection does: an estimator's f_func receives the quantities of all its connections, and components read them
    from connected_quantities."""
    used_by_connection_class: dict[str, set[str]] = {}
    for connection_name, connection_props in spec.connections.items():
        used = used_by_connection_class.setdefault(
            connection_props.class_name or connection_name, set()
        )
        for component_name in connection_props.components:
            component_props = spec.components.get(component_name)
            if component_props is None:
                continue
            parsed = components.get(component_props.class_name or component_name)
            if parsed is None:
                continue
            used.update(getattr(parsed, "f_func_params", {}).keys())
            for reference in getattr(parsed, "connected_quantity_references", []):
                if reference.connection_name == connection_name:
                    used.add(reference.quantity_name)
    for uri, diagnostics in diagnostics_by_uri.items():
        kept: list[Diagnostic] = []
        for diagnostic in diagnostics:
            match = (
                _UNUSED_CONNECTION_QUANTITY.match(diagnostic.message)
                if diagnostic.code == "W202"
                else None
            )
            if match is not None and match.group(1) in used_by_connection_class.get(
                match.group(2), set()
            ):
                continue
            kept.append(diagnostic)
        diagnostics_by_uri[uri] = kept


def add_imported_classes(
    spec: Any,
    python_files: list[Path],
    project_root: Path,
    global_ignore_annotations: DiagnosticIgnoreAnnotations,
    components: ParsedComponents,
    goals: ParsedGoals,
    connections: ParsedConnections,
) -> None:
    """Classes of the spec that the project does not define but imports from outside its root (e.g. from a sibling
    example or a shared package of the repository). Only these classes are taken from such files, and their
    diagnostics are not reported: they belong to another project."""
    wanted = {
        props.class_name or name
        for name, props in spec.components.items()
        if (props.class_name or name) not in components
    }
    wanted |= {
        props.class_name or name
        for name, props in spec.goals.items()
        if (props.class_name or name) not in goals
    }
    wanted |= {
        props.class_name or name
        for name, props in spec.connections.items()
        if (props.class_name or name) not in connections
    }
    if not wanted:
        return
    for class_file, class_names in find_imported_class_files(
        python_files, wanted, project_root
    ).items():
        parsed_file = _parse_python_file(
            spec, class_file, global_ignore_annotations, project_root
        )
        for class_name in class_names:
            if class_name in parsed_file.components:
                components.setdefault(class_name, parsed_file.components[class_name])
            if class_name in parsed_file.goals:
                goals.setdefault(class_name, parsed_file.goals[class_name])
            if class_name in parsed_file.connections:
                connections.setdefault(class_name, parsed_file.connections[class_name])


def _create_component_updater(
    component_name: str, spec_component: SpecComponentProps
) -> SensorUpdater | ActionUpdater | EstimatorUpdater | None:
    if spec_component.component_type == SpecComponentTypes.Sensor:
        return SensorUpdater(component_name, spec_component)
    if spec_component.component_type == SpecComponentTypes.Action:
        return ActionUpdater(component_name, spec_component)
    if spec_component.component_type == SpecComponentTypes.Estimator:
        return EstimatorUpdater(component_name, spec_component)
    return None


def _get_primary_spec_component_for_class(
    parsed_component: ParsedSensor | ParsedAction | ParsedEstimator,
    spec: Any,
) -> SpecComponentProps | None:
    matching_component_instances = spec.get_component_instances_by_class_name(
        parsed_component.name
    )
    if not matching_component_instances:
        return None
    primary_component_instance = matching_component_instances[0]
    return spec.components.get(primary_component_instance)


def apply_missing_quantity_definitions(project_root: Path) -> int:
    resolved_root = project_root.resolve()
    spec_path = resolved_root / "spec.yaml"
    if not spec_path.exists():
        return 0

    spec_content = spec_path.read_text(encoding="utf-8")
    parsed_spec = parse_spec_content(spec_content)
    if parsed_spec.spec is None:
        return 0

    components_fixed = 0
    for python_file in _iter_python_files(resolved_root):
        source = python_file.read_text(encoding="utf-8")
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError:
            continue

        parser = AICONComponentParser(
            parsed_spec.spec,
            document_uri=python_file.resolve().as_uri(),
            project_root=resolved_root,
        )
        MetadataWrapper(module).visit(parser)

        components_missing_definitions: list[tuple[str, SpecComponentProps]] = []
        for parsed_component in parser.components.values():
            if not isinstance(
                parsed_component, (ParsedSensor, ParsedAction, ParsedEstimator)
            ):
                continue
            if parsed_component.quantities_pos is not None:
                continue
            spec_component = _get_primary_spec_component_for_class(
                parsed_component,
                parsed_spec.spec,
            )
            if spec_component is None:
                continue
            components_missing_definitions.append(
                (parsed_component.name, spec_component)
            )

        if not components_missing_definitions:
            continue

        updated_module = module
        file_changed = False
        for component_name, spec_component in components_missing_definitions:
            updater = _create_component_updater(component_name, spec_component)
            if updater is None:
                continue
            updated_module = updated_module.visit(updater)
            if updater.updated:
                components_fixed += 1
                file_changed = True

        if file_changed:
            python_file.write_text(updated_module.code, encoding="utf-8")

    return components_fixed


def _diagnostic_to_dict(diagnostic: Diagnostic, path: Path) -> ReportDiagnostic:
    return {
        "path": str(path),
        "code": str(diagnostic.code) if diagnostic.code is not None else None,
        "message": diagnostic.message,
        "source": diagnostic.source,
        "severity": _severity_to_string(diagnostic.severity),
        "range": {
            "start": {
                "line": diagnostic.range.start.line,
                "character": diagnostic.range.start.character,
            },
            "end": {
                "line": diagnostic.range.end.line,
                "character": diagnostic.range.end.character,
            },
        },
    }


def _build_report(
    project_root: Path,
    spec_path: Path,
    python_file_count: int,
    diagnostics_by_uri: dict[str, list[Diagnostic]],
    uri_to_path: dict[str, Path],
) -> ScanReport:
    files_with_diagnostics: list[ReportFileDiagnostics] = []
    flat_diagnostics: list[ReportDiagnostic] = []

    for uri, path in sorted(uri_to_path.items(), key=lambda item: str(item[1])):
        uri_diagnostics = diagnostics_by_uri.get(uri, [])
        if not uri_diagnostics:
            continue
        serialized = [
            _diagnostic_to_dict(diagnostic, path) for diagnostic in uri_diagnostics
        ]
        files_with_diagnostics.append(
            {
                "path": str(path),
                "uri": uri,
                "diagnostics": serialized,
            }
        )
        flat_diagnostics.extend(serialized)

    error_count = sum(
        1 for diagnostic in flat_diagnostics if diagnostic["severity"] == "error"
    )
    warning_count = sum(
        1 for diagnostic in flat_diagnostics if diagnostic["severity"] == "warning"
    )

    return {
        "project_root": str(project_root),
        "spec_path": str(spec_path),
        "status": "ok" if not flat_diagnostics else "issues_found",
        "summary": {
            "python_files_scanned": python_file_count,
            "files_with_diagnostics": len(files_with_diagnostics),
            "diagnostic_count": len(flat_diagnostics),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "diagnostics_by_file": files_with_diagnostics,
        "diagnostics": flat_diagnostics,
    }


def scan_project(project_root: Path, dynamic_scan: bool = False) -> ScanReport:
    resolved_root = project_root.resolve()
    spec_path = resolved_root / "spec.yaml"
    spec_uri = spec_path.resolve().as_uri()
    uri_to_path: dict[str, Path] = {spec_uri: spec_path}
    diagnostics_by_uri: dict[str, list[Diagnostic]] = {}

    if not spec_path.exists():
        diagnostics_by_uri[spec_uri] = [
            _make_cli_diagnostic(
                SPEC_NOT_FOUND_CODE,
                "spec.yaml was not found in the project root.",
            )
        ]
        return _build_report(
            resolved_root,
            spec_path,
            python_file_count=0,
            diagnostics_by_uri=diagnostics_by_uri,
            uri_to_path=uri_to_path,
        )

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_global_ignore_annotations = parse_global_diagnostic_ignore_annotations(
        spec_content
    )
    parsed_spec = parse_spec_content(spec_content)
    raw_spec_parse_diagnostics = list(parsed_spec.diagnostics)
    diagnostics_by_uri[spec_uri] = filter_diagnostics(
        list(parsed_spec.diagnostics),
        spec_content,
    )

    if parsed_spec.spec is None:
        if not raw_spec_parse_diagnostics and not diagnostics_by_uri[spec_uri]:
            diagnostics_by_uri[spec_uri].append(
                _make_cli_diagnostic(
                    SPEC_MALFORMED_CODE,
                    "spec.yaml could not be parsed.",
                )
            )
        return _build_report(
            resolved_root,
            spec_path,
            python_file_count=0,
            diagnostics_by_uri=diagnostics_by_uri,
            uri_to_path=uri_to_path,
        )

    components: ParsedComponents = {}
    goals: ParsedGoals = {}
    connections: ParsedConnections = {}

    python_files = _iter_python_files(resolved_root)
    parsed_python_files: dict[str, ParsedPythonFile] = {}
    for python_file in python_files:
        uri = python_file.resolve().as_uri()
        uri_to_path[uri] = python_file
        parsed_python_file = _parse_python_file(
            parsed_spec.spec,
            python_file,
            spec_global_ignore_annotations,
            resolved_root,
        )
        parsed_python_files[uri] = parsed_python_file
        components.update(parsed_python_file.components)
        goals.update(parsed_python_file.goals)
        connections.update(parsed_python_file.connections)

    add_imported_classes(
        parsed_spec.spec,
        python_files,
        resolved_root,
        spec_global_ignore_annotations,
        components,
        goals,
        connections,
    )

    has_build_connections_anywhere = any(
        parsed_file.has_build_connections_function
        for parsed_file in parsed_python_files.values()
    )
    for uri, parsed_python_file in parsed_python_files.items():
        diagnostics_by_uri[uri] = _filter_build_connections_function_missing(
            list(parsed_python_file.diagnostics),
            has_build_connections_anywhere=has_build_connections_anywhere,
        )

    drop_unused_quantity_warnings_used_by_components(
        diagnostics_by_uri, parsed_spec.spec, components
    )

    spec_validator = SpecReaderValidator(
        spec_uri=spec_uri,
        spec=parsed_spec.spec,
        reader_components=components,
        reader_goals=goals,
        reader_connections=connections,
        spec_content=spec_content,
    )
    diagnostics_by_uri[spec_uri].extend(spec_validator.diagnostics)
    diagnostics_by_uri[spec_uri] = filter_diagnostics(
        diagnostics_by_uri[spec_uri],
        spec_content,
    )

    if dynamic_scan:
        # imported lazily, so that the static scan does not require torch
        from dynamic.analyzer import dynamic_analysis

        dynamic_diagnostics_by_uri = dynamic_analysis(
            parsed_spec.spec, components, goals, connections
        )
        for uri, diagnostics in dynamic_diagnostics_by_uri.items():
            diagnostics_by_uri.setdefault(uri, []).extend(diagnostics)

    return _build_report(
        resolved_root,
        spec_path,
        python_file_count=len(python_files),
        diagnostics_by_uri=diagnostics_by_uri,
        uri_to_path=uri_to_path,
    )


def format_report(report: ScanReport, pretty: bool) -> str:
    indent = 2 if pretty else None
    return json.dumps(report, indent=indent, ensure_ascii=True)


def _read_line_from_file(path: Path, zero_based_line: int) -> str | None:
    if zero_based_line < 0:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if zero_based_line >= len(lines):
        return None
    return lines[zero_based_line]


def _build_marker(start_char: int, end_char: int, line_text: str) -> str:
    if end_char <= start_char:
        end_char = start_char + 1
    if start_char >= len(line_text):
        return " " * max(len(line_text), 0) + "^"
    pointer_size = max(1, min(end_char, len(line_text)) - start_char)
    return (" " * start_char) + ("^" * pointer_size)


def _colorize(text: str, color: str, use_color: bool, *, bold: bool = False) -> str:
    if not use_color:
        return text
    prefix = f"{ANSI_BOLD}{color}" if bold else color
    return f"{prefix}{text}{ANSI_RESET}"


def format_stylish_report(report: ScanReport, use_color: bool = False) -> str:
    diagnostics_by_file = report["diagnostics_by_file"]
    summary = report["summary"]
    output_lines: list[str] = []

    for file_entry in diagnostics_by_file:
        file_path = file_entry["path"]
        diagnostics = file_entry["diagnostics"]
        output_lines.append(_colorize(file_path, ANSI_CYAN, use_color))
        path_obj = Path(file_path)
        for diagnostic in diagnostics:
            range_data = diagnostic["range"]
            code = diagnostic["code"]
            message = diagnostic["message"]
            severity = diagnostic["severity"]
            start = range_data["start"]
            end = range_data["end"]
            line = start["line"]
            character = start["character"]
            end_character = end["character"]
            display_line = line + 1
            display_column = character + 1
            sev = severity
            msg = message
            diag_code = str(code) if code is not None else "-"
            sev_color = (
                ANSI_RED
                if sev == "error"
                else ANSI_YELLOW
                if sev == "warning"
                else ANSI_CYAN
            )
            sev_label = _colorize(f"{sev:<7}", sev_color, use_color)
            code_label = _colorize(diag_code, ANSI_DIM, use_color)
            output_lines.append(
                f"  {display_line}:{display_column}  {sev_label}  {msg}  {code_label}"
            )

            source_line = _read_line_from_file(path_obj, line)
            if source_line is None:
                continue
            output_lines.append(f"    {source_line}")
            marker = _build_marker(character, end_character, source_line)
            output_lines.append(f"    {_colorize(marker, sev_color, use_color)}")

        output_lines.append("")

    diagnostic_count = summary["diagnostic_count"]
    error_count = summary["error_count"]
    warning_count = summary["warning_count"]
    summary_color = (
        ANSI_RED
        if error_count > 0
        else ANSI_YELLOW
        if warning_count > 0
        else ANSI_GREEN
    )
    summary_label = (
        f"{diagnostic_count} problems ({error_count} errors, {warning_count} warnings)"
    )
    output_lines.append(_colorize(summary_label, summary_color, use_color, bold=True))

    return "\n".join(output_lines).strip()
