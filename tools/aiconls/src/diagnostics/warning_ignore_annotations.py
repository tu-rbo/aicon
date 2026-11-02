import re
from dataclasses import dataclass

from lsprotocol import types


@dataclass(frozen=True)
class SeverityIgnoreAnnotations:
    has_file_annotation: bool
    file_codes: set[str] | None
    inline_codes: dict[int, set[str] | None]


@dataclass(frozen=True)
class DiagnosticIgnoreAnnotations:
    warning: SeverityIgnoreAnnotations
    error: SeverityIgnoreAnnotations


def _parse_codes(raw_codes: str | None) -> set[str] | None:
    if raw_codes is None:
        return None
    return {code.strip() for code in raw_codes.split(",")}


def _compile_file_pattern(annotations: tuple[str, ...]) -> re.Pattern[str]:
    annotation_pattern = "|".join(re.escape(annotation) for annotation in annotations)
    return re.compile(
        rf"^\s*#\s*aiconls:\s*(?:{annotation_pattern})(?:\s+((?:[A-Z]?\d{{3}})(?:\s*,\s*(?:[A-Z]?\d{{3}}))*))?\s*$"
    )


def _compile_inline_pattern(annotations: tuple[str, ...]) -> re.Pattern[str]:
    annotation_pattern = "|".join(re.escape(annotation) for annotation in annotations)
    return re.compile(
        rf"#\s*aiconls:\s*(?:{annotation_pattern})(?:\s+((?:[A-Z]?\d{{3}})(?:\s*,\s*(?:[A-Z]?\d{{3}}))*))?"
    )


def _normalize_codes(
    parsed_codes: set[str] | None, severity_prefix: str
) -> set[str] | None:
    if parsed_codes is None:
        return None
    normalized: set[str] = set()
    for code in parsed_codes:
        if code and code[0].isdigit():
            normalized.add(f"{severity_prefix}{code}")
            continue
        normalized.add(code)
    return normalized


def _parse_severity_ignore_annotations(
    content: str,
    file_annotations: tuple[str, ...],
    inline_annotations: tuple[str, ...],
    severity_prefix: str,
) -> SeverityIgnoreAnnotations:
    file_pattern = _compile_file_pattern(file_annotations)
    inline_pattern = _compile_inline_pattern(inline_annotations)

    has_file_annotation = False
    file_codes: set[str] | None = set()
    inline_codes: dict[int, set[str] | None] = {}

    lines = content.splitlines()
    for index, line in enumerate(lines):
        for match in inline_pattern.finditer(line):
            parsed_codes = _normalize_codes(
                _parse_codes(match.group(1)),
                severity_prefix,
            )
            if parsed_codes is None:
                inline_codes[index] = None
                continue

            existing_codes = inline_codes.get(index)
            if existing_codes is None and index in inline_codes:
                continue
            if existing_codes is None:
                inline_codes[index] = set(parsed_codes)
                continue

            existing_codes.update(parsed_codes)

    for line in lines:
        normalized_line = line.removeprefix("\ufeff")
        stripped_line = normalized_line.strip()
        if stripped_line and not stripped_line.startswith("#"):
            break
        match = file_pattern.match(normalized_line)
        if match is None:
            continue
        has_file_annotation = True
        parsed_codes = _normalize_codes(
            _parse_codes(match.group(1)),
            severity_prefix,
        )
        if parsed_codes is None:
            file_codes = None
            continue
        if file_codes is not None:
            file_codes.update(parsed_codes)

    return SeverityIgnoreAnnotations(
        has_file_annotation=has_file_annotation,
        file_codes=file_codes,
        inline_codes=inline_codes,
    )


def parse_diagnostic_ignore_annotations(content: str) -> DiagnosticIgnoreAnnotations:
    return DiagnosticIgnoreAnnotations(
        warning=_parse_severity_ignore_annotations(
            content=content,
            file_annotations=("ignore-warnings", "ignore-warning"),
            inline_annotations=("ignore-warning", "ignore-warnings"),
            severity_prefix="W",
        ),
        error=_parse_severity_ignore_annotations(
            content=content,
            file_annotations=("ignore-errors", "ignore-error"),
            inline_annotations=("ignore-error", "ignore-errors"),
            severity_prefix="E",
        ),
    )


def parse_global_diagnostic_ignore_annotations(
    content: str,
) -> DiagnosticIgnoreAnnotations:
    annotations = parse_diagnostic_ignore_annotations(content)
    return DiagnosticIgnoreAnnotations(
        warning=SeverityIgnoreAnnotations(
            has_file_annotation=annotations.warning.has_file_annotation,
            file_codes=annotations.warning.file_codes,
            inline_codes={},
        ),
        error=SeverityIgnoreAnnotations(
            has_file_annotation=annotations.error.has_file_annotation,
            file_codes=annotations.error.file_codes,
            inline_codes={},
        ),
    )


def _diagnostic_is_ignored(code: str | None, ignored_codes: set[str] | None) -> bool:
    if ignored_codes is None:
        return True
    if code is None:
        return False
    return code in ignored_codes


def _filter_by_annotations(
    diagnostic: types.Diagnostic,
    annotations: SeverityIgnoreAnnotations,
) -> bool:
    code = str(diagnostic.code) if diagnostic.code is not None else None
    if annotations.has_file_annotation and _diagnostic_is_ignored(
        code, annotations.file_codes
    ):
        return True
    line = diagnostic.range.start.line
    return line in annotations.inline_codes and _diagnostic_is_ignored(
        code, annotations.inline_codes[line]
    )


def filter_diagnostics(
    diagnostics: list[types.Diagnostic],
    content: str,
    global_annotations: DiagnosticIgnoreAnnotations | None = None,
) -> list[types.Diagnostic]:
    local_annotations = parse_diagnostic_ignore_annotations(content)
    filtered: list[types.Diagnostic] = []
    for diagnostic in diagnostics:
        if diagnostic.severity == types.DiagnosticSeverity.Warning:
            if _filter_by_annotations(diagnostic, local_annotations.warning):
                continue
            if global_annotations is not None and _filter_by_annotations(
                diagnostic, global_annotations.warning
            ):
                continue
        elif diagnostic.severity == types.DiagnosticSeverity.Error:
            if _filter_by_annotations(diagnostic, local_annotations.error):
                continue
            if global_annotations is not None and _filter_by_annotations(
                diagnostic, global_annotations.error
            ):
                continue
        else:
            filtered.append(diagnostic)
            continue

        filtered.append(diagnostic)

    return filtered


def filter_warning_diagnostics(
    diagnostics: list[types.Diagnostic], content: str
) -> list[types.Diagnostic]:
    return filter_diagnostics(diagnostics, content)
