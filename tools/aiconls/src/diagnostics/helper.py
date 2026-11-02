from typing import Any

from lsprotocol.types import Diagnostic, Range

from diagnostics.diagnostics import (
    DiagnosticDefinition,
    ErrorCode,
    get_diagnostic_definition,
    get_diagnostic_definition_by_name,
)


def make_diagnostic(code: ErrorCode, range: Range, **kwargs: Any) -> Diagnostic:
    definition = get_diagnostic_definition(code)

    return _build_diagnostic(definition, range, **kwargs)


def make_diagnostic_by_name(name: str, range: Range, **kwargs: Any) -> Diagnostic:
    definition = get_diagnostic_definition_by_name(name)

    return _build_diagnostic(definition, range, **kwargs)


def _build_diagnostic(
    definition: DiagnosticDefinition,
    range: Range,
    **kwargs: Any,
) -> Diagnostic:
    return Diagnostic(
        range=range,
        severity=definition.severity,
        code=definition.code,
        source="aiconls",
        message=definition.template.format(**kwargs),
    )
