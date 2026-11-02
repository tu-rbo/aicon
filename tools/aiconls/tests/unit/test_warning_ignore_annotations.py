from lsprotocol import types

from diagnostics.warning_ignore_annotations import (
    filter_diagnostics,
    parse_global_diagnostic_ignore_annotations,
)


def _diagnostic(
    code: str, severity: types.DiagnosticSeverity, line: int
) -> types.Diagnostic:
    position = types.Position(line=line, character=0)
    return types.Diagnostic(
        range=types.Range(start=position, end=position),
        severity=severity,
        code=code,
        source="aiconls",
        message="diagnostic",
    )


def test_filter_diagnostics_ignores_file_warnings() -> None:
    diagnostics = [
        _diagnostic("W011", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("E012", types.DiagnosticSeverity.Error, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
# aiconls: ignore-warnings
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["E012"]


def test_filter_diagnostics_ignores_selected_warning_codes() -> None:
    diagnostics = [
        _diagnostic("W011", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
# aiconls: ignore-warnings W011
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W013"]


def test_filter_diagnostics_ignores_selected_warning_codes_with_utf8_bom() -> None:
    diagnostics = [
        _diagnostic("W011", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
\ufeff# aiconls: ignore-warnings W011
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W013"]


def test_filter_diagnostics_ignores_selected_warning_codes_in_file_header_singular() -> (
    None
):
    diagnostics = [
        _diagnostic("W202", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
# aiconls: ignore-warning W202
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W013"]


def test_filter_diagnostics_ignores_selected_warning_codes_with_numeric_shorthand() -> (
    None
):
    diagnostics = [
        _diagnostic("W202", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
# aiconls: ignore-warning 202
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W013"]


def test_filter_diagnostics_ignores_inline_warnings() -> None:
    diagnostics = [
        _diagnostic("W011", types.DiagnosticSeverity.Warning, line=1),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=2),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
components:
  TestSensor: # aiconls: ignore-warning W011
  TestConnection: # aiconls: ignore-warning
""".strip(),
    )

    assert filtered == []


def test_filter_diagnostics_ignores_file_errors() -> None:
    diagnostics = [
        _diagnostic("E012", types.DiagnosticSeverity.Error, line=1),
        _diagnostic("W011", types.DiagnosticSeverity.Warning, line=1),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
# aiconls: ignore-errors
components: {}
""".strip(),
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W011"]


def test_filter_diagnostics_ignores_selected_error_codes_inline() -> None:
    diagnostics = [
        _diagnostic("E012", types.DiagnosticSeverity.Error, line=1),
        _diagnostic("E013", types.DiagnosticSeverity.Error, line=2),
    ]

    filtered = filter_diagnostics(
        diagnostics,
        """
components:
  testConnection: # aiconls: ignore-error E012
  testGoal: # aiconls: ignore-error
""".strip(),
    )

    assert filtered == []


def test_filter_diagnostics_applies_global_spec_header_ignores() -> None:
    diagnostics = [
        _diagnostic("W202", types.DiagnosticSeverity.Warning, line=7),
        _diagnostic("W013", types.DiagnosticSeverity.Warning, line=7),
    ]

    global_annotations = parse_global_diagnostic_ignore_annotations(
        """
# aiconls: ignore-warnings W202
components: {}
""".strip()
    )

    filtered = filter_diagnostics(
        diagnostics,
        "class SomeConnection(ActiveInterconnection):\n    pass",
        global_annotations=global_annotations,
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W013"]


def test_filter_diagnostics_does_not_apply_spec_inline_ignores_globally() -> None:
    diagnostics = [_diagnostic("W202", types.DiagnosticSeverity.Warning, line=7)]

    global_annotations = parse_global_diagnostic_ignore_annotations(
        """
components:
  TestConnection: # aiconls: ignore-warning W202
""".strip()
    )

    filtered = filter_diagnostics(
        diagnostics,
        "class SomeConnection(ActiveInterconnection):\n    pass",
        global_annotations=global_annotations,
    )

    assert [diagnostic.code for diagnostic in filtered] == ["W202"]
