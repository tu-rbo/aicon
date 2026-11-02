import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from lsprotocol.types import Diagnostic

import dynamic.analyzer as dynamic_analyzer_module
from aiconls import main, resolve_cli_version
from project_scanner import ScanReport, format_stylish_report, scan_project
from reader.aicon_files.reader_types import (
    ParsedComponents,
    ParsedConnections,
    ParsedGoals,
)
from reader.spec.spec import AICON_SPEC


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scan_project_reports_clean_project(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections: {}
""".strip(),
    )
    _write_file(
        tmp_path / "sensor.py",
        """
class TestSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["diagnostic_count"] == 0
    assert report["summary"]["python_files_scanned"] == 1
    assert report["diagnostics"] == []


def test_scan_project_reports_missing_implementation(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "issues_found"
    assert report["summary"]["warning_count"] == 1
    assert [diagnostic["code"] for diagnostic in report["diagnostics"]] == ["W011"]


def test_scan_project_dynamic_mode_instantiates_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections: {}
""".strip(),
    )
    _write_file(
        tmp_path / "sensor.py",
        """
class TestSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}
""".strip(),
    )
    calls: list[tuple[int, int, int]] = []

    def _fake_dynamic_analysis(
        spec: AICON_SPEC,
        parsed_components: ParsedComponents,
        parsed_goals: ParsedGoals,
        parsed_connections: ParsedConnections,
    ) -> dict[str, list[Diagnostic]]:
        _ = spec
        calls.append(
            (
                len(parsed_components),
                len(parsed_goals),
                len(parsed_connections),
            )
        )
        return {}

    monkeypatch.setattr(
        dynamic_analyzer_module, "dynamic_analysis", _fake_dynamic_analysis
    )

    report = scan_project(tmp_path, dynamic_scan=True)

    assert report["status"] == "ok"
    assert calls == [(1, 0, 0)]


def test_scan_project_skips_missing_build_connections_if_declared_elsewhere(
    tmp_path: Path,
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections:
  knownConnection:
    components: [TestSensor]
""".strip(),
    )
    _write_file(
        tmp_path / "connections.py",
        """
class knownConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"value": (1, 4)}, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        def connection_func(value):
            return value
        return connection_func
""".strip(),
    )
    _write_file(
        tmp_path / "builders.py",
        """
def build_connections():
    return {
        "knownConnection": lambda device, dtype, mockbuild: knownConnection(
            "knownConnection", device=device, dtype=dtype, mockbuild=mockbuild
        )
    }
""".strip(),
    )

    report = scan_project(tmp_path)

    assert all(diagnostic["code"] != "E206" for diagnostic in report["diagnostics"])


def test_scan_project_ignores_spec_warnings_with_file_annotation(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
# aiconls: ignore-warnings
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["diagnostic_count"] == 0


def test_scan_project_ignores_spec_warnings_inline(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor: # aiconls: ignore-warning W011
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["diagnostic_count"] == 0


def test_scan_project_ignores_python_warnings_with_file_annotation(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components: {}
goals: {}
connections: {}
""".strip(),
    )
    _write_file(
        tmp_path / "component.py",
        """
# aiconls: ignore-warnings W101
class UnknownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["diagnostic_count"] == 0


def test_scan_project_applies_spec_warning_ignores_to_python_files(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
# aiconls: ignore-warnings W101
components: {}
goals: {}
connections: {}
""".strip(),
    )
    _write_file(
        tmp_path / "component.py",
        """
class UnknownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["diagnostic_count"] == 0


def test_scan_project_ignores_spec_errors_inline(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestEstimator:
    component_type: Estimator
    goals: [MissingGoal] # aiconls: ignore-error E013
    quantities:
      estimate:
        shape: (1,)
goals:
  ReachGoal:
    class_name: ReachGoal
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["summary"]["error_count"] == 0
    assert all(diagnostic["code"] != "E013" for diagnostic in report["diagnostics"])


def test_scan_project_ignores_python_errors_with_file_annotation(tmp_path: Path):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  KnownSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections: {}
""".strip(),
    )
    _write_file(
        tmp_path / "sensor.py",
        """
# aiconls: ignore-errors E104
class KnownSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (1, 4)}

    def update(self):
        self.connections["unknownConnection"]
""".strip(),
    )

    report = scan_project(tmp_path)

    assert report["summary"]["error_count"] == 0
    assert all(diagnostic["code"] != "E104" for diagnostic in report["diagnostics"])


def test_scan_cli_prints_structured_output_and_returns_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    warning_exit_code = main(["scan", str(tmp_path), "--fail-on", "warning"])
    warning_output = capsys.readouterr().out

    assert warning_exit_code == 1
    assert str(tmp_path / "spec.yaml") in warning_output
    assert "warning" in warning_output
    assert "W011" in warning_output

    error_exit_code = main(["scan", str(tmp_path), "--fail-on", "error"])

    assert error_exit_code == 0


def test_scan_cli_json_output_remains_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    exit_code = main(
        ["scan", str(tmp_path), "--format", "json", "--fail-on", "warning"]
    )
    output = capsys.readouterr().out
    report = cast(ScanReport, json.loads(output))

    assert exit_code == 1
    assert report["summary"]["warning_count"] == 1
    assert report["diagnostics"][0]["code"] == "W011"


def test_stylish_report_shows_line_and_snippet(tmp_path: Path):
    _write_file(tmp_path / "spec.yaml", 'components: "unterminated')
    report = scan_project(tmp_path)

    output = format_stylish_report(report)

    assert str(tmp_path / "spec.yaml") in output
    assert "1:" in output
    assert "E000" in output
    assert 'components: "unterminated' in output


def test_scan_cli_stylish_output_can_be_colorized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
""".strip(),
    )

    exit_code = main(
        ["scan", str(tmp_path), "--fail-on", "warning", "--color", "always"]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "\x1b[" in output


def test_scan_cli_fix_inserts_missing_quantity_definitions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections: {}
""".strip(),
    )
    component_file = tmp_path / "sensor.py"
    _write_file(
        component_file,
        """
class TestSensor(SensorComponent):
    def __init__(self):
        super().__init__()
""".strip(),
    )

    exit_code = main(["scan", str(tmp_path), "--fix", "--fail-on", "warning"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert (
        "cls_quantity_names_and_shapes = {'value': (1, 4)}"
        in component_file.read_text(encoding="utf-8")
    )


def test_scan_cli_fix_does_not_overwrite_existing_quantity_definitions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_file(
        tmp_path / "spec.yaml",
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1, 4)
goals: {}
connections: {}
""".strip(),
    )
    component_file = tmp_path / "sensor.py"
    _write_file(
        component_file,
        """
class TestSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"value": (2, 4)}
""".strip(),
    )

    exit_code = main(["scan", str(tmp_path), "--fix", "--fail-on", "error"])
    _ = capsys.readouterr()

    assert exit_code == 1
    assert (
        'cls_quantity_names_and_shapes = {"value": (2, 4)}'
        in component_file.read_text(encoding="utf-8")
    )


def test_cli_version_option_prints_version(capsys: pytest.CaptureFixture[str]):
    expected_version = resolve_cli_version()

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    output = capsys.readouterr().out.strip()
    assert exc_info.value.code == 0
    assert output == f"aiconls {expected_version}"


def test_static_scan_does_not_require_torch(tmp_path: Path) -> None:
    (tmp_path / "spec.yaml").write_text(
        "components:\n  Sensor:\n    component_type: Sensor\n", encoding="utf-8"
    )
    (tmp_path / "sensors.py").write_text(
        "from aicon_base.components.sensor_components import SensorComponent\n\n\n"
        "class Sensor(SensorComponent):\n    pass\n",
        encoding="utf-8",
    )
    src_dir = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import sys\n"
        "sys.modules['torch'] = None  # makes every 'import torch' fail\n"
        "from project_scanner import scan_project\n"
        f"report = scan_project(__import__('pathlib').Path({str(tmp_path)!r}))\n"
        "print(report['status'])\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=src_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] in {"ok", "issues_found"}
