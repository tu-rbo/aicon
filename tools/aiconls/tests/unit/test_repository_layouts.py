"""Projects laid out like the examples of the AICON repository: shapes from constants and of variable size, annotated
quantity definitions, connection builders returned directly, classes shared between examples, inherited quantity
definitions, switching conditions and batched inputs. The language server reports the same as the scan."""

from pathlib import Path

from lsprotocol import types
from pytest import MonkeyPatch

from project_scanner import ScanReport, scan_project
from server import AICONLanguageServer


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _codes(report: ScanReport) -> list[str]:
    return sorted(str(d["code"]) for d in report["diagnostics"])


def _server_codes(project_root: Path, monkeypatch: MonkeyPatch) -> list[str]:
    """The diagnostics the language server publishes after a spec.yaml of the project is opened."""
    monkeypatch.setenv("AICONLS_PROJECT_ROOT", str(project_root))
    ls = AICONLanguageServer("layout-server", "v1")

    def _publish(_: types.PublishDiagnosticsParams) -> None:
        pass

    monkeypatch.setattr(ls, "text_document_publish_diagnostics", _publish)
    ls.update_spec()
    ls.parse_all_project_python_files()
    ls.validate_spec()
    return sorted(
        str(diagnostic.code)
        for _, diagnostics in ls.diagnostic_sink.diagnostics.values()
        for diagnostic in diagnostics
    )


def test_constant_and_variable_sizes(tmp_path: Path) -> None:
    _write(
        tmp_path / "spec.yaml",
        """
components:
  PointSensor:
    component_type: Sensor
    quantities:
      points:
        shape: (-1, 3)
      grid:
        shape: (10, 10)
""",
    )
    _write(
        tmp_path / "sensors.py",
        """
NUM_BLOCKS = 10


class PointSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"points": (0, 3), "grid": (NUM_BLOCKS, NUM_BLOCKS)}
""",
    )

    assert _codes(scan_project(tmp_path)) == []


def test_annotated_quantities_and_returned_connection_builders(tmp_path: Path) -> None:
    _write(
        tmp_path / "spec.yaml",
        """
components:
  ValueSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (2,)
connections:
  Measurement:
    class_name: MeasurementConnection
    components: [ValueSensor]
""",
    )
    _write(
        tmp_path / "graph.py",
        """
from typing import Dict, Tuple


class ValueSensor(SensorComponent):
    cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {"value": (2,)}


class MeasurementConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(name, {"value": (2,)}, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):
        def connection_func(value):
            return value

        return connection_func


def build_connections():
    return {
        "Measurement": lambda device, dtype, mockbuild: MeasurementConnection(
            "Measurement", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
""",
    )

    codes = _codes(scan_project(tmp_path))
    assert "W103" not in codes and "E203" not in codes


def test_classes_from_a_sibling_package_and_inherited_quantities(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _write(repository / "examples" / "__init__.py", "")
    _write(repository / "examples" / "shared" / "__init__.py", "")
    _write(
        repository / "examples" / "shared" / "sensors.py",
        """
class SimulationSensor(SensorComponent):
    pass


class PositionSensor(SimulationSensor):
    cls_quantity_names_and_shapes = {"position": (3,)}
""",
    )
    example = repository / "examples" / "example"
    _write(example / "__init__.py", "")
    _write(
        example / "spec.yaml",
        """
components:
  Position:
    component_type: Sensor
    class_name: PositionSensor
    quantities:
      position:
        shape: (3,)
  NoisyPosition:
    component_type: Sensor
    class_name: NoisyPositionSensor
    quantities:
      position:
        shape: (3,)
""",
    )
    _write(
        example / "sensors.py",
        """
from ..shared.sensors import PositionSensor


class NoisyPositionSensor(PositionSensor):
    pass
""",
    )

    codes = _codes(scan_project(example))
    # PositionSensor is found in the sibling package, NoisyPositionSensor inherits its quantities
    assert "W011" not in codes and "W103" not in codes


def test_switching_conditions_and_batched_inputs_count_as_used(tmp_path: Path) -> None:
    _write(
        tmp_path / "spec.yaml",
        """
components:
  Estimator:
    component_type: Estimator
    class_name: ContactEstimator
    quantities:
      pusher_point:
        shape: (2,)
      grasped:
        shape: (1,)
connections:
  Contact:
    class_name: ContactConnection
    components: [Estimator]
""",
    )
    _write(
        tmp_path / "graph.py",
        """
class ContactEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"pusher_point": (2,), "grasped": (1,)}


class ContactConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {"pusher_point": (2,), "grasped": (1,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            required_signs_dict={"pusher_point_batched": {"grasped": 1}},
        )

    def define_implicit_connection_function(self):
        def connection_func(pusher_point_batched):
            return pusher_point_batched

        return connection_func
""",
    )

    assert "W202" not in _codes(scan_project(tmp_path))


def test_goal_not_attached_and_local_base_class(tmp_path: Path) -> None:
    _write(
        tmp_path / "spec.yaml",
        """
components:
  ValueSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
goals:
  AlternativeGoal:
    class_name: AlternativeGoal
""",
    )
    _write(
        tmp_path / "graph.py",
        """
class CommonSensor(SensorComponent):
    pass


class ValueSensor(CommonSensor):
    cls_quantity_names_and_shapes = {"value": (1,)}


class AlternativeGoal(Goal):
    def define_goal_cost_function(self):
        def goal_func(value):
            return (value,), (value,)

        return goal_func
""",
    )

    codes = _codes(scan_project(tmp_path))
    # the goal is one an experiment may be run with, attached to no component; CommonSensor is a building block
    assert "E103" not in codes and "W101" not in codes


def test_server_finds_shared_classes_and_quantities_used_by_components(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _write(repository / "examples" / "__init__.py", "")
    _write(repository / "examples" / "shared" / "__init__.py", "")
    _write(
        repository / "examples" / "shared" / "sensors.py",
        """
class PositionSensor(SensorComponent):
    cls_quantity_names_and_shapes = {"position": (3,)}
""",
    )
    example = repository / "examples" / "example"
    _write(example / "__init__.py", "")
    _write(
        example / "spec.yaml",
        """
components:
  Position:
    component_type: Sensor
    class_name: PositionSensor
    quantities:
      position:
        shape: (3,)
  Tracker:
    component_type: Estimator
    class_name: TrackerEstimator
    quantities:
      estimate:
        shape: (3,)
connections:
  Tracking:
    class_name: TrackingConnection
    components: [Position, Tracker]
""",
    )
    _write(
        example / "graph.py",
        """
from ..shared.sensors import PositionSensor


class TrackerEstimator(EstimationComponent):
    cls_quantity_names_and_shapes = {"estimate": (3,)}

    def define_estimation_function_f(self):
        def f_func(estimate, position):
            return estimate + position

        return f_func, [], []


class TrackingConnection(ActiveInterconnection):
    def __init__(self, name, dtype=None, device=None, mockbuild=False):
        super().__init__(
            name,
            {"position": (3,), "estimate": (3,)},
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
        )

    def define_implicit_connection_function(self):
        def connection_func(estimate):
            return estimate

        return connection_func


def build_connections():
    return {
        "Tracking": lambda device, dtype, mockbuild: TrackingConnection(
            "Tracking", device=device, dtype=dtype, mockbuild=mockbuild
        ),
    }
""",
    )

    # PositionSensor is defined in a sibling package; position is used by the estimator, not by connection_func
    assert _codes(scan_project(example)) == []
    assert _server_codes(example, monkeypatch) == []
