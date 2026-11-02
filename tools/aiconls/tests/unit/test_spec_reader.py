import pytest
from yaml.error import YAMLError

from reader.spec import spec_reader
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import (
    SpecComponentQuantityCategorical,
    SpecComponentQuantityGaussian,
    SpecComponentQuantityParticle,
    SpecComponentQuantityReference,
    SpecComponentTypes,
    SpecQuantityTypes,
)


def test_parse_spec_content_loads_sample_project(sample_spec_content: str):
    spec_parser = parse_spec_content(sample_spec_content)

    assert spec_parser.spec is not None
    assert set(spec_parser.spec.components.keys()) == {"TestSensor", "TestEstimator"}
    assert (
        spec_parser.spec.components["TestSensor"].component_type
        == SpecComponentTypes.Sensor
    )
    assert spec_parser.spec.components["TestSensor"].quantities["test_value"].shape == (
        1,
        4,
    )
    assert len(spec_parser.diagnostics) == 0


def test_parse_spec_content_loads_complex_fixture(complex_spec_content: str):
    parsed_spec = parse_spec_content(complex_spec_content)

    assert parsed_spec.spec is not None
    assert set(parsed_spec.spec.components.keys()) == {
        "EEPoseSensor",
        "DrawerPoseEstimator",
        "VeloEEAction",
    }
    assert (
        parsed_spec.spec.components["VeloEEAction"].component_type
        == SpecComponentTypes.Action
    )
    assert parsed_spec.spec.components["EEPoseSensor"].quantities["pose_ee"].shape == (
        6,
    )
    assert parsed_spec.spec.components["DrawerPoseEstimator"].quantities[
        "uncertainty_drawer"
    ].shape == (3, 3)
    assert parsed_spec.spec.components["DrawerPoseEstimator"].goals == [
        "DrawerOpenViaJointGoal"
    ]
    assert parsed_spec.spec.components["VeloEEAction"].goals == ["safetyGoal"]
    assert parsed_spec.spec.goals["drawerOpeningGoal"].name == "DrawerOpenViaJointGoal"
    assert parsed_spec.spec.connections["eeDrawerDistance"].components == [
        "EEPoseSensor",
        "DrawerPoseEstimator",
    ]
    assert parsed_spec.spec.connections["eeControlLoop"].components == [
        "VeloEEAction",
        "EEPoseSensor",
        "DrawerPoseEstimator",
    ]
    assert parsed_spec.diagnostics == []


def test_parse_spec_content_supports_goals_and_connections():
    parsed = parse_spec_content(
        """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      test_value:
        shape: (1, 4)
goals:
  ReachGoal:
    class_name: ReachGoal
connections:
  testConnection:
    components: [TestSensor]
""".strip()
    )

    assert parsed.spec is not None
    assert set(parsed.spec.goals.keys()) == {"ReachGoal"}
    assert parsed.spec.goals["ReachGoal"].name == "ReachGoal"
    assert parsed.spec.connections["testConnection"].components == ["TestSensor"]


def test_parse_spec_content_supports_component_goals():
    parsed = parse_spec_content(
        """
components:
  TestEstimator:
    component_type: Estimator
    goals: [ReachGoal]
    quantities:
      estimate:
        shape: (1,)
goals:
  ReachGoal:
    class_name: ReachGoal
""".strip()
    )

    assert parsed.spec is not None
    assert parsed.spec.components["TestEstimator"].goals == ["ReachGoal"]


def test_parse_spec_content_supports_gaussian_quantity_type():
    parsed = parse_spec_content(
        """
components:
  EEPoseEstimator:
    component_type: Estimator
    quantities:
      pose_ee:
        q_type: Gaussian
        dim: 3
        uncertainty_name: uncertainty_ee
""".strip()
    )

    assert parsed.spec is not None
    assert parsed.diagnostics == []
    quantity = parsed.spec.components["EEPoseEstimator"].quantities["pose_ee"]
    assert isinstance(quantity, SpecComponentQuantityGaussian)
    assert quantity.q_type == SpecQuantityTypes.Gaussian
    assert quantity.dim == 3
    assert quantity.shape == (3,)
    assert quantity.uncertainty_name == "uncertainty_ee"

    uncertainty_quantity = parsed.spec.components["EEPoseEstimator"].quantities[
        "uncertainty_ee"
    ]
    assert isinstance(uncertainty_quantity, SpecComponentQuantityReference)
    assert uncertainty_quantity.q_type == SpecQuantityTypes.Reference
    assert uncertainty_quantity.shape == (3, 3)
    assert uncertainty_quantity.referenced_quantity_name == "pose_ee"


def test_parse_spec_content_supports_categorical_quantity_type():
    parsed = parse_spec_content(
        """
components:
  GraspEstimator:
    component_type: Estimator
    quantities:
      grasp_mode:
        q_type: Categorical
        dim: 5
""".strip()
    )

    assert parsed.spec is not None
    assert parsed.diagnostics == []
    quantity = parsed.spec.components["GraspEstimator"].quantities["grasp_mode"]
    assert isinstance(quantity, SpecComponentQuantityCategorical)
    assert quantity.q_type == SpecQuantityTypes.Categorical
    assert quantity.dim == 5
    assert quantity.shape == (5,)


def test_parse_spec_content_supports_particle_quantity_type():
    parsed = parse_spec_content(
        """
components:
  PoseParticlesEstimator:
    component_type: Estimator
    quantities:
      pose_particles:
        q_type: Particle
        dim: 3
        particle_count: 64
        particle_weights_name: pose_weights
""".strip()
    )

    assert parsed.spec is not None
    assert parsed.diagnostics == []
    quantity = parsed.spec.components["PoseParticlesEstimator"].quantities[
        "pose_particles"
    ]
    assert isinstance(quantity, SpecComponentQuantityParticle)
    assert quantity.q_type == SpecQuantityTypes.Particle
    assert quantity.dim == 3
    assert quantity.particle_count == 64
    assert quantity.particle_weights_name == "pose_weights"
    assert quantity.shape == (64, 3)

    weights_quantity = parsed.spec.components["PoseParticlesEstimator"].quantities[
        "pose_weights"
    ]
    assert isinstance(weights_quantity, SpecComponentQuantityReference)
    assert weights_quantity.q_type == SpecQuantityTypes.Reference
    assert weights_quantity.shape == (64,)
    assert weights_quantity.referenced_quantity_name == "pose_particles"


def test_parse_spec_content_supports_explicit_class_names():
    parsed = parse_spec_content(
        """
components:
  sensor_instance:
    class_name: TestSensor
    component_type: Sensor
    quantities:
      test_value:
        shape: (1, 4)
goals:
  goal_instance:
    class_name: ReachGoal
connections:
  connection_instance:
    class_name: TestConnection
    components: [sensor_instance]
""".strip()
    )

    assert parsed.spec is not None
    component = parsed.spec.components["sensor_instance"]
    assert component.instance_name == "sensor_instance"
    assert component.class_name == "TestSensor"

    goal = parsed.spec.goals["goal_instance"]
    assert goal.instance_name == "goal_instance"
    assert goal.class_name == "ReachGoal"

    connection = parsed.spec.connections["connection_instance"]
    assert connection.instance_name == "connection_instance"
    assert connection.class_name == "TestConnection"


def test_parse_spec_content_rejects_malformed_yaml():
    parsed = parse_spec_content('components: "unterminated')

    assert parsed.spec is None
    assert len(parsed.diagnostics) == 1
    diagnostic = parsed.diagnostics[0]
    assert diagnostic.code == "E000"
    assert "Malformed spec." in diagnostic.message
    assert diagnostic.range.start.line >= 0
    assert diagnostic.range.start.character >= 0


def test_parse_spec_content_handles_empty_file_as_empty_spec() -> None:
    parsed = parse_spec_content("")

    assert parsed.spec is not None
    assert parsed.spec.components == {}
    assert parsed.spec.goals == {}
    assert parsed.spec.connections == {}
    assert parsed.diagnostics == []


@pytest.mark.parametrize(
    ("content"),
    [
        (
            """
components:
  TestComponent:
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
""".strip()
        )
    ],
)
def test_parse_spec_content_valid(content: str):
    parsed_spec = parse_spec_content(content)

    assert parsed_spec.spec is not None
    assert len(parsed_spec.diagnostics) == 0


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (
            """
components:
  BrokenSensor:
    component_type: Unknown
    quantities:
      value:
        shape: (1, 4)
""".strip(),
            "E001",
        ),
        (
            """
components:
  BrokenSensor:
    component_type: Sensor
    dtype: not-a-real-dtype
    quantities:
      value:
        shape: (1, 4)
""".strip(),
            "E002",
        ),
        (
            """
components:
  BrokenSensor:
    component_type: Sensor
    quantities:
      value:
        shape: ()
""".strip(),
            "E003",
        ),
        (
            """
components:
  SomeGoal:
    component_type: Goal
""".strip(),
            "E001",
        ),
        (
            """
components:
  BrokenEstimator:
    component_type: Estimator
    quantities:
      state:
        q_type: Gaussian
        dim: 0
        uncertainty_name: state_cov
""".strip(),
            "E003",
        ),
        (
            """
components:
  BrokenEstimator:
    component_type: Estimator
    quantities:
      particles:
        q_type: Particle
        dim: 2
        particle_count: 0
        particle_weights_name: weights
""".strip(),
            "E003",
        ),
        (
            """
components:
  BrokenEstimator:
    component_type: Estimator
    quantities:
      state:
        q_type: Gaussian
        dim: 3
        uncertainty_name: state_cov
      state_cov:
        shape: (3, 3)
""".strip(),
            "E003",
        ),
        (
            """
components:
  BrokenEstimator:
    component_type: Estimator
    quantities:
      state:
        q_type: Gaussian
        dim: 3
        uncertainty_name: state
""".strip(),
            "E003",
        ),
    ],
)
def test_parse_spec_content_reports_malformed_spec_errors(
    content: str, expected_code: str
):
    parsed_spec = parse_spec_content(content)

    assert parsed_spec.spec is None
    assert [diagnostic.code for diagnostic in parsed_spec.diagnostics] == [
        expected_code
    ]


def test_parse_spec_content_allows_missing_components_field() -> None:
    parsed = parse_spec_content(
        """
goals:
  ReachGoal:
    class_name: ReachGoal
""".strip()
    )

    assert parsed.spec is not None
    assert parsed.spec.components == {}
    assert parsed.spec.goals["ReachGoal"].name == "ReachGoal"
    assert parsed.spec.connections == {}
    assert parsed.diagnostics == []


def test_parse_spec_content_reports_missing_component_type() -> None:
    parsed = parse_spec_content(
        """
components:
  TestSensor:
    quantities:
      value:
        shape: (1, 4)
""".strip()
    )

    assert parsed.spec is None
    assert len(parsed.diagnostics) == 1
    diagnostic = parsed.diagnostics[0]
    assert diagnostic.code == "E001"
    assert "Missing required scalar field 'component_type'" in diagnostic.message


@pytest.mark.parametrize(
    ("content", "expected_code", "expected_name"),
    [
        (
            """
components:
  TestSensor:
    component_type: Sensor
  TestSensor:
    component_type: Sensor
""".strip(),
            "E004",
            "TestSensor",
        ),
        (
            """
components:
  TestSensor:
    component_type: Sensor
goals:
  ReachGoal:
    class_name: ReachGoal
  ReachGoal:
    class_name: ReachGoal
""".strip(),
            "E005",
            "ReachGoal",
        ),
        (
            """
components:
  TestSensor:
    component_type: Sensor
connections:
  testConnection:
    components: [TestSensor]
  testConnection:
    components: [TestSensor]
""".strip(),
            "E006",
            "testConnection",
        ),
    ],
)
def test_parse_spec_content_reports_duplicate_entries(
    content: str, expected_code: str, expected_name: str
) -> None:
    parsed = parse_spec_content(content)

    assert parsed.spec is None
    assert [diagnostic.code for diagnostic in parsed.diagnostics] == [expected_code]
    assert expected_name in parsed.diagnostics[0].message


def test_parse_spec_content_uses_zero_position_without_yaml_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_yaml_error(_content: str):
        raise YAMLError("forced parser failure")

    monkeypatch.setattr(spec_reader.yaml, "compose", _raise_yaml_error)
    parsed = parse_spec_content("components: {}")

    assert parsed.spec is None
    assert len(parsed.diagnostics) == 1
    diagnostic = parsed.diagnostics[0]
    assert diagnostic.code == "E000"
    assert diagnostic.range.start.line == 0
    assert diagnostic.range.start.character == 0


@pytest.mark.parametrize(
    ("content", "expected_code", "unexpected_property"),
    [
        (
            """
components:
  TestSensor:
    component_type: Sensor
    unsupported: value
""".strip(),
            "E007",
            "unsupported",
        ),
        (
            """
components:
  TestSensor:
    component_type: Sensor
goals:
  ReachGoal:
    name: ReachGoal
""".strip(),
            "E008",
            "name",
        ),
        (
            """
components:
  TestSensor:
    component_type: Sensor
goals:
  ReachGoal:
    class_name: ReachGoal
    unsupported: value
""".strip(),
            "E008",
            "unsupported",
        ),
        (
            """
components:
  TestSensor:
    component_type: Sensor
connections:
  testConnection:
    components: [TestSensor]
    unsupported: value
""".strip(),
            "E009",
            "unsupported",
        ),
    ],
)
def test_parse_spec_content_reports_unexpected_entry_properties(
    content: str, expected_code: str, unexpected_property: str
) -> None:
    parsed = parse_spec_content(content)

    assert parsed.spec is None
    diagnostic_codes = [diagnostic.code for diagnostic in parsed.diagnostics]
    assert expected_code in diagnostic_codes
    assert any(
        unexpected_property in diagnostic.message for diagnostic in parsed.diagnostics
    )


def test_parse_spec_content_keeps_normalized_dtype() -> None:
    parsed_spec = parse_spec_content(
        """
components:
  DoubleSensor:
    component_type: Sensor
    dtype: torch.float64
  DefaultSensor:
    component_type: Sensor
""".strip()
    )

    assert parsed_spec.spec is not None
    assert parsed_spec.spec.components["DoubleSensor"].dtype == "float64"
    assert parsed_spec.spec.components["DefaultSensor"].dtype is None
