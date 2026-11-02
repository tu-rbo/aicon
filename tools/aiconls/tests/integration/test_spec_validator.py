import pytest
from lsprotocol import types

from reader.aicon_files.reader_types import (
    ParsedComponent,
    ParsedConnection,
    ParsedGoal,
)
from reader.spec.spec_reader import parse_spec_content
from reader.spec.spec_types import SpecComponentTypes
from validator.spec_validator import SpecReaderValidator


def _make_range() -> types.Range:
    return types.Range(
        start=types.Position(line=1, character=0),
        end=types.Position(line=1, character=1),
    )


@pytest.mark.integration
def test_spec_validator_flags_missing_components(sample_spec_content: str):
    parsed_spec = parse_spec_content(sample_spec_content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        sample_spec_content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W011",
        "W013",
    ]


@pytest.mark.integration
def test_spec_validator_reports_undefined_components_in_connections(
    sample_spec_content: str,
):
    content = sample_spec_content.replace(
        "testConnection:\n    components: [TestSensor, TestEstimator]",
        "testConnection:\n    components: [TestSensor, TestEstimator]\n  brokenConnection:\n    components: [TestSensor, GhostComponent]",
    )
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W011",
        "W013",
        "W013",
        "E012",
    ]
    undefined_component_diagnostic = validator.diagnostics[-1]
    assert (
        undefined_component_diagnostic.message
        == "Component 'GhostComponent' not found in spec."
    )

    missing_name = "GhostComponent"
    missing_start = content.index(missing_name)
    missing_end = missing_start + len(missing_name)
    expected_range = types.Range(
        start=types.Position(
            line=content.count("\n", 0, missing_start),
            character=missing_start - (content.rfind("\n", 0, missing_start) + 1),
        ),
        end=types.Position(
            line=content.count("\n", 0, missing_end),
            character=missing_end - (content.rfind("\n", 0, missing_end) + 1),
        ),
    )
    assert undefined_component_diagnostic.range == expected_range


@pytest.mark.integration
def test_spec_validator_handles_inline_comments_on_connection_headers():
    content = """
components:
  GripperAction:
    component_type: Action
    quantities:
      force:
        shape: (1,)
connections:
  DistGraspHandConnection: # inline comment
    components: [GripperAction, EEForceSensoar]
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W013",
        "E012",
    ]
    assert (
        validator.diagnostics[-1].message
        == "Component 'EEForceSensoar' not found in spec."
    )


@pytest.mark.integration
def test_spec_validator_flags_missing_goals_and_connections():
    content = """
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
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W012",
        "W013",
    ]


@pytest.mark.integration
def test_spec_validator_handles_complex_fixture_connections(complex_spec_content: str):
    parsed_spec = parse_spec_content(complex_spec_content)
    assert parsed_spec.spec is not None

    reader_components = {
        "EEPoseSensor": ParsedComponent(
            component_type=SpecComponentTypes.Sensor,
            quantities={},
            goals=[],
            instance_name="EEPoseSensor",
            class_name="EEPoseSensor",
            position=_make_range(),
            document_uri="file:///components.py",
        ),
        "DrawerPoseEstimator": ParsedComponent(
            component_type=SpecComponentTypes.Estimator,
            quantities={},
            goals=[],
            instance_name="DrawerPoseEstimator",
            class_name="DrawerPoseEstimator",
            position=_make_range(),
            document_uri="file:///components.py",
        ),
        "VeloEEAction": ParsedComponent(
            component_type=SpecComponentTypes.Action,
            quantities={},
            goals=[],
            instance_name="VeloEEAction",
            class_name="VeloEEAction",
            position=_make_range(),
            document_uri="file:///components.py",
        ),
    }
    reader_goals = {
        "DrawerOpenViaJointGoal": ParsedGoal(
            name="DrawerOpenViaJointGoal",
            position=_make_range(),
            document_uri="file:///goals.py",
            goal_func_params={},
        ),
        "KeepEEInWorkspaceGoal": ParsedGoal(
            name="KeepEEInWorkspaceGoal",
            position=_make_range(),
            document_uri="file:///goals.py",
            goal_func_params={},
        ),
    }
    reader_connections = {
        "eeDrawerDistance": ParsedConnection(
            name="eeDrawerDistance",
            position=_make_range(),
            document_uri="file:///connections.py",
            quantities={},
            quantities_pos=_make_range(),
            connection_func_params={},
        ),
        "eeControlLoop": ParsedConnection(
            name="eeControlLoop",
            position=_make_range(),
            document_uri="file:///connections.py",
            quantities={},
            quantities_pos=_make_range(),
            connection_func_params={},
        ),
    }

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        reader_components,
        reader_goals,
        reader_connections,
        complex_spec_content,
    )

    assert validator.diagnostics == []


@pytest.mark.integration
def test_spec_validator_reports_undefined_goal_in_component():
    content = """
components:
  TestEstimator:
    component_type: Estimator
    goals: [MissingGoal]
    quantities:
      estimate:
        shape: (1,)
goals:
  ReachGoal:
    class_name: ReachGoal
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W012",
        "E013",
    ]


@pytest.mark.integration
def test_spec_validator_reports_duplicate_component_in_connection_components():
    content = """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
connections:
  duplicateConnection:
    components: [TestSensor, TestSensor]
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W013",
        "E014",
    ]
    duplicate_diagnostic = validator.diagnostics[-1]
    assert (
        duplicate_diagnostic.message
        == "Connection 'duplicateConnection' contains duplicate component "
        "'TestSensor' in its components list."
    )

    duplicate_name = "TestSensor"
    components_line = "components: [TestSensor, TestSensor]"
    components_line_start = content.index(components_line)
    first_occurrence = content.index(duplicate_name, components_line_start)
    second_occurrence = content.index(duplicate_name, first_occurrence + 1)
    expected_range = types.Range(
        start=types.Position(
            line=content.count("\n", 0, second_occurrence),
            character=second_occurrence
            - (content.rfind("\n", 0, second_occurrence) + 1),
        ),
        end=types.Position(
            line=content.count("\n", 0, second_occurrence + len(duplicate_name)),
            character=(
                second_occurrence
                + len(duplicate_name)
                - (content.rfind("\n", 0, second_occurrence + len(duplicate_name)) + 1)
            ),
        ),
    )
    assert duplicate_diagnostic.range == expected_range


@pytest.mark.integration
def test_spec_validator_reports_undefined_component_in_block_connection_components():
    content = """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
connections:
  brokenConnection:
    components:
      - TestSensor
      - GhostComponent
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W013",
        "E012",
    ]
    undefined_component_diagnostic = validator.diagnostics[-1]
    assert (
        undefined_component_diagnostic.message
        == "Component 'GhostComponent' not found in spec."
    )

    missing_name = "GhostComponent"
    missing_start = content.index(missing_name)
    missing_end = missing_start + len(missing_name)
    expected_range = types.Range(
        start=types.Position(
            line=content.count("\n", 0, missing_start),
            character=missing_start - (content.rfind("\n", 0, missing_start) + 1),
        ),
        end=types.Position(
            line=content.count("\n", 0, missing_end),
            character=missing_end - (content.rfind("\n", 0, missing_end) + 1),
        ),
    )
    assert undefined_component_diagnostic.range == expected_range


@pytest.mark.integration
def test_spec_validator_reports_duplicate_component_in_block_connection_components():
    content = """
components:
  TestSensor:
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
connections:
  duplicateConnection:
    components:
      - TestSensor
      - TestSensor
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        {},
        {},
        {},
        content,
    )

    assert [diagnostic.code for diagnostic in validator.diagnostics] == [
        "W011",
        "W013",
        "E014",
    ]
    duplicate_diagnostic = validator.diagnostics[-1]
    assert (
        duplicate_diagnostic.message
        == "Connection 'duplicateConnection' contains duplicate component "
        "'TestSensor' in its components list."
    )

    duplicate_name = "TestSensor"
    list_items_start = content.index("duplicateConnection:")
    first_occurrence = content.index(duplicate_name, list_items_start)
    second_occurrence = content.index(duplicate_name, first_occurrence + 1)
    expected_range = types.Range(
        start=types.Position(
            line=content.count("\n", 0, second_occurrence),
            character=second_occurrence
            - (content.rfind("\n", 0, second_occurrence) + 1),
        ),
        end=types.Position(
            line=content.count("\n", 0, second_occurrence + len(duplicate_name)),
            character=(
                second_occurrence
                + len(duplicate_name)
                - (content.rfind("\n", 0, second_occurrence + len(duplicate_name)) + 1)
            ),
        ),
    )
    assert duplicate_diagnostic.range == expected_range


@pytest.mark.integration
def test_spec_validator_matches_implementations_by_class_name():
    content = """
components:
  test_sensor_instance:
    class_name: TestSensor
    component_type: Sensor
    quantities:
      value:
        shape: (1,)
goals:
  reach_goal_instance:
    class_name: ReachGoal
connections:
  test_connection_instance:
    class_name: TestConnection
    components: [test_sensor_instance]
""".strip()
    parsed_spec = parse_spec_content(content)
    assert parsed_spec.spec is not None

    reader_components = {
        "TestSensor": ParsedComponent(
            component_type=SpecComponentTypes.Sensor,
            quantities={},
            goals=[],
            instance_name="TestSensor",
            class_name="TestSensor",
            position=_make_range(),
            document_uri="file:///components.py",
        )
    }
    reader_goals = {
        "ReachGoal": ParsedGoal(
            name="ReachGoal",
            position=_make_range(),
            document_uri="file:///goals.py",
            goal_func_params={},
        )
    }
    reader_connections = {
        "TestConnection": ParsedConnection(
            name="TestConnection",
            position=_make_range(),
            document_uri="file:///connections.py",
            quantities={},
            quantities_pos=_make_range(),
            connection_func_params={},
        )
    }

    validator = SpecReaderValidator(
        "file:///sample/spec.yaml",
        parsed_spec.spec,
        reader_components,
        reader_goals,
        reader_connections,
        content,
    )
    assert validator.diagnostics == []
