from pathlib import Path

from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentQuantityGeneric,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
    SpecGoalProps,
)


def test_generate_skeleton_writes_action_file_for_action_components(tmp_path: Path):
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestAction": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={"cmd": SpecComponentQuantityGeneric(shape=(2,))},
                    goals=[],
                    instance_name="TestAction",
                    class_name="TestAction",
                )
            },
            goals={},
            connections={},
        )
    )

    generated_files = spec.generate_skeleton(str(tmp_path))

    assert str(tmp_path / "actions.py") in generated_files
    action_content = (tmp_path / "actions.py").read_text()
    assert "class TestAction(ActionComponent):" in action_content
    assert "def determine_new_actions(self):" in action_content
    assert "def initialize_quantities(self):" in action_content


def test_generate_skeleton_uses_class_names_for_instance_based_entries(tmp_path: Path):
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "sensor_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"meas": SpecComponentQuantityGeneric(shape=(1,))},
                    goals=[],
                    instance_name="sensor_instance",
                    class_name="SensorClass",
                ),
                "action_instance": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={"cmd": SpecComponentQuantityGeneric(shape=(2,))},
                    goals=["goal_instance"],
                    instance_name="action_instance",
                    class_name="ActionClass",
                ),
            },
            goals={
                "goal_instance": SpecGoalProps(
                    name="GoalClass",
                    instance_name="goal_instance",
                )
            },
            connections={
                "connection_instance": SpecConnectionProps(
                    class_name="ConnectionClass",
                    components=["sensor_instance", "action_instance"],
                    instance_name="connection_instance",
                )
            },
        )
    )

    generated_files = spec.generate_skeleton(str(tmp_path))

    assert str(tmp_path / "sensors.py") in generated_files
    assert str(tmp_path / "actions.py") in generated_files
    assert str(tmp_path / "goals.py") in generated_files
    assert str(tmp_path / "connections.py") in generated_files

    assert (
        "class SensorClass(SensorComponent):" in (tmp_path / "sensors.py").read_text()
    )
    assert (
        "class ActionClass(ActionComponent):" in (tmp_path / "actions.py").read_text()
    )
    assert "class GoalClass(Goal):" in (tmp_path / "goals.py").read_text()
    connection_content = (tmp_path / "connections.py").read_text()
    assert "class ConnectionClass(ActiveInterconnection):" in connection_content
    assert (
        '"connection_instance": lambda device, dtype, mockbuild: ConnectionClass('
        in (connection_content)
    )
