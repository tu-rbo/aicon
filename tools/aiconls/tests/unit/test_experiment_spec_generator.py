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


def test_generate_experiment_specification_file(tmp_path: Path):
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "TestSensor": SpecComponentProps(
                    component_type=SpecComponentTypes.Sensor,
                    quantities={"meas": SpecComponentQuantityGeneric(shape=(3,))},
                    goals=[],
                    instance_name="TestSensor",
                    class_name="TestSensor",
                ),
                "TestEstimator": SpecComponentProps(
                    component_type=SpecComponentTypes.Estimator,
                    quantities={"state": SpecComponentQuantityGeneric(shape=(3,))},
                    goals=["stateGoal"],
                    instance_name="TestEstimator",
                    class_name="TestEstimator",
                ),
                "TestAction": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={"cmd": SpecComponentQuantityGeneric(shape=(2,))},
                    goals=[],
                    instance_name="TestAction",
                    class_name="TestAction",
                ),
            },
            goals={"stateGoal": SpecGoalProps(name="ReachStateGoal")},
            connections={
                "stateConn": SpecConnectionProps(
                    components=["TestSensor", "TestEstimator"],
                ),
                "actionConn": SpecConnectionProps(
                    components=["TestAction", "TestEstimator"],
                ),
            },
        )
    )

    spec.generate_experiment_specification(str(tmp_path))
    content = (tmp_path / "experiment_specifications.py").read_text()

    assert "def get_building_functions(sim_env_pointer=None)" in content
    assert "from connections import build_connections" in content
    assert "connection_builders = build_connections()" in content
    assert '"TestEstimator": lambda mockbuild=False: TestEstimator(' in content
    assert (
        '"stateGoal": lambda device, dtype, mockbuild=False: ReachStateGoal(' in content
    )
    assert "sim_env_pointer=sim_env_pointer" in content


def test_generate_experiment_specification_uses_spec_dtype(tmp_path: Path):
    spec = AICON_SPEC(
        SpecFormat(
            components={
                "FloatAction": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={},
                    goals=[],
                    instance_name="FloatAction",
                    class_name="FloatAction",
                    dtype="float32",
                ),
                "DefaultAction": SpecComponentProps(
                    component_type=SpecComponentTypes.Action,
                    quantities={},
                    goals=[],
                    instance_name="DefaultAction",
                    class_name="DefaultAction",
                ),
            },
            goals={},
            connections={},
        )
    )

    spec.generate_experiment_specification(str(tmp_path))
    lines = (tmp_path / "experiment_specifications.py").read_text().splitlines()

    float_line = next(line for line in lines if '"FloatAction":' in line)
    default_line = next(line for line in lines if '"DefaultAction":' in line)
    assert "dtype=torch.float32" in float_line
    assert "dtype=DTYPE" in default_line
