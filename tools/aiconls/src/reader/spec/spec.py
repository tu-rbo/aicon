import logging
import sys
from os import path
from pathlib import Path

from generator.action import gen_actions
from generator.component import generate_general_component
from generator.connection import gen_connection, gen_connections
from generator.estimator import gen_estimators
from generator.experiment_specifications import (
    gen_experiment_specifications,
)
from generator.generator_util import get_file_name_from_component_type
from generator.goal import gen_goal, gen_goals
from generator.sensor import gen_sensors
from reader.spec.spec_types import (
    SpecComponents,
    SpecComponentTypes,
    SpecConnections,
    SpecFormat,
    SpecGoals,
    SpecQuantityTypes,
)

logger = logging.getLogger(__name__)


class AICON_SPEC:
    # theses are the parsd spec components and connections
    components: SpecComponents
    goals: SpecGoals
    connections: SpecConnections

    def __init__(self, spec_yaml: SpecFormat) -> None:
        self.components = {}
        self.goals = {}
        self.connections = {}
        self.parse_yaml(spec_yaml)

    def parse_tuple(self, shape_tuple: str):
        if len(shape_tuple) < 4:
            logger.error("Invalid Tuple")
            sys.exit(1)
        tuple_split = shape_tuple[1:-1].split(",")
        tuple_split = [x for x in tuple_split if len(x) > 0]
        return tuple(int(x) for x in tuple_split)

    # convert the found shapes of quantities to real python tuples
    def normalize_quantities(self, spec_yaml: SpecFormat):
        # yaml parses string
        for comp_props in spec_yaml.components.values():
            for q_props in comp_props.quantities.values():
                match q_props.q_type:
                    case SpecQuantityTypes.Generic:
                        q_props.shape = self.parse_tuple(str(q_props.shape))
                    case SpecQuantityTypes.Gaussian:
                        q_props.shape = (int(q_props.dim),)
                    case SpecQuantityTypes.Particle:
                        q_props.shape = (int(q_props.particle_count), int(q_props.dim))
                    case SpecQuantityTypes.Categorical:
                        q_props.shape = (int(q_props.dim),)
                    case SpecQuantityTypes.Reference:
                        pass

    def parse_yaml(self, spec_yaml: SpecFormat):
        self.normalize_quantities(spec_yaml)
        self.components = spec_yaml.components
        self.goals = spec_yaml.goals
        self.connections = spec_yaml.connections
        self._normalize_identifiers()

    def _normalize_identifiers(self) -> None:
        for instance_name, component_props in self.components.items():
            if not component_props.instance_name:
                component_props.instance_name = instance_name
            if not component_props.class_name:
                component_props.class_name = instance_name

        for instance_name, goal_props in self.goals.items():
            if not goal_props.instance_name:
                goal_props.instance_name = instance_name
            if not goal_props.name:
                goal_props.name = instance_name

        for instance_name, connection_props in self.connections.items():
            if not connection_props.instance_name:
                connection_props.instance_name = instance_name
            if not connection_props.class_name:
                connection_props.class_name = instance_name

    def get_component_instances_by_class_name(self, class_name: str) -> list[str]:
        return [
            instance_name
            for instance_name, component_props in self.components.items()
            if component_props.class_name == class_name
        ]

    def get_goal_instances_by_class_name(self, class_name: str) -> list[str]:
        return [
            instance_name
            for instance_name, goal_props in self.goals.items()
            if goal_props.class_name == class_name
        ]

    def get_connection_instances_by_class_name(self, class_name: str) -> list[str]:
        return [
            instance_name
            for instance_name, connection_props in self.connections.items()
            if connection_props.class_name == class_name
        ]

    def get_component_class_name_by_instance_name(
        self, instance_name: str
    ) -> str | None:
        component_props = self.components.get(instance_name)
        if component_props is None:
            return None
        return component_props.class_name

    def get_goal_class_name_by_instance_name(self, instance_name: str) -> str | None:
        goal_props = self.goals.get(instance_name)
        if goal_props is None:
            return None
        return goal_props.class_name

    def get_connection_class_name_by_instance_name(
        self, instance_name: str
    ) -> str | None:
        connection_props = self.connections.get(instance_name)
        if connection_props is None:
            return None
        return connection_props.class_name

    def get_skeleton_file_paths(self, project_directory: str) -> list[Path]:
        target_directory = Path(project_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        generated_files: list[Path] = []
        if any(
            comp.component_type == SpecComponentTypes.Estimator
            for comp in self.components.values()
        ):
            generated_files.append(target_directory / "estimators.py")
        if any(
            comp.component_type == SpecComponentTypes.Sensor
            for comp in self.components.values()
        ):
            generated_files.append(target_directory / "sensors.py")
        if any(
            comp.component_type == SpecComponentTypes.Action
            for comp in self.components.values()
        ):
            generated_files.append(target_directory / "actions.py")
        if self.goals:
            generated_files.append(target_directory / "goals.py")
        if self.connections:
            generated_files.append(target_directory / "connections.py")

        return generated_files

    def generate_skeleton(self, project_directory: str) -> list[str]:
        target_directory = Path(project_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        generated_files = [
            str(file_path)
            for file_path in self.get_skeleton_file_paths(str(target_directory))
        ]

        gen_estimators(target_directory, self.components)
        gen_sensors(target_directory, self.components)
        gen_actions(target_directory, self.components)
        gen_goals(target_directory, self.goals)
        gen_connections(target_directory, self.connections, self.components)

        return generated_files

    def generate_component(self, project_directory: str, component_name: str):
        # find component
        if component_name not in self.components:
            raise RuntimeError(
                f"Component cant be generate since key: {component_name} not in {self.components.keys()}"
            )

        component = self.components[component_name]

        target_file = Path(
            path.join(
                project_directory,
                get_file_name_from_component_type(component.component_type),
            )
        )
        component_content = generate_general_component(component.class_name, component)
        assert component_content

        return (target_file, component_content)

    def generate_goal(self, project_directory: str, goal_name: str):
        if goal_name not in self.goals:
            raise RuntimeError(
                f"Goal cant be generate since key: {goal_name} not in {self.goals.keys()}"
            )

        goal_props = self.goals[goal_name]
        target_file = Path(path.join(project_directory, "goals.py"))
        goal_content = gen_goal(goal_props.class_name or goal_name)
        return (target_file, goal_content)

    def generate_connection(self, project_directory: str, connection_name: str):
        if connection_name not in self.connections:
            raise RuntimeError(
                "Connection cant be generate since key:"
                f" {connection_name} not in {self.connections.keys()}"
            )

        connection_props = self.connections[connection_name]
        target_file = Path(path.join(project_directory, "connections.py"))
        connection_content = gen_connection(
            connection_props.class_name, connection_props, self.components
        )
        return (target_file, connection_content)

    def generate_experiment_specification(self, project_directory: str):
        target_directory = Path(project_directory)
        target_directory.mkdir(parents=True, exist_ok=True)
        gen_experiment_specifications(
            target_directory,
            self.components,
            self.goals,
            self.connections,
        )
