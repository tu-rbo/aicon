import logging
from collections.abc import Callable

from lsprotocol import types
from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.reader_types import (
    ParsedComponents,
    ParsedConnections,
    ParsedGoals,
)
from reader.spec.spec import AICON_SPEC
from util import (
    find_component_definition_in_spec,
    find_component_in_connection_components,
    find_connection_definition_in_spec,
    find_goal_definition_in_spec,
    find_goal_in_spec_in_component,
    find_nth_component_in_connection_components,
)

logger = logging.getLogger(__name__)


class SpecReaderValidator:
    diagnostics: list[Diagnostic]

    spec_uri: str
    spec: AICON_SPEC
    reader_components: ParsedComponents
    reader_goals: ParsedGoals
    reader_connections: ParsedConnections
    spec_content: str

    def __init__(
        self,
        spec_uri: str,
        spec: AICON_SPEC,
        reader_components: ParsedComponents,
        reader_goals: ParsedGoals,
        reader_connections: ParsedConnections,
        spec_content: str,
    ) -> None:
        self.spec = spec
        self.reader_components = reader_components
        self.reader_goals = reader_goals
        self.reader_connections = reader_connections
        self.spec_content = spec_content
        self.spec_uri = spec_uri
        self.diagnostics = []

        self._find_missing_components()
        self._find_missing_goals()
        self._find_missing_connections()
        self._find_undefined_component_in_connection()
        self._find_duplicate_component_in_connection()
        self._find_undefined_goals_in_component()

    def _spec_range(
        self, find: Callable[..., types.Range], *args: object
    ) -> types.Range:
        """Locate an entry in the spec text; fall back to the file start instead of crashing."""
        try:
            return find(*args, self.spec_content)
        except RuntimeError as e:
            logger.warning(str(e))
            zero = types.Position(line=0, character=0)
            return types.Range(start=zero, end=zero)

    def _find_missing_components(self):
        implemented_components = set(self.reader_components.keys())
        missing_component_classes: set[str] = set()
        for component_props in self.spec.components.values():
            if component_props.class_name in implemented_components:
                continue
            missing_component_classes.add(component_props.class_name)

        for missing_class_name in sorted(missing_component_classes):
            matching_instances = self.spec.get_component_instances_by_class_name(
                missing_class_name
            )
            if not matching_instances:
                continue
            instance_name = matching_instances[0]
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.COMPONENT_NOT_IMPLEMENTED,
                    range=self._spec_range(
                        find_component_definition_in_spec,
                        instance_name,
                        missing_class_name,
                    ),
                    component_name=missing_class_name,
                )
            )

    def _find_missing_goals(self):
        for goal_class_name in sorted(
            {goal.class_name for goal in self.spec.goals.values()}
        ):
            if goal_class_name in self.reader_goals:
                continue
            matching_instances = self.spec.get_goal_instances_by_class_name(
                goal_class_name
            )
            if not matching_instances:
                continue
            goal_instance_name = matching_instances[0]
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.GOAL_NOT_IMPLEMENTED,
                    range=self._spec_range(
                        find_goal_definition_in_spec,
                        goal_instance_name,
                        goal_class_name,
                    ),
                    goal_name=goal_class_name,
                )
            )

    def _find_missing_connections(self):
        implemented_connections = set(self.reader_connections.keys())
        for connection_class_name in sorted(
            {connection.class_name for connection in self.spec.connections.values()}
        ):
            if connection_class_name in implemented_connections:
                continue
            matching_instances = self.spec.get_connection_instances_by_class_name(
                connection_class_name
            )
            if not matching_instances:
                continue
            connection_instance_name = matching_instances[0]
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.CONNECTION_NOT_IMPLEMENTED,
                    range=self._spec_range(
                        find_connection_definition_in_spec,
                        connection_instance_name,
                        connection_class_name,
                    ),
                    connection_name=connection_class_name,
                )
            )

    def _find_undefined_component_in_connection(self):
        for connection_name, connection in self.spec.connections.items():
            for component_name in connection.components:
                if component_name not in self.spec.components:
                    self.diagnostics.append(
                        make_diagnostic(
                            ErrorCode.UNDEFINIED_COMPONENT_IN_CONNECTION,
                            range=self._spec_range(
                                find_component_in_connection_components,
                                connection_name,
                                component_name,
                            ),
                            component_name=component_name,
                        )
                    )

    def _find_duplicate_component_in_connection(self):
        for connection_name, connection in self.spec.connections.items():
            seen_components: dict[str, int] = {}
            for component_name in connection.components:
                occurrence = seen_components.get(component_name, 0) + 1
                seen_components[component_name] = occurrence
                if occurrence < 2:
                    continue
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.DUPLICATE_COMPONENT_IN_CONNECTION,
                        range=self._spec_range(
                            find_nth_component_in_connection_components,
                            connection_name,
                            component_name,
                            occurrence,
                        ),
                        connection_name=connection_name,
                        component_name=component_name,
                    )
                )

    def _find_undefined_goals_in_component(self):
        existing_goal_names = set(self.spec.goals.keys()) | {
            goal.name for goal in self.spec.goals.values()
        }
        for component_name, component in self.spec.components.items():
            for goal_name in component.goals:
                if goal_name in existing_goal_names:
                    continue
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.UNDEFINED_GOAL_IN_COMPONENT,
                        range=self._spec_range(
                            find_goal_in_spec_in_component, goal_name
                        ),
                        component_name=component_name,
                        goal_name=goal_name,
                    )
                )
