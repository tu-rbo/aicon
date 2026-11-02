from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.reader_types import ParsedGoal
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import SpecQuantities


class GoalValidator:
    diagnostics: list[Diagnostic]

    def __init__(self, spec: AICON_SPEC, parsed_goal: ParsedGoal) -> None:
        self.diagnostics = []
        if not spec.get_goal_instances_by_class_name(parsed_goal.name):
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.GOAL_NOT_IN_SPEC,
                    range=parsed_goal.position,
                    goal_name=parsed_goal.name,
                )
            )
            return

        self._validate_goal_func_params(spec, parsed_goal)

    def _resolve_matching_goal_identifiers(
        self, spec: AICON_SPEC, parsed_goal_name: str
    ) -> set[str]:
        matching_goal_identifiers: set[str] = {parsed_goal_name}
        for goal_instance in spec.get_goal_instances_by_class_name(parsed_goal_name):
            matching_goal_identifiers.add(goal_instance)
            matching_goal_identifiers.add(spec.goals[goal_instance].class_name)
        return matching_goal_identifiers

    def _resolve_goal_owner_components(
        self, spec: AICON_SPEC, goal_identifiers: set[str]
    ) -> set[str]:
        return {
            component_name
            for component_name, component_props in spec.components.items()
            if any(goal_name in goal_identifiers for goal_name in component_props.goals)
        }

    def _build_allowed_goal_quantities(
        self, spec: AICON_SPEC, goal_identifiers: set[str]
    ) -> SpecQuantities:
        owner_components = self._resolve_goal_owner_components(spec, goal_identifiers)

        quantities: SpecQuantities = {}
        for component_name in owner_components:
            component_props = spec.components[component_name]
            for quantity_name, quantity_props in component_props.quantities.items():
                quantities[quantity_name] = quantity_props
        return quantities

    def _validate_goal_func_params(
        self, spec: AICON_SPEC, parsed_goal: ParsedGoal
    ) -> None:
        goal_identifiers = self._resolve_matching_goal_identifiers(
            spec, parsed_goal.name
        )
        # a goal that no component holds (e.g. one of several goals an experiment can be run with) has no quantities
        # to check its arguments against
        if not self._resolve_goal_owner_components(spec, goal_identifiers):
            return
        allowed_quantities = self._build_allowed_goal_quantities(spec, goal_identifiers)
        allowed_parameters = set(allowed_quantities.keys())

        for parameter_name, parameter_pos in parsed_goal.goal_func_params.items():
            if parameter_name in allowed_parameters:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.GOAL_FUNC_PARAMETER_NOT_AVAILABLE,
                    range=parameter_pos,
                    parameter_name=parameter_name,
                    goal_name=parsed_goal.name,
                    allowed_parameters=(
                        ", ".join(sorted(allowed_parameters))
                        if allowed_parameters
                        else "(none)"
                    ),
                )
            )
