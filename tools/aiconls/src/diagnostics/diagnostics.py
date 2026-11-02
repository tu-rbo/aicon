from dataclasses import dataclass
from enum import Enum

from lsprotocol.types import DiagnosticSeverity


@dataclass(frozen=True)
class DiagnosticDefinition:
    code: str
    severity: DiagnosticSeverity
    template: str


class ErrorCode(Enum):
    # ------Spec------
    # Parser
    MALFORMED_SPEC = DiagnosticDefinition(
        code="E000",
        severity=DiagnosticSeverity.Error,
        template="Malformed spec.",
    )
    MALFORMED_COMPONENT_TYPE = DiagnosticDefinition(
        code="E001",
        severity=DiagnosticSeverity.Error,
        template="Malformed component type.",
    )
    MALFORMED_DATA_TYPE = DiagnosticDefinition(
        code="E002",
        severity=DiagnosticSeverity.Error,
        template="Malformed data type.",
    )
    MALFORMED_QUANTITY_DEFINITION = DiagnosticDefinition(
        code="E003",
        severity=DiagnosticSeverity.Error,
        template="Malformed quantity definition.",
    )
    DUPLICATE_COMPONENT = DiagnosticDefinition(
        code="E004",
        severity=DiagnosticSeverity.Error,
        template="Duplicated component in spec.",
    )
    DUPLICATE_GOAL = DiagnosticDefinition(
        code="E005",
        severity=DiagnosticSeverity.Error,
        template="Duplicated goal in spec.",
    )
    DUPLICATE_CONNECTION = DiagnosticDefinition(
        code="E006",
        severity=DiagnosticSeverity.Error,
        template="Duplicated connection in spec.",
    )
    UNEXPECTED_COMPONENT_PROPERTY = DiagnosticDefinition(
        code="E007",
        severity=DiagnosticSeverity.Error,
        template="Unexpected component property in spec.",
    )
    UNEXPECTED_GOAL_PROPERTY = DiagnosticDefinition(
        code="E008",
        severity=DiagnosticSeverity.Error,
        template="Unexpected goal property in spec.",
    )
    UNEXPECTED_CONNECTION_PROPERTY = DiagnosticDefinition(
        code="E009",
        severity=DiagnosticSeverity.Error,
        template="Unexpected connection property in spec.",
    )

    # Validator
    COMPONENT_NOT_IMPLEMENTED = DiagnosticDefinition(
        code="W011",
        severity=DiagnosticSeverity.Warning,
        template=(
            "Component '{component_name}' is declared in spec but not implemented in "
            "the project."
        ),
    )
    GOAL_NOT_IMPLEMENTED = DiagnosticDefinition(
        code="W012",
        severity=DiagnosticSeverity.Warning,
        template="Goal '{goal_name}' is declared in spec but not implemented in the project.",
    )
    CONNECTION_NOT_IMPLEMENTED = DiagnosticDefinition(
        code="W013",
        severity=DiagnosticSeverity.Warning,
        template=(
            "Connection '{connection_name}' is declared in spec but not implemented in "
            "the project."
        ),
    )
    UNDEFINIED_COMPONENT_IN_CONNECTION = DiagnosticDefinition(
        code="E012",
        severity=DiagnosticSeverity.Error,
        template="Component '{component_name}' not found in spec.",
    )
    UNDEFINED_GOAL_IN_COMPONENT = DiagnosticDefinition(
        code="E013",
        severity=DiagnosticSeverity.Error,
        template=(
            "Component '{component_name}' references goal '{goal_name}', but the goal "
            "is not defined in spec."
        ),
    )
    DUPLICATE_COMPONENT_IN_CONNECTION = DiagnosticDefinition(
        code="E014",
        severity=DiagnosticSeverity.Error,
        template=(
            "Connection '{connection_name}' contains duplicate component "
            "'{component_name}' in its components list."
        ),
    )

    # -----Component-------
    QUANTITIES_NOT_IMPLEMENTED = DiagnosticDefinition(
        code="W103",
        severity=DiagnosticSeverity.Warning,
        template=(
            "Component '{component_name}' is missing local quantity definitions. "
            "Declare them in the class attribute cls_quantity_names_and_shapes. "
            "Expected quantities from spec: {expected_quantities}."
        ),
    )
    SHAPE_SPEC_MISMATCH = DiagnosticDefinition(
        code="E102",
        severity=DiagnosticSeverity.Error,
        template=(
            "Component '{component_name}' quantities do not match spec: "
            "{mismatch_details}."
        ),
    )
    COMPONENT_NOT_IN_SPEC = DiagnosticDefinition(
        code="W101",
        severity=DiagnosticSeverity.Warning,
        template="Component '{component_name}' not found in spec.",
    )
    GOAL_NOT_IN_SPEC = DiagnosticDefinition(
        code="W102",
        severity=DiagnosticSeverity.Warning,
        template="Goal '{goal_name}' not found in spec.",
    )
    GOAL_FUNC_PARAMETER_NOT_AVAILABLE = DiagnosticDefinition(
        code="E103",
        severity=DiagnosticSeverity.Error,
        template=(
            "goal_func parameter '{parameter_name}' in goal '{goal_name}' is not "
            "available from quantities of connected components owning this goal. "
            "Allowed parameters: "
            "{allowed_parameters}."
        ),
    )
    CONNECTION_NOT_PART_OF_COMPONENT = DiagnosticDefinition(
        code="E104",
        severity=DiagnosticSeverity.Error,
        template=(
            "Component '{component_name}' references connection '{connection_name}', "
            "but this component is not listed in that connection in spec. Allowed "
            "connections for this component: {allowed_connections}."
        ),
    )
    CONNECTED_QUANTITY_NOT_AVAILABLE = DiagnosticDefinition(
        code="E105",
        severity=DiagnosticSeverity.Error,
        template=(
            "Component '{component_name}' accesses connected quantity "
            "'{quantity_name}' through connection '{connection_name}', but this "
            "quantity is not provided by connected components. Available connected "
            "quantities: {available_quantities}."
        ),
    )
    ESTIMATOR_F_FUNC_PARAMETER_NOT_AVAILABLE = DiagnosticDefinition(
        code="E106",
        severity=DiagnosticSeverity.Error,
        template=(
            "f_func parameter '{parameter_name}' in estimator '{component_name}' is "
            "not available from local quantities or directly connected components. "
            "Allowed parameters: {allowed_parameters}."
        ),
    )
    COMPONENT_QUANTITY_NOT_AVAILABLE = DiagnosticDefinition(
        code="E107",
        severity=DiagnosticSeverity.Error,
        template=(
            "Component '{component_name}' accesses local quantity '{quantity_name}' "
            "through self.quantities, but this quantity is not defined in spec. "
            "Available local quantities: {available_quantities}."
        ),
    )
    ESTIMATOR_F_FUNC_NOT_DIFFERENTIABLE = DiagnosticDefinition(
        code="E108",
        severity=DiagnosticSeverity.Error,
        template=(
            "f_func in estimator '{component_name}' failed the derivative check "
            "with torch.func.jacfwd: {reason}."
        ),
    )
    ESTIMATOR_F_FUNC_OUTPUT_QUANTITY_NOT_AVAILABLE = DiagnosticDefinition(
        code="E109",
        severity=DiagnosticSeverity.Error,
        template=(
            "f_func {signature_kind} output '{quantity_name}' in estimator "
            "'{component_name}' is not defined as local quantity in spec. "
            "Available local quantities: {available_quantities}."
        ),
    )

    COMPONENT_NOT_INSTANTIABLE = DiagnosticDefinition(
        code="W104",
        severity=DiagnosticSeverity.Warning,
        template=(
            "{entity_kind} '{instance_name}' ({class_name}) could not be instantiated "
            "by the dynamic check and was skipped: {reason}."
        ),
    )

    # -----Connections-------
    CONNECTION_NOT_IN_SPEC = DiagnosticDefinition(
        code="W201",
        severity=DiagnosticSeverity.Warning,
        template="Connection '{connection_name}' not found in spec.",
    )
    QUANTITY_NOT_AVAILABLE = DiagnosticDefinition(
        code="E201",
        severity=DiagnosticSeverity.Error,
        template="Connection '{connection_name}' has invalid quantities: {mismatch_details}.",
    )
    CONNECTION_FUNC_PARAMETER_NOT_AVAILABLE = DiagnosticDefinition(
        code="E202",
        severity=DiagnosticSeverity.Error,
        template=(
            "connection_func parameter '{parameter_name}' in connection "
            "'{connection_name}' is not defined in super constructor quantities. "
            "Allowed parameters: {allowed_parameters}."
        ),
    )
    UNUSED_CONNECTION_QUANTITY = DiagnosticDefinition(
        code="W202",
        severity=DiagnosticSeverity.Warning,
        template=(
            "Quantity '{quantity_name}' from connection '{connection_name}' super "
            "constructor is unused in connection_func parameters."
        ),
    )
    BUILD_CONNECTIONS_FUNCTION_MISSING = DiagnosticDefinition(
        code="E206",
        severity=DiagnosticSeverity.Error,
        template=(
            "connections.py is missing a build_connections() function for the "
            "connections declared in spec."
        ),
    )
    BUILD_CONNECTIONS_ENTRY_MISSING = DiagnosticDefinition(
        code="E203",
        severity=DiagnosticSeverity.Error,
        template=(
            "build_connections() is missing builder entry for connection "
            "'{connection_name}'."
        ),
    )
    BUILD_CONNECTIONS_CLASS_MISMATCH = DiagnosticDefinition(
        code="E204",
        severity=DiagnosticSeverity.Error,
        template=(
            "build_connections() entry for connection '{connection_name}' constructs "
            "'{actual_class_name}', expected '{expected_class_name}'."
        ),
    )
    BUILD_CONNECTIONS_NAME_ARGUMENT_MISMATCH = DiagnosticDefinition(
        code="E205",
        severity=DiagnosticSeverity.Error,
        template=(
            "build_connections() entry for connection '{connection_name}' passes "
            "constructor name '{actual_name_argument}', expected "
            "'{expected_name_argument}'."
        ),
    )
    GET_BUILDING_FUNCTIONS_CONNECTION_BUILDERS_INVALID = DiagnosticDefinition(
        code="E207",
        severity=DiagnosticSeverity.Error,
        template=(
            "get_building_functions() must assign connection_builders from "
            "build_connections()."
        ),
    )
    CONNECTION_NOT_INSTANTIABLE = DiagnosticDefinition(
        code="W203",
        severity=DiagnosticSeverity.Warning,
        template=(
            "Connection '{instance_name}' ({class_name}) could not be instantiated by "
            "the dynamic check and was skipped: {reason}."
        ),
    )
    CONNECTION_FUNC_NOT_DIFFERENTIABLE = DiagnosticDefinition(
        code="E208",
        severity=DiagnosticSeverity.Error,
        template=(
            "connection_func in connection '{connection_name}' failed the derivative "
            "check with torch.func.jacfwd: {reason}."
        ),
    )


def get_diagnostic_definition(code: ErrorCode) -> DiagnosticDefinition:
    return code.value


def get_diagnostic_definition_by_name(name: str) -> DiagnosticDefinition:
    return get_diagnostic_definition(ErrorCode[name])
