from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.reader_types import ParsedConnection, ParsedQuantities
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    SpecComponentQuantityGeneric,
    SpecQuantities,
    SpecQuantityTypes,
    shapes_match,
)
from util import is_subset_quantities


class ConnectionValidator:
    diagnostics: list[Diagnostic]

    def __init__(self, spec: AICON_SPEC, parsed_connection: ParsedConnection) -> None:
        self.diagnostics = []
        self.spec = spec

        matching_connection_instances = spec.get_connection_instances_by_class_name(
            parsed_connection.name
        )
        if not matching_connection_instances:
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.CONNECTION_NOT_IN_SPEC,
                    range=parsed_connection.position,
                    connection_name=parsed_connection.name,
                )
            )
            return

        expected_quantities = self._build_expected_quantities(
            spec, matching_connection_instances
        )
        # batched estimators get "<name>_batched" inputs: <name> repeated along batch dimensions, so of any shape
        for quantity_name, quantity_props in parsed_connection.quantities.items():
            if (
                quantity_name.endswith("_batched")
                and quantity_name[: -len("_batched")] in expected_quantities
            ):
                expected_quantities[quantity_name] = SpecComponentQuantityGeneric(
                    shape=quantity_props.shape
                )

        if parsed_connection.quantities_pos is None:
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.QUANTITY_NOT_AVAILABLE,
                    range=parsed_connection.position,
                    connection_name=parsed_connection.name,
                    mismatch_details=(
                        f"missing super constructor quantity mapping (available quantities: {self._format_quantity_names(expected_quantities)})"
                    ),
                )
            )
            return

        if not is_subset_quantities(parsed_connection.quantities, expected_quantities):
            mismatch_details = self._describe_quantity_mismatch(
                parsed_connection.quantities,
                expected_quantities,
            )
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.QUANTITY_NOT_AVAILABLE,
                    range=parsed_connection.quantities_pos,
                    connection_name=parsed_connection.name,
                    mismatch_details=mismatch_details,
                )
            )

        self._validate_connection_func_params(parsed_connection)
        self._validate_unused_quantities(parsed_connection)

    def _build_expected_quantities(
        self, spec: AICON_SPEC, connection_instances: list[str]
    ) -> SpecQuantities:
        # The quantities of the components on the connection and, as the connection function is evaluated with all
        # inputs its components have, those of the components they share another connection with.
        quantities: SpecQuantities = {}
        neighbours: list[str] = []
        for connection_name in connection_instances:
            connection = spec.connections[connection_name]
            for component_name in connection.components:
                if component_name not in spec.components:
                    continue
                for quantity_name, quantity_props in spec.components[
                    component_name
                ].quantities.items():
                    quantities[quantity_name] = quantity_props
                for other_connection in spec.connections.values():
                    if component_name in other_connection.components:
                        neighbours.extend(other_connection.components)
        for component_name in neighbours:
            if component_name not in spec.components:
                continue
            for quantity_name, quantity_props in spec.components[
                component_name
            ].quantities.items():
                quantities.setdefault(quantity_name, quantity_props)
        return quantities

    def _validate_connection_func_params(
        self, parsed_connection: ParsedConnection
    ) -> None:
        allowed_parameters = set(parsed_connection.quantities.keys())
        for (
            parameter_name,
            parameter_pos,
        ) in parsed_connection.connection_func_params.items():
            if parameter_name in allowed_parameters:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.CONNECTION_FUNC_PARAMETER_NOT_AVAILABLE,
                    range=parameter_pos,
                    parameter_name=parameter_name,
                    connection_name=parsed_connection.name,
                    allowed_parameters=(
                        ", ".join(sorted(allowed_parameters))
                        if allowed_parameters
                        else "(none)"
                    ),
                )
            )

    def _validate_unused_quantities(self, parsed_connection: ParsedConnection) -> None:
        used_parameters = set(parsed_connection.connection_func_params.keys())

        references_names: list[str] = []
        for c in self.spec.components.values():
            for q_name, q in c.quantities.items():
                if q.q_type == SpecQuantityTypes.Reference:
                    references_names.append(q_name)

        for quantity_name, quantity_pos in parsed_connection.quantity_positions.items():
            if quantity_name in used_parameters or quantity_name in references_names:
                continue
            # a switching condition, or used as "<name>_batched" (repeated along batch dimensions by AICON)
            if quantity_name in parsed_connection.switching_quantities:
                continue
            if f"{quantity_name}_batched" in used_parameters:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.UNUSED_CONNECTION_QUANTITY,
                    range=quantity_pos,
                    quantity_name=quantity_name,
                    connection_name=parsed_connection.name,
                )
            )

    def _describe_quantity_mismatch(
        self, actual_quantities: ParsedQuantities, expected_quantities: SpecQuantities
    ) -> str:
        details: list[str] = []
        for quantity_name, quantity_props in sorted(actual_quantities.items()):
            expected_props = expected_quantities.get(quantity_name)
            if expected_props is None:
                details.append(
                    f"unknown quantity '{quantity_name}' (available quantities: {self._format_quantity_names(expected_quantities)})"
                )
                continue
            if not shapes_match(quantity_props.shape, expected_props.shape):
                details.append(
                    f"quantity '{quantity_name}' has shape {quantity_props.shape}, expected {expected_props.shape}"
                )

        return "; ".join(details)

    def _format_quantity_names(self, quantities: SpecQuantities) -> str:
        if not quantities:
            return "(none)"
        return ", ".join(sorted(quantities.keys()))
