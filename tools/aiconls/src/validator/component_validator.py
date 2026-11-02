from typing import TypeAlias

from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.reader_types import (
    ParsedAction,
    ParsedEstimator,
    ParsedQuantities,
    ParsedSensor,
    QuantityReference,
)
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import SpecQuantities, shapes_match
from util import deep_diff_quantities

ParsedQuantifiedComponent: TypeAlias = ParsedSensor | ParsedAction | ParsedEstimator


class ComponentValidator:
    diagnostics: list[Diagnostic]

    def __init__(
        self, spec: AICON_SPEC, parsed_component: ParsedQuantifiedComponent
    ) -> None:
        self.diagnostics = []
        matching_component_instances = spec.get_component_instances_by_class_name(
            parsed_component.name
        )

        if not matching_component_instances:
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.COMPONENT_NOT_IN_SPEC,
                    range=parsed_component.position,
                    component_name=parsed_component.name,
                )
            )
            return

        primary_component_instance = matching_component_instances[0]
        primary_component_spec = spec.components[primary_component_instance]
        available_local_quantities = {
            quantity_name
            for component_instance in matching_component_instances
            for quantity_name in spec.components[component_instance].quantities
        }

        self._validate_local_quantity_references(
            parsed_component.name,
            parsed_component.quantity_references,
            available_local_quantities,
        )

        if parsed_component.quantities_pos is None:
            expected_quantities = sorted(primary_component_spec.quantities.keys())
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.QUANTITIES_NOT_IMPLEMENTED,
                    range=parsed_component.position,
                    component_name=parsed_component.name,
                    expected_quantities=(
                        ", ".join(expected_quantities)
                        if expected_quantities
                        else "(none)"
                    ),
                )
            )
        # Without a declaration there is nothing to compare, but connection
        # accesses can still be checked.
        elif not deep_diff_quantities(
            parsed_component.quantities,
            primary_component_spec.quantities,
        ):
            mismatch_details = self._describe_quantity_mismatch(
                parsed_component.quantities,
                primary_component_spec.quantities,
            )
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.SHAPE_SPEC_MISMATCH,
                    range=parsed_component.quantities_pos,
                    component_name=parsed_component.name,
                    mismatch_details=mismatch_details,
                )
            )

        allowed_connection_names = self._get_allowed_connection_names(
            spec,
            matching_component_instances,
        )

        for (
            connection_name,
            connection_pos,
        ) in parsed_component.connection_references.items():
            if connection_name in allowed_connection_names:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.CONNECTION_NOT_PART_OF_COMPONENT,
                    range=connection_pos,
                    component_name=parsed_component.name,
                    connection_name=connection_name,
                    allowed_connections=(
                        ", ".join(sorted(allowed_connection_names))
                        if allowed_connection_names
                        else "(none)"
                    ),
                )
            )

        for quantity_reference in parsed_component.connected_quantity_references:
            if quantity_reference.connection_name not in allowed_connection_names:
                continue

            available_connected_quantities = (
                self._get_connection_quantities_for_component(
                    spec,
                    matching_component_instances,
                    quantity_reference.connection_name,
                )
            )
            if quantity_reference.quantity_name in available_connected_quantities:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.CONNECTED_QUANTITY_NOT_AVAILABLE,
                    range=quantity_reference.position,
                    component_name=parsed_component.name,
                    connection_name=quantity_reference.connection_name,
                    quantity_name=quantity_reference.quantity_name,
                    available_quantities=(
                        ", ".join(sorted(available_connected_quantities))
                        if available_connected_quantities
                        else "(none)"
                    ),
                )
            )

    def _get_allowed_connection_names(
        self, spec: AICON_SPEC, matching_component_instances: list[str]
    ) -> set[str]:
        component_instances = set(matching_component_instances)
        return {
            connection_name
            for connection_name, connection_props in spec.connections.items()
            if component_instances.intersection(connection_props.components)
        }

    def _describe_quantity_mismatch(
        self,
        actual_quantities: ParsedQuantities,
        expected_quantities: SpecQuantities,
    ) -> str:
        details: list[str] = []
        actual_names = set(actual_quantities.keys())
        expected_names = set(expected_quantities.keys())

        missing_names = sorted(expected_names - actual_names)
        if missing_names:
            details.append(f"missing quantities: {', '.join(missing_names)}")

        unexpected_names = sorted(actual_names - expected_names)
        if unexpected_names:
            details.append(f"unexpected quantities: {', '.join(unexpected_names)}")

        for quantity_name in sorted(actual_names & expected_names):
            actual_shape = actual_quantities[quantity_name].shape
            expected_shape = expected_quantities[quantity_name].shape
            if not shapes_match(actual_shape, expected_shape):
                details.append(
                    "quantity "
                    f"'{quantity_name}' has shape {actual_shape}, expected "
                    f"{expected_shape}"
                )

        return "; ".join(details)

    def _get_connection_quantities_for_component(
        self,
        spec: AICON_SPEC,
        matching_component_instances: list[str],
        connection_name: str,
    ) -> set[str]:
        connection_props = spec.connections[connection_name]
        own_instances = set(matching_component_instances)
        quantities: set[str] = set()
        for linked_component in connection_props.components:
            if linked_component in own_instances:
                continue
            linked_component_props = spec.components.get(linked_component)
            if linked_component_props is None:
                continue
            quantities.update(linked_component_props.quantities.keys())
        return quantities

    def _validate_local_quantity_references(
        self,
        component_name: str,
        quantity_references: list[QuantityReference],
        available_local_quantities: set[str],
    ) -> None:
        for quantity_reference in quantity_references:
            if quantity_reference.quantity_name in available_local_quantities:
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.COMPONENT_QUANTITY_NOT_AVAILABLE,
                    range=quantity_reference.position,
                    component_name=component_name,
                    quantity_name=quantity_reference.quantity_name,
                    available_quantities=(
                        ", ".join(sorted(available_local_quantities))
                        if available_local_quantities
                        else "(none)"
                    ),
                )
            )
