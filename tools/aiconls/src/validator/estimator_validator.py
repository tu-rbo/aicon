from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.reader_types import ParsedEstimator
from reader.spec.spec import AICON_SPEC
from validator.component_validator import ComponentValidator


class EstimatorValidator(ComponentValidator):
    diagnostics: list[Diagnostic]

    def __init__(self, spec: AICON_SPEC, parsed_estimator: ParsedEstimator) -> None:
        super().__init__(spec, parsed_estimator)
        matching_component_instances = spec.get_component_instances_by_class_name(
            parsed_estimator.name
        )
        if not matching_component_instances:
            return

        own_quantities: set[str] = set()
        for component_instance in matching_component_instances:
            component_props = spec.components.get(component_instance)
            if component_props is None:
                continue
            own_quantities.update(component_props.quantities.keys())

        connected_quantities = self._collect_connected_quantities(
            spec, matching_component_instances
        )
        self._validate_f_func_params(
            parsed_estimator, own_quantities, connected_quantities
        )
        self._validate_f_func_output_signatures(parsed_estimator, own_quantities)

    def _collect_connected_quantities(
        self, spec: AICON_SPEC, matching_component_instances: list[str]
    ) -> set[str]:
        allowed_connection_names = self._get_allowed_connection_names(
            spec, matching_component_instances
        )
        connected_quantities: set[str] = set()
        for connection_name in allowed_connection_names:
            connected_quantities.update(
                self._get_connection_quantities_for_component(
                    spec, matching_component_instances, connection_name
                )
            )
        return connected_quantities

    def _validate_f_func_params(
        self,
        parsed_estimator: ParsedEstimator,
        own_quantities: set[str],
        connected_quantities: set[str],
    ) -> None:
        allowed_parameters = own_quantities.union(connected_quantities).union({"dt"})
        for parameter_name, parameter_pos in parsed_estimator.f_func_params.items():
            if parameter_name in allowed_parameters:
                continue
            # batched estimators get "<name>_batched" inputs (<name> repeated along batch dimensions)
            if (
                parameter_name.endswith("_batched")
                and parameter_name[: -len("_batched")] in allowed_parameters
            ):
                continue
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.ESTIMATOR_F_FUNC_PARAMETER_NOT_AVAILABLE,
                    range=parameter_pos,
                    parameter_name=parameter_name,
                    component_name=parsed_estimator.name,
                    allowed_parameters=(
                        ", ".join(sorted(allowed_parameters))
                        if allowed_parameters
                        else "(none)"
                    ),
                )
            )

    def _validate_f_func_output_signatures(
        self,
        parsed_estimator: ParsedEstimator,
        own_quantities: set[str],
    ) -> None:
        available_quantities = (
            ", ".join(sorted(own_quantities)) if own_quantities else "(none)"
        )

        signatures_to_validate = (
            (
                "differentiable",
                parsed_estimator.f_output_signature_diff_names,
            ),
            (
                "non-differentiable/state",
                parsed_estimator.f_output_signature_non_diff_names,
            ),
        )
        for signature_kind, signature_names in signatures_to_validate:
            for quantity_name, quantity_pos in signature_names.items():
                if quantity_name in own_quantities:
                    continue
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.ESTIMATOR_F_FUNC_OUTPUT_QUANTITY_NOT_AVAILABLE,
                        range=quantity_pos,
                        signature_kind=signature_kind,
                        quantity_name=quantity_name,
                        component_name=parsed_estimator.name,
                        available_quantities=available_quantities,
                    )
                )
