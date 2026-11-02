from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
from aicon_base.components.estimation_components import EstimationComponent
from aicon_base.connections.connections import ActiveInterconnection
from lsprotocol.types import Diagnostic, Range

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from dynamic.loader import (
    InstantiationFailure,
    get_component_quantity_shape_mismatch_details,
    instantiate_network_from_spec,
    run_network_pass_with_random_sensor_values,
)
from dynamic.randomization import (
    randomize_tensor_mapping,
)
from dynamic.randomization import (
    sample_allowed_tensor_values as _sample_allowed_tensor_values_impl,
)
from reader.aicon_files.reader_types import (
    ParsedComponents,
    ParsedConnection,
    ParsedConnections,
    ParsedGoals,
)
from reader.spec.spec import AICON_SPEC

_DYNAMIC_CHECK_CYCLES = 3


# random inputs tried before a function counts as having all-zero derivatives; functions that are flat in most of
# their input space (e.g. a field-of-view likelihood, non-zero for ~1% of random poses) need many
DERIVATIVE_CHECK_SAMPLES = 1024


def _seeded_draw(sample: int, draw: Callable[[], Any]) -> Any:
    """`draw()` with the random generator seeded per sample, so that the check gives the same result every run; the
    generator's state is restored afterwards."""
    state = torch.random.get_rng_state()
    torch.manual_seed(1000 + sample)  # pyright: ignore[reportUnknownMemberType]
    try:
        return draw()
    finally:
        torch.random.set_rng_state(state)


def check_differentials(
    spec: AICON_SPEC,
    parsed_components: ParsedComponents,
    parsed_goals: ParsedGoals,
    parsed_connections: ParsedConnections,
):

    d_instance = instantiate_network_from_spec(
        spec, parsed_components, parsed_goals, parsed_connections
    )
    diagnostics_by_uri: dict[str, list[Diagnostic]] = {}
    for failure in getattr(d_instance, "instantiation_failures", ()):
        _append_instantiation_failure_diagnostic(diagnostics_by_uri, failure)
    has_component_spec = hasattr(spec, "components")
    flagged_estimator_instances: set[str] = set()
    flagged_connection_instances: set[str] = set()
    for check_iteration in range(_DYNAMIC_CHECK_CYCLES):
        if has_component_spec:
            _append_runtime_shape_diagnostics(
                diagnostics_by_uri=diagnostics_by_uri,
                spec=spec,
                d_instance=d_instance,
                phase_label=_shape_phase_label(check_iteration),
            )
        _append_differentiability_diagnostics(
            diagnostics_by_uri=diagnostics_by_uri,
            d_instance=d_instance,
            flagged_estimator_instances=flagged_estimator_instances,
            flagged_connection_instances=flagged_connection_instances,
        )
        if has_component_spec:
            run_network_pass_with_random_sensor_values(
                spec=spec,
                instantiation=d_instance,
                new_time=torch.tensor([float(check_iteration)]),
            )

    return diagnostics_by_uri


def _append_instantiation_failure_diagnostic(
    diagnostics_by_uri: dict[str, list[Diagnostic]], failure: InstantiationFailure
) -> None:
    if failure.kind == "connection":
        diagnostic = make_diagnostic(
            ErrorCode.CONNECTION_NOT_INSTANTIABLE,
            range=failure.position,
            instance_name=failure.instance_name,
            class_name=failure.class_name,
            reason=failure.reason,
        )
    else:
        diagnostic = make_diagnostic(
            ErrorCode.COMPONENT_NOT_INSTANTIABLE,
            range=failure.position,
            entity_kind=failure.kind.capitalize(),
            instance_name=failure.instance_name,
            class_name=failure.class_name,
            reason=failure.reason,
        )
    diagnostics_by_uri.setdefault(failure.document_uri, []).append(diagnostic)


def _shape_phase_label(check_iteration: int) -> str:
    if check_iteration == 0:
        return "after initialization"
    return f"after runtime pass {check_iteration}"


def _append_differentiability_diagnostics(
    *,
    diagnostics_by_uri: dict[str, list[Diagnostic]],
    d_instance: Any,
    flagged_estimator_instances: set[str],
    flagged_connection_instances: set[str],
) -> None:
    for component_instance_name, component in d_instance.components.items():
        if component_instance_name in flagged_estimator_instances:
            continue
        if not isinstance(component.instance, EstimationComponent):
            continue

        maybe_diagnostic = _validate_estimator_f_func_differentiability(
            component_instance_name=component_instance_name,
            component_class_name=component.class_name,
            estimator_instance=component.instance,
            diagnostic_range=component.parsed_component.position,
        )
        if maybe_diagnostic is None:
            continue
        flagged_estimator_instances.add(component_instance_name)
        diagnostics_by_uri.setdefault(
            component.parsed_component.document_uri, []
        ).append(maybe_diagnostic)

    for connection_instance_name, connection in d_instance.connections.items():
        if connection_instance_name in flagged_connection_instances:
            continue
        maybe_diagnostic = _validate_connection_func_differentiability(
            connection_instance_name=connection_instance_name,
            connection_class_name=connection.class_name,
            connection_instance=connection.instance,
            parsed_connection=connection.parsed_connection,
            diagnostic_range=connection.parsed_connection.position,
        )
        if maybe_diagnostic is None:
            continue
        flagged_connection_instances.add(connection_instance_name)
        diagnostics_by_uri.setdefault(
            connection.parsed_connection.document_uri, []
        ).append(maybe_diagnostic)


def _append_runtime_shape_diagnostics(
    *,
    diagnostics_by_uri: dict[str, list[Diagnostic]],
    spec: AICON_SPEC,
    d_instance: Any,
    phase_label: str,
) -> None:
    mismatches_by_component = get_component_quantity_shape_mismatch_details(
        spec, d_instance
    )
    for component_instance_name, mismatch_details in mismatches_by_component.items():
        component = d_instance.components.get(component_instance_name)
        if component is None:
            continue
        component_name = f"{component_instance_name} ({component.class_name})"
        maybe_diagnostic = make_diagnostic(
            ErrorCode.SHAPE_SPEC_MISMATCH,
            range=component.parsed_component.position,
            component_name=component_name,
            mismatch_details=f"{phase_label}: {'; '.join(mismatch_details)}",
        )
        diagnostics_by_uri.setdefault(
            component.parsed_component.document_uri, []
        ).append(maybe_diagnostic)


def _validate_estimator_f_func_differentiability(
    *,
    component_instance_name: str,
    component_class_name: str,
    estimator_instance: EstimationComponent,
    diagnostic_range: Range,
) -> Diagnostic | None:
    if not hasattr(estimator_instance, "f_func"):
        return None

    try:
        args, diff_argnums = _collect_estimator_jacfwd_arguments(estimator_instance)
    except KeyError as error:
        component_name = f"{component_instance_name} ({component_class_name})"
        return make_diagnostic(
            ErrorCode.ESTIMATOR_F_FUNC_NOT_DIFFERENTIABLE,
            range=diagnostic_range,
            component_name=component_name,
            reason=f"f_func parameter {error} is not available as input",
        )
    if not diff_argnums:
        # nothing to differentiate, e.g. a freshly generated skeleton
        return None
    estimator = cast(Any, estimator_instance)
    f_func = cast(Callable[..., object], estimator.f_func)

    # A function can be flat at one random input (e.g. a saturated likelihood) while it is differentiable elsewhere:
    # only all-zero derivatives at every one of several random inputs count as not differentiable.
    for sample in range(DERIVATIVE_CHECK_SAMPLES):
        args, diff_argnums = _seeded_draw(
            sample, lambda: _collect_estimator_jacfwd_arguments(estimator_instance)
        )
        try:
            partial_jacs, _ = torch.func.jacfwd(
                f_func,
                argnums=diff_argnums,
                has_aux=True,
                randomness="same",
            )(*args)
        except Exception as error:  # noqa: BLE001 - any failure of user code counts
            component_name = f"{component_instance_name} ({component_class_name})"
            return make_diagnostic(
                ErrorCode.ESTIMATOR_F_FUNC_NOT_DIFFERENTIABLE,
                range=diagnostic_range,
                component_name=component_name,
                reason=f"{error.__class__.__name__}: {error}",
            )
        if not _all_jacobian_entries_are_zero(partial_jacs):
            return None
    # all samples gave all-zero derivatives
    component_name = f"{component_instance_name} ({component_class_name})"
    return make_diagnostic(
        ErrorCode.ESTIMATOR_F_FUNC_NOT_DIFFERENTIABLE,
        range=diagnostic_range,
        component_name=component_name,
        reason="all computed partial derivatives are zero",
    )

    return None


def _collect_estimator_jacfwd_arguments(
    estimator_instance: EstimationComponent,
) -> tuple[list[Any], tuple[int, ...]]:
    estimator = cast(Any, estimator_instance)
    new_time = estimator.timestamp + 1
    all_inputs = estimator.quantities.clone()
    for connection in estimator.connections.values():
        with connection.lock:
            all_inputs = all_inputs.update(connection.connected_quantities)
    all_inputs["dt"] = estimator.get_dt(new_time)
    all_inputs = _as_receiver_dtype(all_inputs, getattr(estimator, "dtype", None))
    all_inputs = _randomize_inputs_for_differentiability_check(all_inputs)

    f_input_signature = cast(list[str], estimator.f_input_signature)
    f_output_signature_diff = cast(list[str], estimator.f_output_signature_diff)
    # AICON names the f_func arguments to differentiate against (differentiate_against); older versions gave their
    # indices (idxs_diff_against_args). Without either, all arguments except the outputs and dt are differentiated.
    diff_names = cast(
        list[str] | tuple[str, ...] | None,
        getattr(estimator, "differentiate_against", None),
    )
    if diff_names is not None:
        diff_argnums = tuple(
            f_input_signature.index(name)
            for name in diff_names
            if name in f_input_signature
        )
    else:
        diff_argnums = cast(
            tuple[int, ...] | None, getattr(estimator, "idxs_diff_against_args", None)
        )
    if diff_argnums is None:
        diff_argnums = tuple(
            index
            for index, argument_name in enumerate(f_input_signature)
            if argument_name not in [*f_output_signature_diff, "dt"]
        )

    _raise_for_missing_inputs(f_input_signature, all_inputs)
    args = list(all_inputs.select(*f_input_signature).values())
    return args, diff_argnums


def _validate_connection_func_differentiability(
    *,
    connection_instance_name: str,
    connection_class_name: str,
    connection_instance: ActiveInterconnection,
    parsed_connection: ParsedConnection,
    diagnostic_range: Range,
) -> Diagnostic | None:
    if not hasattr(connection_instance, "c_func"):
        return None

    try:
        args, diff_argnums = _collect_connection_jacfwd_arguments(
            connection_instance=connection_instance,
            parsed_connection=parsed_connection,
        )
    except KeyError as error:
        connection_name = f"{connection_instance_name} ({connection_class_name})"
        return make_diagnostic(
            ErrorCode.CONNECTION_FUNC_NOT_DIFFERENTIABLE,
            range=diagnostic_range,
            connection_name=connection_name,
            reason=(
                f"connection_func parameter {error} is not a quantity of the "
                "connection, so its argument cannot be supplied by name"
            ),
        )
    if not diff_argnums:
        # nothing to differentiate, e.g. a freshly generated skeleton
        return None
    connection = cast(Any, connection_instance)
    c_func = cast(Callable[..., object], connection.c_func)

    # see _validate_estimator_f_func_differentiability: several random inputs before concluding all-zero
    for sample in range(DERIVATIVE_CHECK_SAMPLES):
        args, diff_argnums = _seeded_draw(
            sample,
            lambda: _collect_connection_jacfwd_arguments(
                connection_instance=connection_instance,
                parsed_connection=parsed_connection,
            ),
        )
        try:
            partial_jacs = torch.func.jacfwd(
                c_func,
                argnums=diff_argnums,
                randomness="same",
            )(*args)
        except Exception as error:  # noqa: BLE001 - any failure of user code counts
            connection_name = f"{connection_instance_name} ({connection_class_name})"
            return make_diagnostic(
                ErrorCode.CONNECTION_FUNC_NOT_DIFFERENTIABLE,
                range=diagnostic_range,
                connection_name=connection_name,
                reason=f"{error.__class__.__name__}: {error}",
            )
        if not _all_jacobian_entries_are_zero(partial_jacs):
            return None
    # all samples gave all-zero derivatives
    connection_name = f"{connection_instance_name} ({connection_class_name})"
    return make_diagnostic(
        ErrorCode.CONNECTION_FUNC_NOT_DIFFERENTIABLE,
        range=diagnostic_range,
        connection_name=connection_name,
        reason="all computed partial derivatives are zero",
    )


def _collect_connection_jacfwd_arguments(
    *,
    connection_instance: ActiveInterconnection,
    parsed_connection: ParsedConnection,
) -> tuple[list[Any], tuple[int, ...]]:
    connection = cast(Any, connection_instance)
    all_inputs = _randomize_inputs_for_differentiability_check(
        _as_receiver_dtype(
            connection.connected_quantities.clone(), getattr(connection, "dtype", None)
        )
    )
    input_signature = list(parsed_connection.connection_func_params.keys())
    diff_argnums = tuple(
        index
        for index, argument_name in enumerate(input_signature)
        if argument_name != "dt"
    )
    _raise_for_missing_inputs(input_signature, all_inputs)
    args = [all_inputs[argument_name] for argument_name in input_signature]
    return args, diff_argnums


def _as_receiver_dtype(all_inputs: Any, dtype: torch.dtype | None) -> Any:
    """Floating-point inputs in the dtype of the component or connection that uses them. Components of one system can
    have different dtypes (e.g. a float32 estimator next to float64 ones); the ROS middleware converts every quantity
    it receives to the dtype of the receiving component, so the check does the same."""
    if dtype is None:
        return all_inputs
    for name, value in list(all_inputs.items()):
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dtype != dtype
        ):
            all_inputs[name] = value.to(dtype)
    return all_inputs


def _raise_for_missing_inputs(input_signature: list[str], all_inputs: Any) -> None:
    available_names = set(all_inputs.keys())
    for argument_name in input_signature:
        if argument_name not in available_names:
            raise KeyError(argument_name)


def _all_jacobian_entries_are_zero(jacobian: Any) -> bool:
    jacobian_tensors = list(_iter_jacobian_tensors(jacobian))
    if not jacobian_tensors:
        return False
    return all(torch.count_nonzero(tensor).item() == 0 for tensor in jacobian_tensors)


def _iter_jacobian_tensors(jacobian: Any) -> Iterator[torch.Tensor]:
    if isinstance(jacobian, torch.Tensor):
        yield jacobian
        return
    if isinstance(jacobian, (tuple, list)):
        for value in cast(list[Any] | tuple[Any, ...], jacobian):
            yield from _iter_jacobian_tensors(value)


def _randomize_inputs_for_differentiability_check(all_inputs: Any) -> Any:
    return randomize_tensor_mapping(all_inputs)


def _sample_allowed_tensor_values(  # pyright: ignore[reportUnusedFunction]
    *, quantity_name: str, tensor: torch.Tensor
) -> torch.Tensor:
    return _sample_allowed_tensor_values_impl(
        quantity_name=quantity_name,
        tensor=tensor,
    )
