# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportPrivateUsage=false

from types import SimpleNamespace
from typing import Any, cast

import torch
from lsprotocol import types
from tensordict import TensorDict

import dynamic.diff_checker as analyzer_module
from dynamic.loader import InstantiationFailure
from reader.spec.spec import AICON_SPEC


def _range() -> types.Range:
    zero = types.Position(line=0, character=0)
    return types.Range(start=zero, end=zero)


class _FakeEstimationComponent:
    pass


class _FakeActiveInterconnection:
    def __init__(self) -> None:
        self.c_func = lambda value: value
        self.connected_quantities = {"value": torch.tensor([0.0])}


class _FakeEstimator(_FakeEstimationComponent):
    def __init__(self) -> None:
        self.timestamp = 0
        self.quantities = {}
        self.connections = {}
        self.f_input_signature = ["value"]
        self.idxs_diff_against_args = None
        self.f_output_signature_diff = ["value"]
        self.f_func = lambda value: (value, (value,))

    def get_dt(self, _new_time):
        return 1


def test_dynamic_analysis_reports_non_differentiable_estimator(
    monkeypatch,
) -> None:
    estimator = _FakeEstimator()
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///estimators.py",
    )
    instantiated_component = SimpleNamespace(
        instance=estimator,
        class_name="TestEstimator",
        parsed_component=parsed_component,
    )
    instantiated_network = SimpleNamespace(
        components={"EstimatorInstance": instantiated_component},
        connections={},
    )

    monkeypatch.setattr(
        analyzer_module, "EstimationComponent", _FakeEstimationComponent
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_collect_estimator_jacfwd_arguments",
        lambda *_args, **_kwargs: (["fake-arg"], (0,)),
    )

    def _failing_jacfwd(*_args, **_kwargs):
        def _runner(*_fn_args):
            raise RuntimeError("not differentiable")

        return _runner

    monkeypatch.setattr(analyzer_module.torch.func, "jacfwd", _failing_jacfwd)

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )

    assert list(diagnostics_by_uri.keys()) == ["file:///estimators.py"]
    diagnostics = diagnostics_by_uri["file:///estimators.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E108"
    assert "EstimatorInstance (TestEstimator)" in diagnostics[0].message
    assert "RuntimeError: not differentiable" in diagnostics[0].message


def test_dynamic_analysis_accepts_differentiable_estimator(monkeypatch) -> None:
    estimator = _FakeEstimator()
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///estimators.py",
    )
    instantiated_component = SimpleNamespace(
        instance=estimator,
        class_name="TestEstimator",
        parsed_component=parsed_component,
    )
    instantiated_network = SimpleNamespace(
        components={"EstimatorInstance": instantiated_component},
        connections={},
    )

    monkeypatch.setattr(
        analyzer_module, "EstimationComponent", _FakeEstimationComponent
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_collect_estimator_jacfwd_arguments",
        lambda *_args, **_kwargs: (["fake-arg"], (0,)),
    )

    def _passing_jacfwd(*_args, **_kwargs):
        def _runner(*_fn_args):
            return (((torch.tensor([1.0]),),), ())

        return _runner

    monkeypatch.setattr(analyzer_module.torch.func, "jacfwd", _passing_jacfwd)

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )

    assert diagnostics_by_uri == {}


def test_dynamic_analysis_reports_zero_jacobian_estimator(monkeypatch) -> None:
    estimator = _FakeEstimator()
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///estimators.py",
    )
    instantiated_component = SimpleNamespace(
        instance=estimator,
        class_name="TestEstimator",
        parsed_component=parsed_component,
    )
    instantiated_network = SimpleNamespace(
        components={"EstimatorInstance": instantiated_component},
        connections={},
    )

    monkeypatch.setattr(
        analyzer_module, "EstimationComponent", _FakeEstimationComponent
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_collect_estimator_jacfwd_arguments",
        lambda *_args, **_kwargs: (["fake-arg"], (0,)),
    )

    def _zero_jacfwd(*_args, **_kwargs):
        def _runner(*_fn_args):
            return (((torch.tensor([0.0]),),), ())

        return _runner

    monkeypatch.setattr(analyzer_module.torch.func, "jacfwd", _zero_jacfwd)

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )

    diagnostics = diagnostics_by_uri["file:///estimators.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E108"
    assert "all computed partial derivatives are zero" in diagnostics[0].message


def test_sample_allowed_tensor_values_respects_common_ranges() -> None:
    likelihood = analyzer_module._sample_allowed_tensor_values(
        quantity_name="likelihood_grasped_drawer",
        tensor=torch.zeros(1),
    )
    dt = analyzer_module._sample_allowed_tensor_values(
        quantity_name="dt",
        tensor=torch.zeros(1),
    )
    uncertainty = analyzer_module._sample_allowed_tensor_values(
        quantity_name="uncertainty_drawer",
        tensor=torch.zeros(3, 3),
    )

    assert torch.all(likelihood > 0)
    assert torch.all(likelihood < 1)
    assert torch.all(dt > 0)
    assert uncertainty.shape == (3, 3)
    assert torch.all(torch.diagonal(uncertainty) > 0)


def test_dynamic_analysis_reports_non_differentiable_connection_func(
    monkeypatch,
) -> None:
    parsed_connection = SimpleNamespace(
        position=_range(),
        document_uri="file:///connections.py",
        connection_func_params={"value": _range()},
    )
    fake_connection = _FakeActiveInterconnection()
    instantiated_connection = SimpleNamespace(
        instance=fake_connection,
        class_name="TestConnection",
        parsed_connection=parsed_connection,
    )
    instantiated_network = SimpleNamespace(
        components={},
        connections={"ConnectionInstance": instantiated_connection},
    )

    monkeypatch.setattr(
        analyzer_module,
        "ActiveInterconnection",
        _FakeActiveInterconnection,
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_collect_connection_jacfwd_arguments",
        lambda *_args, **_kwargs: ([torch.tensor([0.0])], (0,)),
    )

    def _failing_jacfwd(*_args, **_kwargs):
        def _runner(*_fn_args):
            raise RuntimeError("connection not differentiable")

        return _runner

    monkeypatch.setattr(analyzer_module.torch.func, "jacfwd", _failing_jacfwd)

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )
    diagnostics = diagnostics_by_uri["file:///connections.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E208"
    assert "ConnectionInstance (TestConnection)" in diagnostics[0].message


def test_dynamic_analysis_reports_zero_jacobian_connection_func(monkeypatch) -> None:
    parsed_connection = SimpleNamespace(
        position=_range(),
        document_uri="file:///connections.py",
        connection_func_params={"value": _range()},
    )
    fake_connection = _FakeActiveInterconnection()
    instantiated_connection = SimpleNamespace(
        instance=fake_connection,
        class_name="TestConnection",
        parsed_connection=parsed_connection,
    )
    instantiated_network = SimpleNamespace(
        components={},
        connections={"ConnectionInstance": instantiated_connection},
    )

    monkeypatch.setattr(
        analyzer_module,
        "ActiveInterconnection",
        _FakeActiveInterconnection,
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_collect_connection_jacfwd_arguments",
        lambda *_args, **_kwargs: ([torch.tensor([0.0])], (0,)),
    )

    def _zero_jacfwd(*_args, **_kwargs):
        def _runner(*_fn_args):
            return torch.tensor([0.0])

        return _runner

    monkeypatch.setattr(analyzer_module.torch.func, "jacfwd", _zero_jacfwd)

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )
    diagnostics = diagnostics_by_uri["file:///connections.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E208"
    assert "all computed partial derivatives are zero" in diagnostics[0].message


def test_dynamic_analysis_reports_runtime_shape_mismatch_before_pass(
    monkeypatch,
) -> None:
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///sensor.py",
    )
    instantiated_component = SimpleNamespace(
        instance=SimpleNamespace(),
        class_name="SensorImpl",
        parsed_component=parsed_component,
    )
    instantiated_network = SimpleNamespace(
        components={"sensor_instance": instantiated_component},
        connections={},
    )

    shape_check_call_count = 0

    def _shape_mismatches(*_args, **_kwargs):
        nonlocal shape_check_call_count
        shape_check_call_count += 1
        if shape_check_call_count == 1:
            return {"sensor_instance": ("value: expected shape (1,), got (2,)",)}
        return {}

    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "get_component_quantity_shape_mismatch_details",
        _shape_mismatches,
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_network_pass_with_random_sensor_values",
        lambda *_args, **_kwargs: None,
    )

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, SimpleNamespace(components={})),
        {},
        {},
        {},
    )

    diagnostics = diagnostics_by_uri["file:///sensor.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E102"
    assert "sensor_instance (SensorImpl)" in diagnostics[0].message
    assert "after initialization" in diagnostics[0].message
    assert shape_check_call_count == 3


def test_dynamic_analysis_reports_runtime_shape_mismatch_after_pass(
    monkeypatch,
) -> None:
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///sensor.py",
    )
    instantiated_component = SimpleNamespace(
        instance=SimpleNamespace(),
        class_name="SensorImpl",
        parsed_component=parsed_component,
    )
    instantiated_network = SimpleNamespace(
        components={"sensor_instance": instantiated_component},
        connections={},
    )

    shape_check_call_count = 0

    def _shape_mismatches(*_args, **_kwargs):
        nonlocal shape_check_call_count
        shape_check_call_count += 1
        if shape_check_call_count == 2:
            return {"sensor_instance": ("value: expected shape (1,), got (3,)",)}
        return {}

    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )
    monkeypatch.setattr(
        analyzer_module,
        "get_component_quantity_shape_mismatch_details",
        _shape_mismatches,
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_network_pass_with_random_sensor_values",
        lambda *_args, **_kwargs: None,
    )

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, SimpleNamespace(components={})),
        {},
        {},
        {},
    )

    diagnostics = diagnostics_by_uri["file:///sensor.py"]
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "E102"
    assert "sensor_instance (SensorImpl)" in diagnostics[0].message
    assert "after runtime pass 1" in diagnostics[0].message
    assert shape_check_call_count == 3


def test_dynamic_analysis_runs_three_check_cycles_in_order(monkeypatch) -> None:
    estimator = _FakeEstimator()
    parsed_component = SimpleNamespace(
        position=_range(),
        document_uri="file:///estimators.py",
    )
    instantiated_component = SimpleNamespace(
        instance=estimator,
        class_name="TestEstimator",
        parsed_component=parsed_component,
    )
    parsed_connection = SimpleNamespace(
        position=_range(),
        document_uri="file:///connections.py",
        connection_func_params={"value": _range()},
    )
    instantiated_connection = SimpleNamespace(
        instance=_FakeActiveInterconnection(),
        class_name="TestConnection",
        parsed_connection=parsed_connection,
    )
    instantiated_network = SimpleNamespace(
        components={"EstimatorInstance": instantiated_component},
        connections={"ConnectionInstance": instantiated_connection},
    )
    call_sequence: list[str] = []

    monkeypatch.setattr(
        analyzer_module, "EstimationComponent", _FakeEstimationComponent
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )

    def _shape_mismatches(*_args, **_kwargs):
        call_sequence.append("shape")
        return {}

    def _estimator_diff(*_args, **_kwargs):
        call_sequence.append("jacobian:estimator")

    def _connection_diff(*_args, **_kwargs):
        call_sequence.append("jacobian:connection")

    def _run_sequence(*_args, **kwargs):
        timestamp = kwargs["new_time"]
        call_sequence.append(f"run:{int(timestamp.item())}")

    monkeypatch.setattr(
        analyzer_module,
        "get_component_quantity_shape_mismatch_details",
        _shape_mismatches,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_validate_estimator_f_func_differentiability",
        _estimator_diff,
    )
    monkeypatch.setattr(
        analyzer_module,
        "_validate_connection_func_differentiability",
        _connection_diff,
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_network_pass_with_random_sensor_values",
        _run_sequence,
    )

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, SimpleNamespace(components={})),
        {},
        {},
        {},
    )

    assert diagnostics_by_uri == {}
    assert call_sequence == [
        "shape",
        "jacobian:estimator",
        "jacobian:connection",
        "run:0",
        "shape",
        "jacobian:estimator",
        "jacobian:connection",
        "run:1",
        "shape",
        "jacobian:estimator",
        "jacobian:connection",
        "run:2",
    ]


def _validate_fake_connection(connection_func_params: list[str]):
    parsed_connection = SimpleNamespace(
        connection_func_params={name: _range() for name in connection_func_params},
    )
    connection = _FakeActiveInterconnection()
    connection.connected_quantities = TensorDict(
        {"value": torch.tensor([0.0])}, batch_size=()
    )
    return analyzer_module._validate_connection_func_differentiability(
        connection_instance_name="ConnectionInstance",
        connection_class_name="TestConnection",
        connection_instance=cast(Any, connection),
        parsed_connection=cast(Any, parsed_connection),
        diagnostic_range=_range(),
    )


def test_connection_func_parameter_without_quantity_is_reported() -> None:
    diagnostic = _validate_fake_connection(["unknown_value"])

    assert diagnostic is not None
    assert diagnostic.code == "E208"
    assert "'unknown_value'" in diagnostic.message
    assert "not a quantity of the connection" in diagnostic.message


def test_connection_func_without_parameters_is_not_reported() -> None:
    assert _validate_fake_connection([]) is None


def test_estimator_f_func_parameter_without_input_is_reported() -> None:
    estimator = _FakeEstimator()
    estimator.quantities = TensorDict({"value": torch.tensor([0.0])}, batch_size=())
    estimator.f_input_signature = ["value", "unknown_value"]
    estimator.f_output_signature_diff = []

    diagnostic = analyzer_module._validate_estimator_f_func_differentiability(
        component_instance_name="EstimatorInstance",
        component_class_name="TestEstimator",
        estimator_instance=cast(Any, estimator),
        diagnostic_range=_range(),
    )

    assert diagnostic is not None
    assert diagnostic.code == "E108"
    assert "'unknown_value'" in diagnostic.message


def test_estimator_f_func_without_differentiable_arguments_is_not_reported() -> None:
    estimator = _FakeEstimator()
    estimator.quantities = TensorDict({"value": torch.tensor([0.0])}, batch_size=())

    diagnostic = analyzer_module._validate_estimator_f_func_differentiability(
        component_instance_name="EstimatorInstance",
        component_class_name="TestEstimator",
        estimator_instance=cast(Any, estimator),
        diagnostic_range=_range(),
    )

    assert diagnostic is None


def test_dynamic_analysis_reports_instantiation_failures(monkeypatch) -> None:
    failures = (
        InstantiationFailure(
            kind="component",
            instance_name="EstimatorInstance",
            class_name="TestEstimator",
            document_uri="file:///estimators.py",
            position=_range(),
            reason="KeyError: 'DirectMeasurement'",
        ),
        InstantiationFailure(
            kind="connection",
            instance_name="ConnectionInstance",
            class_name="TestConnection",
            document_uri="file:///connections.py",
            position=_range(),
            reason="RuntimeError: Missing constructor parameters: offset.",
        ),
    )
    instantiated_network = SimpleNamespace(
        components={}, connections={}, instantiation_failures=failures
    )
    monkeypatch.setattr(
        analyzer_module,
        "instantiate_network_from_spec",
        lambda *_args, **_kwargs: instantiated_network,
    )

    diagnostics_by_uri = analyzer_module.check_differentials(
        cast(AICON_SPEC, object()), {}, {}, {}
    )

    [component_diagnostic] = diagnostics_by_uri["file:///estimators.py"]
    [connection_diagnostic] = diagnostics_by_uri["file:///connections.py"]
    assert component_diagnostic.code == "W104"
    assert "Component 'EstimatorInstance' (TestEstimator)" in (
        component_diagnostic.message
    )
    assert "KeyError: 'DirectMeasurement'" in component_diagnostic.message
    assert connection_diagnostic.code == "W203"
    assert "Connection 'ConnectionInstance' (TestConnection)" in (
        connection_diagnostic.message
    )
