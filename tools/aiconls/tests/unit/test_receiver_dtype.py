from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lsprotocol import types
from tensordict import TensorDict

from dynamic import diff_checker

RANGE = types.Range(
    start=types.Position(line=0, character=0),
    end=types.Position(line=0, character=1),
)


def _check_connection(c_func: Any, quantities: dict[str, torch.Tensor]) -> Any:
    """Derivative check of a float64 connection whose quantities come from components of other dtypes."""
    connection = SimpleNamespace(
        c_func=c_func,
        connected_quantities=TensorDict(quantities, batch_size=()),
        dtype=torch.float64,
    )
    parsed_connection = SimpleNamespace(
        connection_func_params={name: None for name in quantities}
    )
    return diff_checker._validate_connection_func_differentiability(  # pyright: ignore[reportPrivateUsage]
        connection_instance_name="Mixed",
        connection_class_name="Mixed",
        connection_instance=cast(Any, connection),
        parsed_connection=cast(Any, parsed_connection),
        diagnostic_range=RANGE,
    )


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # einsum, unlike most operations, does not promote mixed dtypes
    return torch.einsum("i,i->", a, b)


MIXED = {
    "a": torch.ones(3, dtype=torch.float32),
    "b": torch.ones(3, dtype=torch.float64),
}


def test_inputs_are_converted_to_the_receiver_dtype() -> None:
    assert _check_connection(_dot, MIXED) is None


def test_without_conversion_mixed_dtypes_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def unchanged(inputs: Any, dtype: torch.dtype) -> Any:
        return inputs

    monkeypatch.setattr(diff_checker, "_as_receiver_dtype", unchanged)

    diagnostic = _check_connection(_dot, MIXED)

    assert diagnostic is not None
    assert "Double" in diagnostic.message or "Float" in diagnostic.message


def test_other_failures_are_still_reported() -> None:
    def wrong_shapes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("i,ij->", a, b)

    assert _check_connection(wrong_shapes, MIXED) is not None


def test_only_floating_point_inputs_are_converted() -> None:
    inputs = TensorDict(
        {
            "x": torch.ones(2, dtype=torch.float32),
            "n": torch.ones(2, dtype=torch.int64),
        },
        batch_size=(),
    )

    converted = diff_checker._as_receiver_dtype(inputs, torch.float64)  # pyright: ignore[reportPrivateUsage]

    assert converted["x"].dtype == torch.float64
    assert converted["n"].dtype == torch.int64
