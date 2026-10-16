"""Timestamps keep sub-second resolution at epoch scale, also in float32 components."""

import pytest
import torch

from aicon_base.components import EstimationComponent
from aicon_base.timestamps import TIME_DTYPE


class ElapsedTime(EstimationComponent):
    """Integrates dt, so its state is the time elapsed since initialisation."""

    cls_quantity_names_and_shapes = {"elapsed": (1,)}

    def initialize_quantities(self):
        return True

    def define_estimation_function_f(self):
        def f_func(elapsed, dt):
            new = elapsed + dt
            return (new,), (new,)

        return f_func, ["elapsed"], ["elapsed"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dt_at_epoch_time(dtype):
    component = ElapsedTime("ElapsedTime", {}, dtype=dtype, device=torch.device("cpu"), no_differentiation=True)
    t0 = 1.79e9
    for i in range(4):
        component.update(torch.tensor(t0 + 0.1 * i, dtype=TIME_DTYPE))
    assert component.quantities["elapsed"].dtype == dtype
    assert component.timestamp.dtype == TIME_DTYPE
    assert component.quantities["elapsed"].item() == pytest.approx(0.3, abs=1e-4)
