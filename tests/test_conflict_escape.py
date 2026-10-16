"""The conflict escape steers sideways along the quantity's recent motion; without motion it has no direction and
must pass the gradients on unchanged instead of producing NaNs."""

from collections import deque

import torch

from aicon_base.derivatives.derivative_maintenance import QuantityRelatedDerivativeBlock
from aicon_base.derivatives.derivative_processing import project_gradients_with_conflict_resolution
from aicon_base.derivatives.projections import escape_conflicts


def opposing_gradients():
    block = QuantityRelatedDerivativeBlock("q")
    block.derivatives_tensor = torch.tensor([[[1.0, 0.0]], [[-0.9, 0.05]]], dtype=torch.float64)
    block.quantity_traces = [["GoalA", "q"], ["GoalB", "q"]]
    return block


def escape(history_newest_first):
    original = opposing_gradients()
    grad_norm = torch.norm(original.derivatives_tensor.view(2, -1), dim=1)
    idx_sorted = torch.argsort(grad_norm, descending=True)
    updated_conflicts = dict()
    result = escape_conflicts(
        dict(),
        grad_norm,
        opposing_gradients(),
        original,
        {"q": deque(torch.tensor(v, dtype=torch.float64) for v in history_newest_first)},
        "q",
        updated_conflicts,
        idx_sorted,
        dict(),
    )
    return result, updated_conflicts


def test_escape_follows_sideways_motion():
    result, updated_conflicts = escape([[1.0, 1.0]] + [[0.0, 0.0]] * 4)
    assert updated_conflicts["q"] == 0
    # the motion [1, 1] without its component along the dominant gradient [1, 0], at 1.05 times its length
    assert torch.allclose(result.derivatives_tensor, torch.tensor([[[0.0, 1.05]], [[0.0, 0.0]]], dtype=torch.float64))
    assert result.quantity_traces[0][1].startswith("ConflictResolution(")


def test_escape_without_motion_keeps_gradients():
    result, updated_conflicts = escape([[0.0, 0.0]] * 5)
    assert updated_conflicts["q"] == 0  # the conflict stays active, so the escape starts once the quantity moves
    assert torch.equal(result.derivatives_tensor, opposing_gradients().derivatives_tensor)
    assert result.quantity_traces == opposing_gradients().quantity_traces


def phase_out(diff_vector):
    """One tick of the batched resolution while a conflict from the previous tick phases out."""
    # one input element, two batch candidates on a 2D quantity; they agree, so no new conflict fires
    t = torch.tensor([[[[1.0, 0.0], [0.5, 0.1]]]], dtype=torch.float64)
    history = [torch.tensor([[0.0, 0.0]], dtype=torch.float64)] * 5
    return project_gradients_with_conflict_resolution(
        t,
        select_along_dimensions=(2,),
        curr_diff_vector=torch.tensor(diff_vector, dtype=torch.float64),
        curr_conflict={0: 1},
        output_buffer=history,
        conflict_entry_points=dict(),
    )


def test_batched_phase_out_follows_sideways_motion():
    new_t, resolved, _ = phase_out([1.0, 1.0])
    assert resolved[0] == 2
    assert torch.allclose(new_t, torch.tensor([[[0.0, -1.0]]], dtype=torch.float64))


def test_batched_phase_out_without_motion_keeps_gradient():
    new_t, resolved, _ = phase_out([0.0, 0.0])
    assert resolved[0] == 2
    assert torch.equal(new_t, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64))
