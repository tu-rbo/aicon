import math

import torch

from aicon_base.math.angles import normalize_angle, shortest_angular_distance


def test_normalize_angle_range():
    angles = torch.linspace(-20.0, 20.0, 2001)
    wrapped = normalize_angle(angles)
    assert torch.all(wrapped > -math.pi - 1e-12) and torch.all(wrapped <= math.pi + 1e-12)
    assert torch.allclose(torch.sin(wrapped), torch.sin(angles), atol=1e-9)
    assert torch.allclose(torch.cos(wrapped), torch.cos(angles), atol=1e-9)


def test_shortest_angular_distance_matches_scalar_reference():
    def reference(a, b):
        d = (b - a) % (2.0 * math.pi)
        return d - 2.0 * math.pi if d > math.pi else d

    g = torch.Generator().manual_seed(0)
    a = (torch.rand(500, generator=g) - 0.5) * 20
    b = (torch.rand(500, generator=g) - 0.5) * 20
    ours = shortest_angular_distance(a, b)
    for i in range(len(a)):
        assert ours[i].item() == reference(a[i].item(), b[i].item())


def test_shortest_angular_distance_gradient():
    a = torch.tensor([0.3], requires_grad=True)
    b = torch.tensor([3.0])
    shortest_angular_distance(a, b).sum().backward()
    assert a.grad.item() == -1.0
