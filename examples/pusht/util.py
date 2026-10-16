"""
Geometry Helpers for the PushT Example

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch

# Below this ratio of |area| to the vertices' mean squared spread the polygon counts as degenerate.
DEGENERATE_AREA_RATIO = 0.35


def polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    """Area-weighted (shoelace) centroid of a polygon; falls back to the vertex mean if it is degenerate."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)

    cross = x * y_next - x_next * y
    A = 0.5 * torch.sum(cross)

    mean = torch.mean(vertices, dim=0)
    r2 = torch.mean(torch.sum((vertices - mean) ** 2, dim=-1))
    if torch.abs(A) <= DEGENERATE_AREA_RATIO * r2:
        return mean

    Cx = torch.sum((x + x_next) * cross) / (6 * A)
    Cy = torch.sum((y + y_next) * cross) / (6 * A)
    return torch.stack([Cx, Cy])
