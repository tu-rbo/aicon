"""
Geometry Helpers for Real-Robot Object Pushing

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import torch
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# Below this ratio of |area| to the vertices' mean squared spread the polygon counts as degenerate.
DEGENERATE_AREA_RATIO = 0.35


def polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    """Area-weighted (shoelace) centroid of a polygon; falls back to the vertex mean if it is degenerate or
    self-intersecting."""
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


def point_in_polygon(point: torch.Tensor, vertices: torch.Tensor) -> bool:
    """Crossing-number test, valid for non-convex polygons. Not differentiable."""
    x, y = point[0], point[1]
    x_i, y_i = vertices[:, 0], vertices[:, 1]
    x_j, y_j = torch.roll(x_i, shifts=-1, dims=0), torch.roll(y_i, shifts=-1, dims=0)

    # half-open rule on y, so a vertex exactly at the ray's height is counted once
    straddles = (y_i > y) != (y_j > y)
    denom = torch.where(y_j - y_i == 0, torch.ones_like(y_j), y_j - y_i)
    x_cross = x_i + (y - y_i) / denom * (x_j - x_i)
    return bool(torch.sum(straddles & (x < x_cross)) % 2 == 1)


SDF_PADDING_PX = 8.0


def make_signed_distance_transform(contour_points, res=256, padding=SDF_PADDING_PX, device="cpu"):
    """Distance transform of a closed contour on a res x res grid.

    contour_points: (N,2) points in world units, centred on the origin
    padding: margin in pixels around the shape
    returns: (sdf, scale, min_xy), with grid coordinates = scale * (p - min_xy) + padding
    """
    pts = contour_points.cpu().numpy()
    # The measured points are rotated about the origin before sampling, so the grid has to hold the shape at any
    # orientation: size it by the circumradius, not by the bounding box.
    radius = float(np.max(np.linalg.norm(pts, axis=1)))
    min_xy = np.array([-radius, -radius], dtype=np.float32)
    scale = (res - 2 * padding) / (2 * radius)

    img = np.zeros((res, res), np.uint8)
    pts_scaled = np.round(scale * (pts - min_xy) + padding).astype(np.int32)
    cv2.polylines(img, [pts_scaled], isClosed=True, color=255, thickness=1)

    dist_out = distance_transform_edt(255 - img)
    sdf = torch.from_numpy(dist_out).float().to(device)
    return sdf, scale, torch.from_numpy(min_xy).to(torch.get_default_dtype()).to(device)


def sample_sdf(sdf, pts, scale, min_xy, padding=SDF_PADDING_PX):
    """Bilinear lookup of the distance transform at world points (N,2)."""
    sdf = sdf.to(dtype=pts.dtype)
    min_xy = min_xy.to(dtype=pts.dtype)
    grid = scale * (pts - min_xy) + padding  # (N,2)
    grid = grid / (sdf.shape[0] - 1) * 2 - 1  # normalise to [-1,1]
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
    vals = torch.nn.functional.grid_sample(sdf.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode="bilinear")
    return vals.squeeze()


def fit_se2_to_sdf(
    pts_measured,
    sdf,
    scale,
    min_xy,
    n_angles=1440,
    padding=SDF_PADDING_PX,
    theta_prior=None,
    prior_weight=0.0,
    solve_translation="centroid",
    point_mask=None,
    mask_scope="all",
    hold_margin=None,
    search_span=0.05,
    search_steps=7,
    search_angle_stride=12,
):
    """
    Fit the SE(2) pose that maps `pts_measured` onto a template encoded as a distance transform.

    All `n_angles` orientations are evaluated at once and the best is taken, so the angle is globally optimal. The
    translation per angle is either
      * "centroid": aligns the measurement's centroid with the template's (origin), exact for complete contours, or
      * "search": centroid alignment plus a local offset grid search (evaluated on a coarse set of angles), for
        truncated contours whose centroid is not the object's centre.

    theta_prior / prior_weight: soft orientation priors, prior_weight * wrap(theta - theta_prior)^2 added to the
        objective. Scalars or equal-length sequences (one term per cue). They resolve near-symmetric minima while a
        clearly better shape minimum still wins.
    point_mask: (N,) bool, points to use. With mask_scope="all" it applies to the translation and the residual,
        with "translation" only to the translation (the angle is still scored on all points).
    hold_margin: [m] if given, the argmin is only accepted if its shape loss beats the loss at angle 0 (the prior,
        since callers de-rotate the measurement by the prediction) by this margin; otherwise angle 0 is kept.

    Returns (theta, t, residual) with residual the mean point-to-contour distance in world units, without priors.
    """
    device, dtype = pts_measured.device, pts_measured.dtype
    res = sdf.shape[0]
    sdf = sdf.to(dtype=dtype)
    min_xy = min_xy.to(dtype=dtype)

    def _grid(pts_batched):
        # (A,N,2) world -> (1,A,N,2) normalised sampling grid
        g = (scale * (pts_batched - min_xy) + padding) / (res - 1) * 2 - 1
        return g.unsqueeze(0)

    def _sample(pts_batched):
        vals = torch.nn.functional.grid_sample(
            sdf.unsqueeze(0).unsqueeze(0),
            _grid(pts_batched),
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        return vals[0, 0]  # (A,N)

    all_idx = torch.arange(pts_measured.shape[0], device=device)
    if point_mask is None:
        m_idx = all_idx
    else:
        m_idx = torch.nonzero(point_mask.reshape(-1).to(torch.bool), as_tuple=True)[0]
        if m_idx.numel() < 2:  # nothing left to fit; fall back to the full set
            m_idx = all_idx
    idx_loss = m_idx if mask_scope == "all" else all_idx
    idx_t = m_idx

    # --- exhaustive sweep over theta ---
    thetas = torch.arange(n_angles, device=device, dtype=dtype) * (2 * torch.pi / n_angles)
    c, s = torch.cos(thetas), torch.sin(thetas)
    R_all = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)  # (A,2,2)
    pts_rot = torch.einsum("aij,nj->ani", R_all, pts_measured)  # (A,N,2)
    if solve_translation == "centroid":
        t_all = -pts_rot[:, idx_t].mean(dim=1, keepdim=True)  # (A,1,2)
    elif solve_translation == "search":
        # The best offset varies slowly with the angle (it is applied after rotation), so it is searched on a coarse
        # set of angles and reused for the nearest fine angles.
        coarse = max(int(n_angles // search_angle_stride), 8)
        c_idx = torch.linspace(0, n_angles - 1, coarse, device=device).round().long()
        off1 = torch.linspace(-search_span, search_span, search_steps, dtype=dtype, device=device)
        gx, gy = torch.meshgrid(off1, off1, indexing="ij")
        offsets = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (T,2)
        t_c = -pts_rot[c_idx][:, idx_t].mean(dim=1, keepdim=True)  # (C,1,2)
        cand = pts_rot[c_idx][:, idx_loss].unsqueeze(1) + (t_c.unsqueeze(1) + offsets.reshape(1, -1, 1, 2))  # (C,T,M,2)
        C_, T_, M_, _ = cand.shape
        r_c = _sample(cand.reshape(C_ * T_, M_, 2))  # (C*T,M)
        l_c = torch.sqrt(r_c**2 + 1e-6).mean(dim=1).reshape(C_, T_)
        best_off = offsets[torch.argmin(l_c, dim=1)]  # (C,2)
        nearest = torch.bucketize(torch.arange(n_angles, device=device, dtype=dtype), c_idx.to(dtype)).clamp(
            max=coarse - 1
        )
        base = -pts_rot[:, idx_t].mean(dim=1, keepdim=True)
        t_all = base + best_off[nearest].unsqueeze(1)
    else:
        raise ValueError("solve_translation must be 'centroid' or 'search', got " + str(solve_translation))
    residuals = _sample(pts_rot[:, idx_loss] + t_all)  # (A,M)
    shape_loss = torch.sqrt(residuals**2 + 1e-6).mean(dim=1)  # (A,), SDF pixels
    objective = shape_loss
    if theta_prior is not None and prior_weight is not None:
        tp = torch.as_tensor(theta_prior, dtype=dtype, device=device).reshape(-1)  # (P,)
        pw = torch.as_tensor(prior_weight, dtype=dtype, device=device).reshape(-1)  # (P,)
        if pw.numel() > 0 and float(pw.sum()) > 0.0:
            d = torch.atan2(
                torch.sin(thetas.unsqueeze(0) - tp.unsqueeze(1)), torch.cos(thetas.unsqueeze(0) - tp.unsqueeze(1))
            )
            objective = shape_loss + (pw.unsqueeze(1) * d**2).sum(dim=0)
    best = torch.argmin(objective)

    # Hold the prior angle unless the argmin's shape evidence (not the objective, which already favours the prior)
    # is better by the margin.
    if hold_margin is not None:
        margin_px = hold_margin * scale
        if not bool(shape_loss[best] + margin_px < shape_loss[0]):
            best = torch.zeros((), dtype=torch.long, device=device)

    theta = thetas[best].clone()
    t = t_all[best, 0].clone()

    final = torch.sqrt(
        _sample(
            (
                pts_measured[idx_loss]
                @ torch.stack(
                    [
                        torch.stack([torch.cos(theta), -torch.sin(theta)]),
                        torch.stack([torch.sin(theta), torch.cos(theta)]),
                    ]
                ).T
                + t
            ).unsqueeze(0)
        )[0]
        ** 2
        + 1e-6
    ).mean()
    return theta.detach(), t.detach(), (final / scale).detach()
