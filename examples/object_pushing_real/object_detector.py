"""
Object Detection from a Depth Frame

The detector does not rely on the current belief about the object. It enumerates every blob in the frame that is
high enough above the table to be the object, fits the registered object template to each of them, and selects the
blob (and the template orientation) whose projected template overlaps it best. Orientation ambiguities between the
local minima of the fit are resolved by overlap, then by continuity with the previous angle.

The table height and object height constants are in the robot base frame and specific to the setup.

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .params import NUM_KEYPOINTS
from .util import SDF_PADDING_PX, fit_se2_to_sdf, make_signed_distance_transform, polygon_centroid

SCALING_FACTOR = 0.4  # image scaling of ObjectPushingSensor

# Base-frame heights [m]: top surface of the objects, and the cut separating them from the table.
OBJECT_Z = 0.157
Z_CUT = 0.140

# Blob size range [px] for candidates. Deliberately wide: it only filters candidates, the template overlap decides
# identity. Blobs below MIN_BLOB_PX are speckle.
AREA_PX = (600, 9000)
MIN_BLOB_PX = 150

# Minimum overlap of the projected template with a blob (IoU, and fraction of the blob covered) to accept it.
MIN_IOU = 0.45
MIN_COVERAGE = 0.37
# Mean contour residual [m] above which a fit is discarded.
SHAPE_MAX_RESIDUAL = 0.030
# Fewest keypoints that are not on the image border for a fit to be attempted.
MIN_VALID_KEYPOINTS = 4
N_ANGLES = 1440

# With fewer keypoints off the image border, the angle is not updated (the previous one is held).
MIN_VALID_FOR_ANGLE = 5

# Continuity: candidates within ORIENT_CONTINUITY_WINDOW of the previous angle win if their score reaches
# ORIENT_CONTINUITY_MARGIN of the best score. Otherwise the best candidate is taken, but only if its score is at
# least ESCAPE_MIN_IOU.
ORIENT_CONTINUITY_MARGIN = 0.85
ORIENT_CONTINUITY_WINDOW = np.radians(23.0)
ESCAPE_MIN_IOU = 0.70
SCORE_TIE_TOL = 0.95  # candidates within this fraction of the best score are tied

# Penalty on the fraction of the projected template that covers valid depth outside the blob (visible table),
# subtracted from the IoU score.
FREE_PENALTY = 1.25

# Minimum fraction of the (unclipped) projected template that must overlap the blob at the published pose.
MIN_VISIBLE_FRAC = 0.35

# Orientation hysteresis: a change larger than HYST_BIG_STEP is only accepted after HYST_CONFIRM consecutive frames
# that agree within HYST_AGREE.
HYST_CONFIRM = 5
HYST_BIG_STEP = np.radians(60.0)
HYST_AGREE = np.radians(30.0)

_PUSHER_MASK = None


@dataclass
class Candidate:
    """One blob in the frame."""

    area_px: int
    label: int
    z_base: float
    contour: np.ndarray  # (M,1,2) cv2 contour, scaled px
    keypoints_px: np.ndarray  # (NK,2) scaled px
    keypoints_CF: np.ndarray  # (NK,3) camera frame, metres
    on_border: np.ndarray  # (NK,) bool
    mask: np.ndarray  # (H,W) uint8, this blob's pixels
    fit: Optional["Fit"] = None
    rival_scores: Optional[list] = None  # [(theta, iou, coverage)] for every rival angle
    iou: float = 0.0
    coverage: float = 0.0
    theta_best: float = 0.0  # rival angle chosen by overlap and continuity


@dataclass
class Fit:
    theta: float
    t: np.ndarray  # (2,) template-frame translation
    residual: float  # metres, mean |SDF| over the non-border points
    loss_curve: np.ndarray  # (N_ANGLES,)
    rivals: List[Tuple[float, float]] = field(default_factory=list)  # local minima (theta, loss_ratio)


@dataclass
class Detection:
    keypoints_CF: np.ndarray  # (NK,3)
    valid_mask: np.ndarray  # (NK,) bool, keypoints not on the image border
    theta: float  # angle of the base -> template rotation
    centroid_base: np.ndarray  # (2,) object centre under the fitted pose
    residual: float
    n_candidates: int
    n_rivals: int
    decided_by: str  # "only" | "overlap" | "previous" | "held"
    area_px: int
    z_base: float
    n_valid: int = 0  # keypoints not on the image border
    theta_reliable: bool = True  # False: theta is the previous angle, not measured this frame
    keypoints_px: Optional[np.ndarray] = None  # (NK,2) scaled px, raw contour keypoints
    template_px: Optional[np.ndarray] = None  # (NK,2) scaled px, fitted template projected into the image
    contour: Optional[np.ndarray] = None  # (M,1,2) winning blob contour, scaled px
    mask: Optional[np.ndarray] = None  # (H,W) uint8 winning blob
    iou: float = 0.0
    coverage: float = 0.0


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def preprocess(depth_mm: np.ndarray, pusher_mask_path: Optional[str] = None) -> np.ndarray:
    """Zero depth edges and the pusher region; returns depth in mm. The edge band is what separates the object from
    the table."""
    global _PUSHER_MASK
    if _PUSHER_MASK is None and pusher_mask_path is None:
        pusher_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pusher_mask.png")
    if _PUSHER_MASK is None and os.path.exists(pusher_mask_path):
        _PUSHER_MASK = cv2.imread(pusher_mask_path, cv2.IMREAD_GRAYSCALE)
    d = depth_mm.astype(np.float32) / 1000.0
    filled = cv2.medianBlur(d, 5)
    filled = cv2.bilateralFilter(filled, 5, 10, 2.5, borderType=cv2.BORDER_CONSTANT)
    dx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    d[cv2.magnitude(dx, dy) > 0.017] = 0
    if _PUSHER_MASK is not None and _PUSHER_MASK.shape == d.shape:
        d[_PUSHER_MASK > 0] = 0
    return (d * 1000.0).astype(np.float32)


def _keypoints_from_contour(contour: np.ndarray, n: int = NUM_KEYPOINTS) -> Optional[np.ndarray]:
    """`n` points around a contour, corners preserved. Returns (n,2) or None."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)[:, 0, :]
    if len(approx) < 3:
        return None
    idx = np.linspace(0, len(approx) - 1, n).round().astype(int)
    return approx[idx].astype(float)


_HGAIN_KEY = None
_HGAIN = None


def height_map(dep_mm: np.ndarray, P: np.ndarray, H_bc: np.ndarray) -> np.ndarray:
    """Base-frame height of every pixel; -9 where depth is invalid.

    The height is affine in depth, z_base = z * G(u, v) + H_bc[2,3], and G only depends on the intrinsics and the
    camera tilt, so it is cached.
    """
    global _HGAIN_KEY, _HGAIN
    h, w = dep_mm.shape
    key = (h, w, P[0, 0], P[1, 1], P[0, 2], P[1, 2], H_bc[2, 0], H_bc[2, 1], H_bc[2, 2])
    if _HGAIN_KEY != key:
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        uv = np.stack([uu / SCALING_FACTOR, vv / SCALING_FACTOR, np.ones_like(uu)], -1)
        _HGAIN = (uv @ np.linalg.inv(P[:3, :3]).T @ H_bc[2, :3]).astype(np.float32)
        _HGAIN_KEY = key
    return np.where(dep_mm > 0, (dep_mm / 1000.0) * _HGAIN + H_bc[2, 3], -9.0).astype(np.float32)


def candidate_blobs(
    dep_mm: np.ndarray,
    P: np.ndarray,
    H_bc: np.ndarray,
    z_cut: float = Z_CUT,
    area_px: Tuple[int, int] = AREA_PX,
    edge_band: int = 10,
) -> List[Candidate]:
    """Every blob that could be the object: pixels above `z_cut` in the base frame, grouped into components within
    the area range."""
    P_inv = np.linalg.inv(P[:3, :3])
    # Gating on height per pixel before labelling also separates rounded objects, whose edge band does not close.
    # The opening removes the speckle this produces at depth discontinuities.
    seg = ((dep_mm > 0) & (height_map(dep_mm, P, H_bc) > z_cut)).astype(np.uint8)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(seg, 4)
    h, w = dep_mm.shape
    out: List[Candidate] = []
    for i in range(1, n_lab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < MIN_BLOB_PX or not (area_px[0] <= a <= area_px[1]):
            continue
        m = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(m)
        z = dep_mm[ys, xs] / 1000.0
        good = z > 0
        if good.sum() < 30:
            continue
        # base-frame height of the blob, from a subsample
        sel = np.linspace(0, good.sum() - 1, min(300, int(good.sum()))).astype(int)
        gi = np.nonzero(good)[0][sel]
        uv = np.stack([xs[gi] / SCALING_FACTOR, ys[gi] / SCALING_FACTOR, np.ones(len(gi))], 1)
        cf = (P_inv @ uv.T).T * z[gi][:, None]
        z_base = float(np.median((H_bc @ np.concatenate([cf, np.ones((len(gi), 1))], 1).T).T[:, 2]))
        if z_base <= z_cut:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        kp = _keypoints_from_contour(c)
        if kp is None:
            continue
        # lift the keypoints with their own depths, filling holes with the median
        zk = np.array([dep_mm[min(max(int(p[1]), 0), h - 1), min(max(int(p[0]), 0), w - 1)] / 1000.0 for p in kp])
        if np.all(zk <= 0):
            continue
        zk[zk <= 0] = np.median(zk[zk > 0])
        rays = (P_inv @ np.concatenate([kp / SCALING_FACTOR, np.ones((len(kp), 1))], 1).T).T
        # keypoints on the image border lie on the frame's cut, not on the object
        on_border = (
            (kp[:, 0] < edge_band) | (kp[:, 1] < edge_band) | (kp[:, 0] > w - edge_band) | (kp[:, 1] > h - edge_band)
        )
        out.append(
            Candidate(
                area_px=a,
                label=i,
                z_base=z_base,
                contour=c,
                keypoints_px=kp,
                keypoints_CF=rays * zk[:, None],
                on_border=on_border,
                mask=m,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Overlap of the projected template with a blob
# --------------------------------------------------------------------------------------


def project_template(
    poly_centered: np.ndarray,
    theta: float,
    t: np.ndarray,
    centroid_meas: np.ndarray,
    P: np.ndarray,
    H_bc: np.ndarray,
    object_z: float = OBJECT_Z,
) -> Optional[np.ndarray]:
    """The template polygon under a fitted pose, placed at `object_z` and projected into scaled image pixels.

    The fit maps a measured base point p to template coordinates as R(p - centroid) + t; this is its inverse.
    """
    c_, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c_, -s_], [s_, c_]])
    base = (R.T @ (poly_centered - t).T).T + centroid_meas
    xyz = np.concatenate([base, np.full((len(base), 1), object_z), np.ones((len(base), 1))], 1)
    cam = (np.linalg.inv(H_bc) @ xyz.T).T[:, :3]
    if np.any(cam[:, 2] <= 1e-6):
        return None
    uv = (P[:3, :3] @ cam.T).T
    return uv[:, :2] / uv[:, 2:3] * SCALING_FACTOR


class OrientationHysteresis:
    """Holds the published angle through large changes until they have been confirmed on several frames."""

    def __init__(self):
        self.current = None
        self._pending = None
        self._count = 0

    def update(self, theta: float) -> float:
        """Feed the per-frame proposal, get the angle to publish."""
        if theta is None or not np.isfinite(theta):
            return self.current
        if self.current is None:
            self.current = float(theta)
            return self.current
        if abs(_wrap(theta - self.current)) > HYST_BIG_STEP:
            if self._pending is not None and abs(_wrap(theta - self._pending)) < HYST_AGREE:
                self._count += 1
            else:
                self._pending, self._count = float(theta), 1
            if self._count >= HYST_CONFIRM:
                self.current, self._pending, self._count = float(theta), None, 0
        else:
            self.current, self._pending, self._count = float(theta), None, 0
        return self.current


def overlap(
    template_uv: np.ndarray, blob_mask: np.ndarray, free_mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """(score, coverage) of a projected template on a blob.

    The score is the IoU, minus FREE_PENALTY times the fraction of the template lying on `free_mask` (valid depth
    that is not the blob). Coverage is the fraction of the blob the template explains. Template area outside the
    image is not counted.
    """
    tpl = np.zeros(blob_mask.shape, np.uint8)
    cv2.fillPoly(tpl, [np.round(template_uv).astype(np.int32)], 1)
    b = (blob_mask > 0).astype(np.uint8)
    inter = int(np.count_nonzero(tpl & b))
    union = int(np.count_nonzero(tpl | b))
    if union == 0:
        return 0.0, 0.0
    iou, cov = inter / union, inter / max(int(np.count_nonzero(b)), 1)
    if free_mask is not None:
        tot = int(np.count_nonzero(tpl))
        if tot:
            iou -= FREE_PENALTY * (int(np.count_nonzero(tpl & free_mask)) / tot)
    return iou, cov


# --------------------------------------------------------------------------------------
# Template fit
# --------------------------------------------------------------------------------------


def _sample_border(ref: torch.Tensor, pts: torch.Tensor, scale, min_xy) -> torch.Tensor:
    """SDF lookup that clamps points off the grid to the border (instead of reading them as on the contour)."""
    g = (scale * (pts - min_xy) + SDF_PADDING_PX) / (ref.shape[0] - 1) * 2 - 1
    return torch.nn.functional.grid_sample(
        ref.unsqueeze(0).unsqueeze(0).to(pts.dtype),
        g.unsqueeze(0).unsqueeze(0),
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    ).squeeze()


def fit_candidate(
    kp_CF: np.ndarray, on_border: np.ndarray, H_bc: np.ndarray, sdf_pack, n_angles: int = N_ANGLES
) -> Optional[Fit]:
    """Fit the registered template to one candidate's keypoints and return all local minima over the angle.

    The translation is searched instead of taken from centroid alignment, and border keypoints are excluded from
    the translation, because the centroid of a truncated contour is not the object's centre.
    """
    ref, scale, min_xy = sdf_pack[0], sdf_pack[1], sdf_pack[2]
    kp = torch.as_tensor(kp_CF, dtype=torch.double)
    finite = torch.isfinite(kp[:, 0])
    valid = finite & torch.as_tensor(~on_border, dtype=torch.bool)
    if int(valid.sum()) < MIN_VALID_KEYPOINTS or int(finite.sum()) < 3:
        return None
    Hb = torch.as_tensor(H_bc, dtype=torch.double)
    n = int(finite.sum())
    w = (Hb @ torch.cat([kp[finite], torch.ones(n, 1, dtype=torch.double)], 1).T).T[:, :2]
    pts = w - polygon_centroid(w)
    try:
        theta, t, _ = fit_se2_to_sdf(
            pts,
            ref,
            scale,
            min_xy,
            n_angles=n_angles,
            solve_translation="search",
            point_mask=valid[finite],
            mask_scope="translation",
        )
    except RuntimeError:
        return None

    # the objective over all angles at the fitted translation, so rivals come from the same surface as the argmin
    ang = torch.arange(n_angles, dtype=torch.double) * (2 * np.pi / n_angles)
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (A,2,2)
    rot = torch.einsum("aij,nj->ani", R, pts) + t
    keep = valid[finite]
    vals = _sample_border(ref, rot[:, keep].reshape(-1, 2), scale, min_xy).reshape(n_angles, -1)
    loss = torch.sqrt(vals**2 + 1e-6).mean(dim=1) / scale
    loss_np = loss.numpy()

    g = int(np.argmin(loss_np))
    lo, hi = np.roll(loss_np, 1), np.roll(loss_np, -1)
    local = np.nonzero((loss_np < lo) & (loss_np < hi))[0]
    rivals = [(float(i * 2 * np.pi / n_angles), float(loss_np[i] / loss_np[g])) for i in local]
    rivals.sort(key=lambda r: r[1])
    return Fit(
        theta=float(ang[g]), t=t.numpy().reshape(-1)[:2], residual=float(loss_np[g]), loss_curve=loss_np, rivals=rivals
    )


# --------------------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------------------


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _choose_angle(scores, gbest, prev_theta):
    """Continuity rule over the scored rivals (theta, score, coverage); returns one of them."""
    if prev_theta is None or not np.isfinite(prev_theta):
        return gbest
    near = [x for x in scores if abs(_wrap(x[0] - prev_theta)) <= ORIENT_CONTINUITY_WINDOW]
    if not near:
        return gbest
    nbest = max(near, key=lambda x: x[1])
    if nbest[1] >= ORIENT_CONTINUITY_MARGIN * gbest[1]:
        return nbest
    # only jump away from the previous angle if the alternative is itself a good fit
    if gbest[1] < ESCAPE_MIN_IOU:
        return nbest
    return gbest


def detect(
    depth_mm: np.ndarray,
    P: np.ndarray,
    H_bc: np.ndarray,
    sdf_pack,
    prev_theta: Optional[float] = None,
    max_residual: float = SHAPE_MAX_RESIDUAL,
    z_cut: float = Z_CUT,
    already_preprocessed: bool = False,
) -> Optional[Detection]:
    """Detect the object in a depth frame. Returns None if no candidate fits well enough.

    `depth_mm` is preprocessed with `preprocess` unless `already_preprocessed`.
    """
    poly = sdf_pack[3]
    dep = depth_mm if already_preprocessed else preprocess(depth_mm)
    cands = candidate_blobs(dep, P, H_bc, z_cut=z_cut)
    if not cands:
        return None
    Hb = torch.as_tensor(H_bc, dtype=torch.double)

    def meas_centroid(c: Candidate) -> np.ndarray:
        kp = torch.as_tensor(c.keypoints_CF, dtype=torch.double)
        f = torch.isfinite(kp[:, 0])
        w = (Hb @ torch.cat([kp[f], torch.ones(int(f.sum()), 1, dtype=torch.double)], 1).T).T[:, :2]
        return polygon_centroid(w).numpy()

    scored = []
    for c in cands:
        c.fit = fit_candidate(c.keypoints_CF, c.on_border, H_bc, sdf_pack)
        if c.fit is None or c.fit.residual > max_residual:
            continue
        cen = meas_centroid(c)
        # Score every rival angle by overlap: the contour loss cannot distinguish a flipped pose, overlap can.
        free = ((dep > 0) & (c.mask == 0)).astype(np.uint8)
        scores = []
        for th, _ratio in c.fit.rivals or [(c.fit.theta, 1.0)]:
            uv = project_template(poly, th, c.fit.t, cen, P, H_bc)
            if uv is None:
                continue
            iou, cov = overlap(uv, c.mask, free_mask=free)
            scores.append((th, iou, cov))
        if not scores:
            continue
        c.rival_scores = scores
        # The blob is judged by its best rival; the angle additionally prefers continuity.
        gbest = max(scores, key=lambda x: x[1])
        c.iou, c.coverage = gbest[1], gbest[2]
        c.theta_best = _choose_angle(scores, gbest, prev_theta)[0]
        if c.iou >= MIN_IOU and c.coverage >= MIN_COVERAGE:
            scored.append(c)
    if not scored:
        return None

    if len(scored) == 1:
        win, how = scored[0], "only"
    else:
        win, how = max(scored, key=lambda r: r.iou), "overlap"

    theta = win.theta_best
    decided = "overlap"
    if len(win.fit.rivals) > 1:
        # Break near-ties between rival angles within the set the continuity rule chose by the previous angle.
        scores = win.rival_scores or []
        gbest = max(scores, key=lambda x: x[1]) if scores else None
        chosen = _choose_angle(scores, gbest, prev_theta) if scores else None
        if chosen is not None and gbest is not None and chosen[0] != gbest[0]:
            pool = [x for x in scores if abs(_wrap(x[0] - prev_theta)) <= ORIENT_CONTINUITY_WINDOW]
        else:
            pool = scores
        ranked = sorted([(x[0], x[1]) for x in pool], key=lambda x: -x[1])
        if len(ranked) > 1 and ranked[1][1] >= SCORE_TIE_TOL * ranked[0][1]:
            tied = [th for th, sc in ranked if sc >= SCORE_TIE_TOL * ranked[0][1]]
            if prev_theta is not None and np.isfinite(prev_theta):
                theta = min(tied, key=lambda th: abs(_wrap(th - prev_theta)))
                decided = "previous"
    if how == "only" and decided == "overlap":
        decided = "only"

    # With too few keypoints off the image border the angle is not constrained; hold the previous one.
    n_valid = int(np.count_nonzero(~win.on_border))
    theta_reliable = n_valid >= MIN_VALID_FOR_ANGLE
    if not theta_reliable and prev_theta is not None and np.isfinite(prev_theta):
        theta = float(prev_theta)
        decided = "held"

    # object centre under the fitted pose (the raw contour centroid is biased when the contour is truncated)
    raw_c = meas_centroid(win)
    c_, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c_, -s_], [s_, c_]])
    centroid_base = raw_c - R.T @ win.fit.t

    # Fitted template in the image, not clipped: points outside the frame keep their correct rays.
    template_px = project_template(poly, theta, win.fit.t, raw_c, P, H_bc)

    # Re-check the pose that is actually published: enough of the template must be observed, and it must still
    # overlap the blob.
    if template_px is not None:
        tx, ty = template_px[:, 0], template_px[:, 1]
        full = 0.5 * abs(float(np.dot(tx, np.roll(ty, -1)) - np.dot(ty, np.roll(tx, -1))))
        if full > 1.0:
            tpl_img = np.zeros(win.mask.shape, np.uint8)
            cv2.fillPoly(tpl_img, [np.round(template_px).astype(np.int32)], 1)
            blob_img = (win.mask > 0).astype(np.uint8)
            seen = int(np.count_nonzero(tpl_img & blob_img))
            if seen / full < MIN_VISIBLE_FRAC:
                return None
            union = int(np.count_nonzero(tpl_img | blob_img))
            if union and seen / union < MIN_IOU:
                return None

    return Detection(
        keypoints_CF=win.keypoints_CF,
        valid_mask=~win.on_border,
        theta=float(theta),
        centroid_base=centroid_base,
        residual=win.fit.residual,
        n_candidates=len(cands),
        n_rivals=len(win.fit.rivals),
        decided_by=decided,
        area_px=win.area_px,
        z_base=win.z_base,
        n_valid=n_valid,
        theta_reliable=theta_reliable,
        keypoints_px=win.keypoints_px,
        template_px=template_px,
        contour=win.contour,
        mask=win.mask,
        iou=win.iou,
        coverage=win.coverage,
    )


def sdf_from_registered(kr: torch.Tensor):
    """(ref, scale, min_xy, poly_centered) for the registered template, on CPU in double precision.

    `poly_centered` is the template outline about its mean, in metres.
    """
    kr = torch.as_tensor(kr, dtype=torch.double)
    centred = kr - torch.mean(kr, dim=0, keepdim=True)
    ref, scale, min_xy = make_signed_distance_transform(centred, res=256, device=torch.device("cpu"))
    return ref.to(torch.double), scale, min_xy.to(torch.double), centred.numpy()
