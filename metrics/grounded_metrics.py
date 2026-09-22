"""
metrics/grounded_metrics.py
===========================
Canonical evaluation metrics for grounded counting (VTOS).

All matching uses **Hungarian (bipartite) assignment** — optimal one-to-one
matching between predicted and ground-truth boxes, following DETR
(Carion et al., ECCV 2020). This replaces the earlier greedy matcher and
the unmatched-max-IoU mIoU, which both inflated scores in adversarial cases.

Bbox format conventions:
  xyxy: [x1, y1, x2, y2]  normalized [0,1], top-left / bottom-right corners
  xywh: [x,  y,  w,  h]   normalized [0,1], top-left corner + width/height

Predictions come in **xyxy** (toolbox output). GT comes in **xywh**
(benchmark annotation format). Conversions are handled internally.
"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Internal helpers ──────────────────────────────────────────────────────────

def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = box[:4]
    return [x, y, x + w, y + h]


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes.

    Box areas are clamped to non-negative to guard against malformed boxes
    where x2<x1 or y2<y1 (which can happen with hallucinated VLM outputs).
    """
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _iou_matrix(pred_xyxy: list, gt_xyxy: list) -> np.ndarray:
    """Build N×M IoU matrix between predictions and GT (both xyxy)."""
    N, M = len(pred_xyxy), len(gt_xyxy)
    iou = np.zeros((N, M), dtype=np.float64)
    for i, p in enumerate(pred_xyxy):
        for j, g in enumerate(gt_xyxy):
            iou[i, j] = _iou_xyxy(p, g)
    return iou


def _symmetric_hit_matrix(pred_xyxy: list, gt_xywh: list) -> np.ndarray:
    """
    N×M binary matrix: hit[i, j] = 1 iff
        pred_center_i ∈ gt_box_j  OR  gt_center_j ∈ pred_box_i.
    """
    N, M = len(pred_xyxy), len(gt_xywh)
    hit = np.zeros((N, M), dtype=np.float64)
    pred_centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in pred_xyxy]
    gt_centers   = [(gx + gw / 2.0, gy + gh / 2.0) for gx, gy, gw, gh in gt_xywh]
    for i, (px, py) in enumerate(pred_centers):
        px1, py1, px2, py2 = pred_xyxy[i][:4]
        for j, (gx, gy, gw, gh) in enumerate(gt_xywh):
            gcx, gcy = gt_centers[j]
            pred_in_gt = (gx <= px <= gx + gw) and (gy <= py <= gy + gh)
            gt_in_pred = (px1 <= gcx <= px2) and (py1 <= gcy <= py2)
            if pred_in_gt or gt_in_pred:
                hit[i, j] = 1.0
    return hit


def _hungarian_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run Hungarian assignment on a (possibly rectangular) cost matrix.

    Returns row_indices, col_indices — pairs of matched (pred, gt) indices.
    For rectangular matrices the smaller dimension is fully matched; the
    excess rows/cols are simply unmatched and treated as FP/FN by the caller.
    """
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    row_idx, col_idx = linear_sum_assignment(cost)
    return row_idx, col_idx


# ── Public functions ──────────────────────────────────────────────────────────

def compute_point_f1_metrics(pred_bboxes_xyxy: list, gt_bboxes_xywh: list) -> dict:
    """
    Hungarian-matched point-in-box F1 (point_f1).

    A *hit* is a Hungarian-assigned pair (i, j) where
        pred_center_i ∈ gt_box_j   OR   gt_center_j ∈ pred_box_i.
    Pairs Hungarian assigns with no symmetric-hit are NOT counted.

    point_f1 = 2H / (N + M)   (= harmonic-mean of precision and recall.)

    Args:
        pred_bboxes_xyxy: list of [x1,y1,x2,y2] normalized
        gt_bboxes_xywh:  list of [x,y,w,h] normalized

    Returns:
        dict: {hits, misses, false_pos, precision, recall, point_f1, grounded_score}
    """
    N, M = len(pred_bboxes_xyxy), len(gt_bboxes_xywh)

    # Edge cases — degenerate empties
    if N == 0 and M == 0:
        return {"hits": 0, "misses": 0, "false_pos": 0,
                "precision": 1.0, "recall": 1.0, "point_f1": 1.0, "grounded_score": 1.0}
    if N == 0 or M == 0:
        return {"hits": 0, "misses": M, "false_pos": N,
                "precision": 0.0, "recall": 0.0, "point_f1": 0.0, "grounded_score": 0.0}

    # Cost = 0 for symmetric-hit pairs, 1 otherwise. Hungarian prefers cost-0 pairs.
    # To break ties (e.g., one pred could hit multiple GTs), add a tiny tiebreak
    # by negative distance so closer centers are preferred at equal hit cost.
    hit = _symmetric_hit_matrix(pred_bboxes_xyxy, gt_bboxes_xywh)
    pred_centers = np.array([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                             for b in pred_bboxes_xyxy])
    gt_centers   = np.array([(gx + gw / 2.0, gy + gh / 2.0)
                             for gx, gy, gw, gh in gt_bboxes_xywh])
    # Pairwise center distances (used only as tie-breaker, scaled to << 1)
    diff = pred_centers[:, None, :] - gt_centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    # Normalize distance to [0, 0.1] so it never overrides the hit-vs-miss gap of 1.0
    if dist.max() > 0:
        dist = 0.1 * dist / dist.max()
    cost = (1.0 - hit) + dist  # in [0, 0.1] for hits, [1, 1.1] for misses

    row_idx, col_idx = _hungarian_match(cost)

    # Count only pairs that were actual symmetric hits
    hits = int(sum(1 for r, c in zip(row_idx, col_idx) if hit[r, c] > 0))

    misses    = M - hits
    false_pos = N - hits
    precision = hits / max(1, N)
    recall    = hits / max(1, M)
    point_f1       = (2 * hits) / max(1, N + M)
    grounded  = hits / max(N, M)

    return {
        "hits": hits, "misses": misses, "false_pos": false_pos,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "point_f1": round(point_f1, 4), "grounded_score": round(grounded, 4),
    }


def compute_f1_iou(pred_bboxes_xyxy: list, gt_bboxes_xywh: list,
                   iou_threshold: float = 0.5) -> dict:
    """
    Standard detection-style F1 at IoU≥`iou_threshold` (PASCAL VOC convention).

    Hit criterion:
        A (pred_i, gt_j) Hungarian-assigned pair is a TRUE POSITIVE iff
        IoU(pred_i, gt_j) ≥ iou_threshold.

    Pairs Hungarian assigns with IoU below threshold are counted as FP/FN.

    F1 = 2·TP / (TP+FP + TP+FN) = 2·TP / (N_pred + N_gt)
       = harmonic mean of precision and recall (i.e., mathematically identical
       to "point_f1", just with a stricter IoU-based hit criterion instead of
       symmetric point-in-bbox).

    Args:
        pred_bboxes_xyxy: list of [x1,y1,x2,y2] normalized
        gt_bboxes_xywh:   list of [x,y,w,h] normalized
        iou_threshold:    minimum IoU for a TP (default 0.5)

    Returns:
        dict: {hits, misses, false_pos, precision, recall, f1, iou_threshold}
    """
    N, M = len(pred_bboxes_xyxy), len(gt_bboxes_xywh)

    # Edge cases
    if N == 0 and M == 0:
        return {"hits": 0, "misses": 0, "false_pos": 0,
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "iou_threshold": iou_threshold}
    if N == 0 or M == 0:
        return {"hits": 0, "misses": M, "false_pos": N,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "iou_threshold": iou_threshold}

    # Build IoU matrix and Hungarian-match on cost = 1 - IoU (DETR-style)
    gt_xyxy = [_xywh_to_xyxy(b) for b in gt_bboxes_xywh]
    iou = _iou_matrix(pred_bboxes_xyxy, gt_xyxy)
    cost = 1.0 - iou
    row_idx, col_idx = _hungarian_match(cost)

    # TP = matched pairs with IoU >= threshold
    hits = int(sum(1 for r, c in zip(row_idx, col_idx)
                   if iou[r, c] >= iou_threshold))

    misses    = M - hits
    false_pos = N - hits
    precision = hits / max(1, N)
    recall    = hits / max(1, M)
    f1        = (2 * hits) / max(1, N + M)

    return {
        "hits": hits, "misses": misses, "false_pos": false_pos,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou_threshold": iou_threshold,
    }


def compute_miou(pred_bboxes_xyxy: list, gt_bboxes_xywh: list) -> float:
    """
    Hungarian-matched mean IoU.

    Builds an N×M IoU matrix, runs Hungarian on cost = 1 − IoU (DETR-style),
    and returns:
        mIoU = sum(IoU over matched pairs) / max(N, M)

    The `max(N, M)` denominator penalises count mismatches: unmatched preds
    (FPs) and unmatched GTs (FNs) implicitly contribute IoU=0.

    Args:
        pred_bboxes_xyxy: list of [x1,y1,x2,y2] normalized
        gt_bboxes_xywh:  list of [x,y,w,h] normalized

    Returns:
        float: mean IoU in [0, 1].
          - 1.0 if both lists empty (trivial perfect match)
          - 0.0 if exactly one is empty
    """
    N, M = len(pred_bboxes_xyxy), len(gt_bboxes_xywh)
    if N == 0 and M == 0:
        return 1.0
    if N == 0 or M == 0:
        return 0.0

    gt_xyxy = [_xywh_to_xyxy(b) for b in gt_bboxes_xywh]
    iou = _iou_matrix(pred_bboxes_xyxy, gt_xyxy)
    cost = 1.0 - iou
    row_idx, col_idx = _hungarian_match(cost)
    matched_iou_sum = float(iou[row_idx, col_idx].sum())
    return round(matched_iou_sum / max(N, M), 4)


def compute_count_metrics(pred_count: int, gt_count: int) -> dict:
    """
    Per-image count error metrics.

    Returns:
        dict: {ae, squared_error, bias}
          ae             = |pred - gt|
          squared_error  = (pred - gt)^2  (use for RMSE aggregation)
          bias           = pred - gt      (signed; positive = over-counting)
    """
    diff = pred_count - gt_count
    return {
        "ae": abs(diff),
        "squared_error": diff ** 2,
        "bias": diff,
    }


def aggregate_metrics(per_image_list: list) -> dict:
    """
    Aggregate per-image metrics into overall and per-density summaries.

    Precision/Recall are **macro-averaged** across images (each image gets
    equal weight, regardless of object count). MAE/RMSE/Bias are also
    macro-averaged. Per-image point_f1/mIoU values are kept at full precision
    here; rounding is applied only at the summary layer.

    Args:
        per_image_list: list of dicts, each with keys:
            pred_bboxes  (list[xyxy]), gt_bboxes (list[xywh]),
            pred_count   (int),        gt_count  (int),
            density_tier (str)         one of `vtos.constants.TIER_ORDER`

    Returns:
        {
          overall: {point_f1, mIoU, MAE, RMSE, Bias, Precision, Recall, n},
          by_density: { <tier>: {...}, ... },   # tiers from vtos.constants.TIER_ORDER
        }
    """
    # Density tiers are the single source of truth in vtos.constants — no hard-coding.
    from vtos.constants import TIER_ORDER

    buckets: dict[str, list] = {t: [] for t in TIER_ORDER}
    buckets["_all"] = []

    for img in per_image_list:
        point_f1_d  = compute_point_f1_metrics(img["pred_bboxes"], img["gt_bboxes"])
        miou   = compute_miou(img["pred_bboxes"], img["gt_bboxes"])
        cnt_d  = compute_count_metrics(img["pred_count"], img["gt_count"])
        row = {**point_f1_d, "miou": miou, **cnt_d}
        tier = img.get("density_tier", "unknown")
        if tier in buckets:
            buckets[tier].append(row)
        buckets["_all"].append(row)

    def _summarise(rows):
        if not rows:
            return {"point_f1": None, "mIoU": None, "MAE": None,
                    "RMSE": None, "Bias": None, "Precision": None, "Recall": None, "n": 0}
        n = len(rows)
        return {
            "point_f1":       round(sum(r["point_f1"]             for r in rows) / n, 4),
            "mIoU":      round(sum(r["miou"]            for r in rows) / n, 4),
            "MAE":       round(sum(r["ae"]              for r in rows) / n, 2),
            "RMSE":      round(sqrt(sum(r["squared_error"] for r in rows) / n), 2),
            "Bias":      round(sum(r["bias"]            for r in rows) / n, 2),
            "Precision": round(sum(r["precision"]       for r in rows) / n, 4),
            "Recall":    round(sum(r["recall"]          for r in rows) / n, 4),
            "n": n,
        }

    return {
        "overall":    _summarise(buckets["_all"]),
        "by_density": {tier: _summarise(buckets[tier]) for tier in TIER_ORDER},
    }
