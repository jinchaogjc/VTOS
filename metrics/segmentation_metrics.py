"""
metrics/segmentation_metrics.py
================================

Canonical evaluation metrics for grounded segmentation (PlantSeg-OOD).

Mirrors the design of `grounded_metrics.py` (the counting metrics module):
  - Hungarian one-to-one matching for instance-level F1
  - Pixel-level mIoU/Dice as semantic-level metrics
  - Symmetric, well-defined edge cases

Coordinate conventions (matches grounded_metrics.py):
  bboxes:   xywh normalized [0,1]  (e.g., [x, y, w, h])
  polygons: list of [x, y] vertices, normalized [0,1]  (one polygon = one instance)

Key functions:
  - polygons_to_mask(polygons, width, height) → np.uint8 binary 2D mask (image-level)
  - polygon_to_mask(polygon, width, height)   → np.uint8 binary 2D mask (single polygon)
  - bboxes_to_mask(bboxes_xywh, width, height) → np.uint8 binary 2D mask (axis-aligned rects)
  - compute_pixel_iou(pred_mask, gt_mask) → float in [0,1]
  - compute_dice(pred_mask, gt_mask)      → float in [0,1]
  - compute_pixel_accuracy(pred_mask, gt_mask) → float in [0,1]
  - compute_polygon_iou(poly_a, poly_b, width, height) → float
  - compute_seg_f1_iou(pred_polys, gt_polys, iou_threshold=0.5, image_size=...) →
      {f1, precision, recall, hits, misses, false_pos}
      (analog to grounded_metrics.compute_f1_iou but over MASKS)
  - compute_seg_metrics(pred_polys, gt_polys, image_size, iou_threshold=0.5) →
      one-call wrapper returning all metrics in a dict
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ────────────────────────────────────────────────────────────────────────────
# Rasterization: polygons / bboxes → binary masks
# ────────────────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon: Sequence[Sequence[float]],
                    width: int, height: int) -> np.ndarray:
    """Rasterize a single polygon to a binary 2D mask of shape (H, W).

    Args:
        polygon: list of [x, y] normalized [0,1] vertices.
        width:   target mask width  in pixels.
        height:  target mask height in pixels.

    Returns:
        np.uint8 array of shape (height, width), values in {0, 1}.
    """
    from PIL import Image, ImageDraw
    if not polygon or len(polygon) < 3:
        return np.zeros((height, width), dtype=np.uint8)
    img = Image.new("L", (width, height), 0)
    abs_pts = [(float(p[0]) * width, float(p[1]) * height) for p in polygon]
    ImageDraw.Draw(img).polygon(abs_pts, fill=1, outline=1)
    return np.array(img, dtype=np.uint8)


def polygons_to_mask(polygons: Sequence[Sequence[Sequence[float]]],
                     width: int, height: int) -> np.ndarray:
    """Rasterize a LIST of polygons (union) to a single binary 2D mask.

    Used for IMAGE-LEVEL (semantic) IoU/Dice — "all disease pixels" vs "all
    predicted disease pixels". Pixels covered by ANY polygon are 1.
    """
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        mask |= polygon_to_mask(poly, width, height)
    return mask


def mask_to_polygons(mask: np.ndarray,
                     min_area_px: int = 16,
                     simplify_epsilon: float = 1.5) -> List[List[List[float]]]:
    """Extract polygon vertex lists from a binary 2D mask, normalized to [0,1].

    Bridges SAM-style raster mask outputs into the polygon-based metric framework.
    Each connected component becomes one polygon (its outer contour); tiny
    components below ``min_area_px`` are dropped to suppress noise.

    Args:
        mask: 2D array, foreground = nonzero.
        min_area_px: drop contours with area smaller than this.
        simplify_epsilon: Douglas-Peucker tolerance in pixels (0 = no simplify).

    Returns:
        List of polygons, each a list of [x, y] in normalized [0,1] coords.
    """
    import cv2
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    H, W = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: List[List[List[float]]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        if simplify_epsilon > 0 and len(c) > 4:
            c = cv2.approxPolyDP(c, simplify_epsilon, closed=True)
        verts = c.reshape(-1, 2)
        if len(verts) < 3:
            continue
        poly = [[float(x) / W, float(y) / H] for x, y in verts]
        polys.append(poly)
    return polys


def bboxes_to_mask(bboxes_xywh: Sequence[Sequence[float]],
                   width: int, height: int) -> np.ndarray:
    """Rasterize bboxes (xywh normalized) as axis-aligned rectangles.

    Useful for evaluating bbox-based methods (DINO, VTOS) on a segmentation task —
    treats each bbox as a coarse rectangular mask. This is a LOWER BOUND on
    segmentation quality (real segmentations are tighter than bboxes).
    """
    if not bboxes_xywh:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in bboxes_xywh:
        if len(b) < 4:
            continue
        x, y, w, h = b[:4]
        x1 = max(0, int(round(x * width)))
        y1 = max(0, int(round(y * height)))
        x2 = min(width,  int(round((x + w) * width)))
        y2 = min(height, int(round((y + h) * height)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


# ────────────────────────────────────────────────────────────────────────────
# Pixel-level metrics (image-level / semantic)
# ────────────────────────────────────────────────────────────────────────────

def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU of two binary masks (1 = foreground)."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        # both masks empty → perfect match
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return inter / union


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient = 2*|A∩B| / (|A| + |B|)."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    a = int(pred.sum()); b = int(gt.sum())
    if a + b == 0:
        return 1.0
    inter = int(np.logical_and(pred, gt).sum())
    return (2.0 * inter) / (a + b)


def compute_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Fraction of pixels correctly classified (both fg and bg)."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    if pred.size == 0:
        return 1.0
    return float((pred == gt).sum()) / float(pred.size)


# ────────────────────────────────────────────────────────────────────────────
# Polygon-level IoU + Hungarian-matched instance F1
# ────────────────────────────────────────────────────────────────────────────

def compute_polygon_iou(poly_a: Sequence[Sequence[float]],
                        poly_b: Sequence[Sequence[float]],
                        width: int = 256, height: int = 256) -> float:
    """IoU between two polygons via rasterization at the given resolution.

    Resolution 256x256 is a good speed/accuracy trade-off; finer doesn't
    materially change ranking, but is 4x slower.
    """
    if not poly_a or not poly_b:
        return 0.0
    ma = polygon_to_mask(poly_a, width, height)
    mb = polygon_to_mask(poly_b, width, height)
    return compute_pixel_iou(ma, mb)


def _polygon_iou_matrix(pred_polys: List, gt_polys: List,
                        raster_size: int = 256) -> np.ndarray:
    """Build N×M polygon-IoU matrix by rasterizing each polygon ONCE.

    Caches rasterizations for speed when both sets have multiple polygons.
    """
    N, M = len(pred_polys), len(gt_polys)
    iou = np.zeros((N, M), dtype=np.float64)
    if N == 0 or M == 0:
        return iou
    # Cache rasters
    pred_masks = [polygon_to_mask(p, raster_size, raster_size) for p in pred_polys]
    gt_masks   = [polygon_to_mask(g, raster_size, raster_size) for g in gt_polys]
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou[i, j] = compute_pixel_iou(pm, gm)
    return iou


def compute_seg_f1_iou(pred_polys: List[Sequence[Sequence[float]]],
                       gt_polys:   List[Sequence[Sequence[float]]],
                       iou_threshold: float = 0.5,
                       raster_size: int = 256) -> dict:
    """Instance-level F1 at polygon-IoU ≥ threshold (PASCAL VOC convention).

    Hungarian one-to-one matching minimises (1 - IoU); a TP is a matched pair
    with IoU ≥ iou_threshold. Direct parallel to grounded_metrics.compute_f1_iou
    but over polygon masks rather than axis-aligned bboxes.

    Args:
        pred_polys: predicted polygons (list of polygon, each polygon a list of [x,y]).
        gt_polys:   ground-truth polygons (same format).
        iou_threshold: IoU above which a match is a TP (default 0.5).
        raster_size: resolution for rasterization (default 256 — good trade-off).

    Returns:
        {hits, misses, false_pos, precision, recall, f1, iou_threshold}
    """
    N, M = len(pred_polys), len(gt_polys)
    if N == 0 and M == 0:
        return {"hits": 0, "misses": 0, "false_pos": 0,
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "iou_threshold": iou_threshold}
    if N == 0 or M == 0:
        return {"hits": 0, "misses": M, "false_pos": N,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "iou_threshold": iou_threshold}

    iou = _polygon_iou_matrix(pred_polys, gt_polys, raster_size=raster_size)
    cost = 1.0 - iou
    row_idx, col_idx = linear_sum_assignment(cost)
    hits = int(sum(1 for r, c in zip(row_idx, col_idx) if iou[r, c] >= iou_threshold))
    misses    = M - hits
    false_pos = N - hits
    precision = hits / max(1, N)
    recall    = hits / max(1, M)
    f1        = (2 * hits) / max(1, N + M)
    return {
        "hits":      hits,
        "misses":    misses,
        "false_pos": false_pos,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "iou_threshold": iou_threshold,
    }


# ────────────────────────────────────────────────────────────────────────────
# Core Grounded Segmentation Metrics for PlantSeg (6-Metric System)
# ────────────────────────────────────────────────────────────────────────────

def compute_bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection-over-Union (IoU) of two normalized xywh bboxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1, y1, w1, h1 = box_a[:4]
    x2, y2, w2, h2 = box_b[:4]
    
    # Convert to xyxy
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area_a = w1 * h1
    area_b = w2 * h2
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    iou = inter_area / union_area
    return min(1.0, max(0.0, iou))


def calculate_instance_miou_hungarian(
    preds: List[Any],
    gts: List[Any],
    is_bbox: bool,
    image_size: Tuple[int, int] = (256, 256)
) -> float:
    """Calculate instance-level mean IoU (mIoU) using DETR-style Hungarian Matching.
    
    Args:
        preds: predicted bboxes (list of xywh) or predicted polygons (list of [x,y]).
        gts: ground-truth bboxes (list of xywh) or ground-truth polygons (list of [x,y]).
        is_bbox: True if evaluating bboxes, False if evaluating polygon masks.
        image_size: rasterization resolution for polygon masks.
        
    Returns:
        Instance-level Hungarian Mean IoU in range [0, 1].
    """
    N, M = len(preds), len(gts)
    if N == 0 and M == 0:
        return 1.0
    if N == 0 or M == 0:
        return 0.0
        
    # Build pairwise IoU matrix
    iou_matrix = np.zeros((N, M), dtype=np.float64)
    if is_bbox:
        for i, pb in enumerate(preds):
            for j, gb in enumerate(gts):
                iou_matrix[i, j] = compute_bbox_iou(pb, gb)
    else:
        iou_matrix = _polygon_iou_matrix(preds, gts, raster_size=image_size[0])
        
    # linear sum assignment maximizes total IoU -> minimize (1.0 - IoU)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    
    # Collect matching overlaps
    matched_ious = []
    matched_preds = set()
    matched_gts = set()
    for r, c in zip(row_idx, col_idx):
        matched_ious.append(iou_matrix[r, c])
        matched_preds.add(r)
        matched_gts.add(c)
        
    # Unmatched predictions and ground truths get 0.0 IoU penalty
    unmatched_preds_count = N - len(matched_preds)
    unmatched_gts_count = M - len(matched_gts)
    
    all_ious = matched_ious + [0.0] * (unmatched_preds_count + unmatched_gts_count)
    return round(float(np.mean(all_ious)), 4)


def calculate_pixel_precision_recall(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Tuple[float, float]:
    """Calculate image-level pixel precision and pixel recall.
    
    Args:
        pred_mask: binary uint8 prediction mask.
        gt_mask: binary uint8 ground-truth mask.
        
    Returns:
        (Pixel Precision, Pixel Recall)
    """
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, 1 - gt).sum())
    fn = int(np.logical_and(1 - pred, gt).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if gt.sum() == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if pred.sum() == 0 else 0.0
    
    return round(precision, 4), round(recall) if recall == 0.0 or recall == 1.0 else round(recall, 4)


def compute_plantseg_pack(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    pred_polys: List[List[List[float]]],
    gt_polys: List[List[List[float]]],
    image_size: Tuple[int, int] = (256, 256)
) -> dict:
    """One-call academic package to calculate all 6 target PlantSeg metrics.
    
    Metrics:
        1. Dice^bbox: Image-level bbox intersection Dice
        2. Dice^mask: Image-level polygon mask intersection Dice
        3. mIoU^bbox: Instance-level Hungarian matched bbox IoU
        4. mIoU^mask: Instance-level Hungarian matched polygon IoU
        5. Pixel P^mask: Image-level pixel precision
        6. Pixel R^mask: Image-level pixel recall
    """
    w, h = image_size
    
    # 1. Rasterize image-level masks
    pred_mask_bbox = bboxes_to_mask(pred_boxes, w, h)
    gt_mask_bbox = bboxes_to_mask(gt_boxes, w, h)
    
    pred_mask_poly = polygons_to_mask(pred_polys, w, h)
    gt_mask_poly = polygons_to_mask(gt_polys, w, h)
    
    # 2. Overlap Quality (lenient Dice, image-level)
    dice_bbox = compute_dice(pred_mask_bbox, gt_mask_bbox)
    dice_mask = compute_dice(pred_mask_poly, gt_mask_poly)
    
    # 3. Overlap Quality (strict mIoU, instance-level Hungarian)
    miou_bbox = calculate_instance_miou_hungarian(pred_boxes, gt_boxes, is_bbox=True, image_size=image_size)
    miou_mask = calculate_instance_miou_hungarian(pred_polys, gt_polys, is_bbox=False, image_size=image_size)
    
    # 4. Pixel Imbalance Diagnostics
    pixel_p, pixel_r = calculate_pixel_precision_recall(pred_mask_poly, gt_mask_poly)
    
    return {
        "dice_bbox": round(dice_bbox, 4),
        "dice_mask": round(dice_mask, 4),
        "miou_bbox": round(miou_bbox, 4),
        "miou_mask": round(miou_mask, 4),
        "pixel_precision": round(pixel_p, 4),
        "pixel_recall": round(pixel_r, 4)
    }

