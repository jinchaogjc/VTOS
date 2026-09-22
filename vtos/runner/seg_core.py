"""baselines/seg_vtos_core.py — shared core for all VTOS-Seg baselines.

Single source of truth for the machinery every VTOS-Seg variant (v2, analyzer,
VFE, EFE) shares:
  - normalize_model_id : provider-aware model-ID prefixing (one helper)
  - PROMPT_HEADER      : the search prompt — NEUTRAL: describes tools, output
                         contract and the ranking objective, but does NOT
                         prescribe a strategy. The LLM freely chooses single
                         vs ensemble queries, thresholds, post-processing.
  - build_toolbox      : the seg toolbox (detect / segment / filter / nms),
                         optionally instrumented for EFE
  - exec_candidate     : execute one candidate program in a sandbox
  - score_candidate    : evaluate a candidate on train tasks → raw metrics
                         (dice_mask, miou_mask, per-task stats). It does NOT
                         collapse the two metrics into one number.
  - rank_history       : rank candidates by objective. Default "borda" =
                         Borda dual-rank over (dice_mask, miou_mask): rank by
                         each metric, sum the ranks. Scale-invariant — both
                         metrics weigh equally regardless of magnitude.
                         "miou" (miou_mask alone) and "linear" also selectable.
  - history_block      : ranked history rendering for the propose prompt
  - propose_program    : one explore/exploit proposal (LLM call) — shared
  - Candidate          : per-iteration record (raw metrics + feedback)
"""
from __future__ import annotations

from typing import Dict, List, Optional

SAM2_MODEL_ID = "facebook/sam2-hiera-large"
def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── toolbox (plain or instrumented for EFE) ──────────────────────────────────
def build_toolbox(dino, sam2, image_path: str, W: int, H: int,
                  exec_log: Optional[List[Dict]] = None,
                  gt_boxes: Optional[List] = None) -> Dict:
    """Build the seg toolbox exposed to LLM-emitted code.

    If exec_log is provided, every tool call appends a record to it (EFE mode).
    If gt_boxes is provided (ORACLE-VTOS diagnostic only), a `gt_bboxes()` tool
    is added that returns the ground-truth boxes — detection is "solved" so the
    search can isolate whether the mask tools add value over plain SAM 2.
    """
    from metrics.segmentation_metrics import mask_to_polygons

    def detect(text: str, threshold: float = 0.30):
        try:
            bbs_px, _, _ = dino.detect(image_path=image_path,
                                       text_query=text, box_threshold=threshold)
            out = []
            for bb in bbs_px:
                if len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append([max(0.0, x1 / W), max(0.0, y1 / H),
                            min(1.0, (x2 - x1) / W), min(1.0, (y2 - y1) / H)])
        except Exception:
            out = []
        if exec_log is not None:
            exec_log.append({"tool": "detect", "text": text,
                             "threshold": threshold, "n_out": len(out)})
        return out

    def segment(bboxes_xywh):
        polys = []
        try:
            if bboxes_xywh:
                xyxy_abs = []
                for b in bboxes_xywh:
                    if len(b) != 4:
                        continue
                    x, y, w, h = b
                    xyxy_abs.append([x * W, y * H, (x + w) * W, (y + h) * H])
                if xyxy_abs:
                    masks = sam2.segment(image_path, xyxy_abs)
                    for m in masks:
                        polys.extend(mask_to_polygons(m))
        except Exception:
            polys = []
        if exec_log is not None:
            exec_log.append({"tool": "segment", "n_in": len(bboxes_xywh or []),
                             "n_out": len(polys)})
        return polys

    def filter_by_area(bboxes, min_norm_area=0.001, max_norm_area=0.9):
        kept = [b for b in bboxes
                if len(b) == 4 and (min_norm_area <= b[2] * b[3] <= max_norm_area)]
        if exec_log is not None:
            exec_log.append({"tool": "filter_by_area", "n_in": len(bboxes),
                             "n_out": len(kept)})
        return kept

    def nms(bboxes, iou_thresh=0.5):
        sorted_b = sorted(bboxes, key=lambda b: -(b[2] * b[3]))
        kept = []
        for b in sorted_b:
            if all(bbox_iou(b, k) < iou_thresh for k in kept):
                kept.append(b)
        if exec_log is not None:
            exec_log.append({"tool": "nms", "n_in": len(bboxes), "n_out": len(kept)})
        return kept

    # ── mask-level (segmentation) tools ─────────────────────────────────
    def _poly_area_norm(poly) -> float:
        """Normalized polygon area via the shoelace formula."""
        n = len(poly)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def filter_polygons_by_area(polygons, min_norm_area=0.0005, max_norm_area=0.95):
        """Drop polygons whose normalized area is outside [min, max]."""
        kept = [p for p in polygons
                if min_norm_area <= _poly_area_norm(p) <= max_norm_area]
        if exec_log is not None:
            exec_log.append({"tool": "filter_polygons_by_area",
                             "n_in": len(polygons), "n_out": len(kept)})
        return kept

    def clean_mask(polygons):
        """Morphological close+open on the rasterized mask: fill small holes,
        remove tiny specks. Returns cleaned polygons."""
        import cv2
        from metrics.segmentation_metrics import polygons_to_mask
        if not polygons:
            if exec_log is not None:
                exec_log.append({"tool": "clean_mask", "n_in": 0, "n_out": 0})
            return []
        mask = polygons_to_mask(polygons, W, H)
        k = max(3, int(round(0.006 * min(W, H))))
        if k % 2 == 0:
            k += 1
        k = min(k, 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        out = mask_to_polygons(m)
        if exec_log is not None:
            exec_log.append({"tool": "clean_mask", "n_in": len(polygons),
                             "n_out": len(out)})
        return out

    def merge_polygons(polygons, iou_thresh=0.5):
        """Merge polygons that overlap (polygon-IoU > iou_thresh) by unioning
        their rasterized masks; returns one polygon per merged group."""
        from metrics.segmentation_metrics import (polygon_to_mask,
                                                  polygons_to_mask, compute_pixel_iou)
        n = len(polygons)
        if n <= 1:
            if exec_log is not None:
                exec_log.append({"tool": "merge_polygons", "n_in": n, "n_out": n})
            return list(polygons)
        masks = [polygon_to_mask(p, W, H) for p in polygons]
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if compute_pixel_iou(masks[i], masks[j]) > iou_thresh:
                    parent[find(i)] = find(j)
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out = []
        for members in groups.values():
            if len(members) == 1:
                out.append(polygons[members[0]])
            else:
                union = polygons_to_mask([polygons[m] for m in members], W, H)
                out.extend(mask_to_polygons(union))
        if exec_log is not None:
            exec_log.append({"tool": "merge_polygons", "n_in": n, "n_out": len(out)})
        return out

    def segment_from_points(points_xy_norm):
        """SAM 2 with point prompts (each [x,y] normalized = one foreground
        point). Fallback when box detection misses. Returns polygons."""
        polys = []
        try:
            pts_abs = [[float(p[0]) * W, float(p[1]) * H]
                       for p in points_xy_norm if len(p) >= 2]
            if pts_abs:
                masks = sam2.segment_points(image_path, pts_abs)
                for m in masks:
                    polys.extend(mask_to_polygons(m))
        except Exception:
            polys = []
        if exec_log is not None:
            exec_log.append({"tool": "segment_from_points",
                             "n_in": len(points_xy_norm or []), "n_out": len(polys)})
        return polys

    def polygon_iou(poly_a, poly_b):
        """IoU between two polygons (rasterized). Lets code reason about masks."""
        from metrics.segmentation_metrics import compute_polygon_iou
        try:
            return compute_polygon_iou(poly_a, poly_b, width=256, height=256)
        except Exception:
            return 0.0

    tools = {"detect": detect, "segment": segment,
             "filter_by_area": filter_by_area, "nms": nms,
             "filter_polygons_by_area": filter_polygons_by_area,
             "clean_mask": clean_mask, "merge_polygons": merge_polygons,
             "segment_from_points": segment_from_points,
             "polygon_iou": polygon_iou}

    if gt_boxes is not None:
        def gt_bboxes():
            """ORACLE diagnostic: return ground-truth bboxes (normalized xywh)."""
            out = [list(b) for b in gt_boxes]
            if exec_log is not None:
                exec_log.append({"tool": "gt_bboxes", "n_out": len(out)})
            return out
        tools["gt_bboxes"] = gt_bboxes
    return tools


