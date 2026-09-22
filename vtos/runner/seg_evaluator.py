"""SegPersistentEvaluator — drop-in replacement for PersistentEvaluator when
the VTOS task is segmentation. Execs LLM-emitted code in the seg toolbox sandbox,
scores per-task via compute_plantseg_pack, returns aggregates compatible with
VisionSearchEngine."""
from __future__ import annotations
from statistics import mean
from typing import Dict, List, Optional, Tuple

from vtos.constants import seg_tier
from vtos.runner.seg_adapter import load_split, SEG_IMAGE_DIR


def _build_toolbox(dino, sam2, image_path: str, W: int, H: int) -> Dict:
    """Seg sandbox: the seg_core tools (detect, segment, filter_by_area, nms,
    filter_polygons_by_area, clean_mask, merge_polygons, segment_from_points)
    plus three CLIP tools from seg_toolbox (clip_select, plant_healthy_prompt,
    plant_disease_prompt). split='test' keeps the CLIP prompt helpers on their
    generic templates (no train-only disease descriptors)."""
    from vtos.runner.seg_core import build_toolbox as _build_seg_core
    from vtos.runner.seg_toolbox import build_toolbox as _build_seg_clip
    tools = _build_seg_core(dino, sam2, image_path, W, H)
    clip_extras = _build_seg_clip(sam2, image_path, W, H, split="test")
    for k in ("clip_select", "plant_healthy_prompt", "plant_disease_prompt"):
        if k in clip_extras:
            tools[k] = clip_extras[k]
    return tools


def _exec_candidate(dino, sam2, code: str, task: Dict, image_path: str) -> Tuple[List, List]:
    """Run one seg program in the sandbox; return (bboxes, polygons)."""
    import sys
    from PIL import Image as PILImage
    try:
        with PILImage.open(image_path) as img:
            W, H = img.size
    except Exception:
        return [], []

    toolbox = _build_toolbox(dino, sam2, image_path, W, H)
    ns: Dict = {**toolbox,
                "target": task.get("target_class", ""),
                "plant":  task.get("plant", ""),
                "__builtins__": {
                    "range": range, "len": len, "min": min, "max": max,
                    "sum": sum, "list": list, "tuple": tuple, "dict": dict,
                    "abs": abs, "round": round, "int": int, "float": float,
                    "True": True, "False": False, "None": None,
                }}
    try:
        exec(code, ns)
    except Exception as e:
        sys.stderr.write(f"    [exec-fail] {e}\n")
        return [], []
    fb = ns.get("final_bboxes", []) or []
    fp = ns.get("final_polygons", []) or []
    clean_fb = [list(b) for b in fb if isinstance(b, (list, tuple)) and len(b) == 4]
    clean_fp = [list(p) for p in fp if isinstance(p, (list, tuple)) and len(p) >= 3]
    return clean_fb, clean_fp


class SegPersistentEvaluator:
    """Scores a candidate program on a PlantSeg v2 split.

    Mirrors PersistentEvaluator's interface (.evaluate returns a dict with
    `score` + per-metric breakdown) but uses Dice^mask as the primary score
    and adds per-tier / per-plant aggregates.
    """

    def __init__(self, sam2_model_id: str = "facebook/sam2-hiera-large"):
        self.sam2_model_id = sam2_model_id
        self._dino = None
        self._sam2 = None

    def _ensure(self) -> None:
        if self._sam2 is not None:
            return
        from vtos.toolbox import GroundingDINOSkill, Sam2Skill
        self._dino = GroundingDINOSkill()
        self._sam2 = Sam2Skill(model_id=self.sam2_model_id)

    def evaluate(self, code: str, split: str = "train",
                 n_tasks: Optional[int] = None) -> Dict:
        self._ensure()

        from metrics.segmentation_metrics import compute_plantseg_pack
        from PIL import Image as PILImage

        tasks = load_split(split)
        if n_tasks is not None:
            tasks = tasks[:n_tasks]

        # polygons_to_mask used per task for signed_area_error. Same rasterizer
        # the metric pack uses internally — keeps the area numbers consistent.
        from metrics.segmentation_metrics import polygons_to_mask
        import numpy as np

        per_task = []
        for t in tasks:
            img_path = str(SEG_IMAGE_DIR / t["image_path"])
            pred_bb, pred_poly = _exec_candidate(self._dino, self._sam2, code, t, img_path)
            try:
                with PILImage.open(img_path) as img:
                    W, H = img.size
            except Exception:
                W, H = 256, 256
            m = compute_plantseg_pack(
                pred_boxes=pred_bb, gt_boxes=t["bounding_boxes"],
                pred_polys=pred_poly, gt_polys=t["segmentations"],
                image_size=(W, H),
            )
            # Signed errors. Direction convention:
            #   + ⇒ over-prediction (too many bboxes / too much masked area)
            #   - ⇒ under-prediction (missed lesions / masks too small)
            # signed_area_error is normalized to image fraction so it's
            # comparable across resolutions; same scale as `mask_ratio`.
            try:
                pred_mask = polygons_to_mask(pred_poly, W, H) if pred_poly else \
                    np.zeros((H, W), dtype=bool)
                gt_mask = polygons_to_mask(t["segmentations"], W, H) \
                    if t.get("segmentations") else np.zeros((H, W), dtype=bool)
                pred_area_norm = float(pred_mask.sum()) / max(1, W * H)
                gt_area_norm = float(gt_mask.sum()) / max(1, W * H)
                signed_area_error = round(pred_area_norm - gt_area_norm, 4)
            except Exception:
                signed_area_error = 0.0
            signed_bb_error = len(pred_bb) - len(t["bounding_boxes"])
            per_task.append({
                **m,
                "tier": seg_tier(t["mask_ratio"]),
                "plant": t.get("plant", "?"),
                "target_class": t.get("target_class", ""),
                "mask_ratio": t["mask_ratio"],
                "task_id": t["task_id"],
                "n_pred_bb": len(pred_bb),
                "n_gt_bb": len(t["bounding_boxes"]),
                "signed_bb_error":   signed_bb_error,
                "signed_area_error": signed_area_error,
            })

        if not per_task:
            return {"score": 0.0, "dice_mask": 0.0, "n": 0}

        def _agg(rows, key):
            return round(mean([r[key] for r in rows]), 4) if rows else 0.0

        per_tier = {
            "all": {k: _agg(per_task, k) for k in
                    ("dice_mask", "miou_mask", "dice_bbox", "miou_bbox",
                     "pixel_precision", "pixel_recall")} | {"n": len(per_task)},
            "moderate": {k: _agg([r for r in per_task if r["tier"] == "moderate"], k)
                         for k in ("dice_mask", "miou_mask")} | {
                             "n": sum(1 for r in per_task if r["tier"] == "moderate")},
            "small": {k: _agg([r for r in per_task if r["tier"] == "small"], k)
                      for k in ("dice_mask", "miou_mask")} | {
                          "n": sum(1 for r in per_task if r["tier"] == "small")},
        }

        plants = sorted({r["plant"] for r in per_task})
        per_plant = sorted(
            [(p, _agg([r for r in per_task if r["plant"] == p], "dice_mask"))
             for p in plants], key=lambda x: x[1])

        all_metrics = per_tier["all"]
        return {
            "score": all_metrics["dice_mask"],     # primary VTOS score
            "dice_mask": all_metrics["dice_mask"],
            "miou_mask": all_metrics["miou_mask"],
            "dice_bbox": all_metrics["dice_bbox"],
            "miou_bbox": all_metrics["miou_bbox"],
            "pixel_precision": all_metrics["pixel_precision"],
            "pixel_recall": all_metrics["pixel_recall"],
            "n": all_metrics["n"],
            "per_tier": per_tier,
            "per_plant": per_plant,
            # _per_image: rich per-task records (analyzers use these via
            # the engine's `per_image_results` arg). Includes all per-task
            # metrics + tier/plant/mask_ratio/n_pred_bb/n_gt_bb so the analyzer
            # can group, sort, and target failure modes.
            "_per_image": [
                {
                    "task_id":           r["task_id"],
                    "plant":             r["plant"],
                    "target_class":      r["target_class"],
                    "tier":              r["tier"],
                    "mask_ratio":        r["mask_ratio"],
                    "dice_mask":         r["dice_mask"],
                    "miou_mask":         r["miou_mask"],
                    "dice_bbox":         r["dice_bbox"],
                    "miou_bbox":         r["miou_bbox"],
                    "pixel_precision":   r["pixel_precision"],
                    "pixel_recall":      r["pixel_recall"],
                    "n_pred_bb":         r["n_pred_bb"],
                    "n_gt_bb":           r["n_gt_bb"],
                    # Signed errors: + ⇒ over-predict, - ⇒ under-predict.
                    "signed_bb_error":   r["signed_bb_error"],
                    "signed_area_error": r["signed_area_error"],
                }
                for r in per_task
            ],
        }


class SegEvaluatorAdapter:
    """Adapter so SegPersistentEvaluator matches the surface VisionSearchEngine expects.

    The engine calls `.evaluate(code)`; this wrapper fixes split="train" and
    forwards the n_train_eval cap so the engine doesn't have to know about it.

    The engine ALSO calls `.evaluate_analyzer(analyzer_code, per_image, ...)`
    when analyzer-mode is ON (search_engine.py:2166, 2697). The missing
    method made every analyzer-ON run silently broken — analyzers were
    LLM-generated and recorded in analyzer_snapshots, but every one had
    error="...has no attribute 'evaluate_analyzer'" and produced 0
    observations. That meant the search loop's diagnostic feedback channel
    was dead for psv2_013, psv2_014, psv2_015 (and the original psv2_p016
    until this fix landed). Mirrors baselines/seg_evaluator.py's impl
    (the sibling that the legacy SegmentationEvaluator path uses)."""
    def __init__(self, seg_eval: SegPersistentEvaluator, n_train_eval: int):
        self._inner = seg_eval
        self._n = n_train_eval

    def evaluate(self, code: str, **kwargs) -> Dict:
        return self._inner.evaluate(code, split="train", n_tasks=self._n)

    def evaluate_analyzer(self, analyzer_code: str, per_image_results):
        """In-process exec of analyzer program. The analyzer defines
        `analyze(per_image_results) -> List[str]`; we run it on the seg
        per_image dicts and return a list of observation strings.

        Failures (no `analyze`, runtime error, type errors) become a single
        observation string so the search loop never crashes on a bad
        analyzer."""
        try:
            import numpy as np
            namespace = {"np": np}
            exec(analyzer_code, namespace)
            analyze_fn = namespace.get("analyze")
            if not callable(analyze_fn):
                return ["Analyzer execution error: no callable `analyze` "
                        "defined in analyzer code."]
            result = analyze_fn(per_image_results)
            if result is None:
                return []
            if not isinstance(result, (list, tuple)):
                return [str(result)]
            return [str(obs) for obs in result]
        except Exception as e:
            return [f"Analyzer execution error: {e}"]
