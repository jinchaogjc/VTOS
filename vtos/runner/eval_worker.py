"""
VTOS Persistent Evaluation Worker
=================================
Hosts vision models (DINO + SAM) in memory and evaluates algorithm code
via a JSON-based stdin/stdout protocol.

Evaluation Metric: Symmetric Point-in-BBox with F1 primary score
  - Hit condition (symmetric): pred_center ∈ GT_box  OR  GT_center ∈ pred_box
  - One-to-one greedy matching sorted by center distance (closest first)
  - Primary score: point_f1 = 2H / (N + M)  [point-in-box F1, harmonic mean of P & R]
  - Secondary:     MAE, signed_error (bias direction for LLM guidance)
  - Dropped:       mIoU (measures localization, not counting)

Math:
  c_pred = ((x1+x2)/2, (y1+y2)/2)        pred box center
  c_gt   = (gx+gw/2,  gy+gh/2)            GT box center
  hit(P,G) = [c_pred ∈ G] OR [c_gt ∈ P]  symmetric containment
  H = |greedy_match(valid_pairs)|
  point_f1 = 2H / (N + M)    (better training signal than min(P,R); point-in-box F1)
  grounded = H / max(N,M) = min(P,R)      (kept for reference)
"""
import sys, os, json, traceback
from concurrent.futures import ThreadPoolExecutor

def grounded_matching(pred_bboxes, gt_bboxes):
    """
    Hungarian-matched point-in-box F1 (point_f1).

    [v2 update] This is now a thin wrapper around the canonical implementation
    in ``metrics/grounded_metrics.py::compute_point_f1_metrics``, ensuring VTOS's
    training/val/test feedback uses the SAME metric the paper reports.

    Previously this used greedy distance-sorted matching, which inflated point_f1
    slightly when predictions clustered around true objects. Hungarian
    bipartite assignment (DETR-style, ICIP 2022 / ECCV 2020) is provably
    optimal one-to-one matching.

    Args:
        pred_bboxes: list of [x1, y1, x2, y2] normalized (toolbox output)
        gt_bboxes:   list of [x,  y,  w,  h]  normalized (GT annotation)

    Returns:
        dict with hits, misses, false_pos, precision, recall, point_f1, grounded_score
        (same shape as before — drop-in compatible with all call sites).
    """
    # Lazy import so the subprocess's sys.path is set up first (main() inserts
    # code_root into sys.path before any eval call reaches this function).
    from metrics.grounded_metrics import compute_point_f1_metrics
    return compute_point_f1_metrics(pred_bboxes, gt_bboxes)

def main():
    # Setup paths
    code_root = sys.argv[1]
    img_dir = sys.argv[2]
    json_path = sys.argv[3]
    
    sys.path.insert(0, code_root)
    from vtos.toolbox import ToolboxWrapper

    class MockField: pass
    toolbox = ToolboxWrapper(MockField())

    sys.stderr.write("🚀 [Worker] Loading models into memory...\n")

    # Force-load DINO before ThreadPoolExecutor starts — prevents race condition
    # where multiple threads each see _DINO_MODEL=None and all attempt concurrent loads.
    from vtos.toolbox import _get_dino_model_and_processor
    import torch
    _device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
    _get_dino_model_and_processor(_device)
    sys.stderr.write("✅ [Worker] Ready.\n")

    with open(json_path, 'r') as f:
        full_data = json.load(f)

    # ── Pre-warm disk cache ────────────────────────────────────────────
    # Run DINO on every image × common thresholds at startup so all future
    # evaluations (baseline + search solutions) hit the disk cache instead
    # of re-running the model. On a warm cache this entire block is instant.
    _PREWARM_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]
    import re as _re, time as _time
    _prewarm_hits, _prewarm_misses = 0, 0
    _t_prewarm = _time.time()
    sys.stderr.write("🔥 [Cache] Pre-warming DINO detection cache...\n")
    for _item in full_data:
        _img_path = os.path.join(img_dir, _item['image_path'])
        _tq = _re.sub(r'\s*\([^)]*\)', '', _item['target_class']).strip()
        for _thr in _PREWARM_THRESHOLDS:
            # Check disk cache directly to get accurate hit/miss count
            _disk_hit = (toolbox.dino_skill.cache.get(_img_path, _tq, {"threshold": _thr}) is not None
                         if toolbox.dino_skill.cache else False)
            if _disk_hit:
                _prewarm_hits += 1
            else:
                # grounding_dino_detect: checks _DINO_CACHE (empty) → dino_skill.detect →
                # checks ImmutableEmbeddingCache (disk) → on miss runs DINO and
                # writes to BOTH disk cache and _DINO_CACHE.
                toolbox.grounding_dino_detect(_img_path, text_query=_tq, box_threshold=_thr)
                _prewarm_misses += 1
    _prewarm_elapsed = _time.time() - _t_prewarm
    sys.stderr.write(f"🔥 [Cache] Pre-warm done in {_prewarm_elapsed:.1f}s: {_prewarm_hits} hits, {_prewarm_misses} misses\n")

    # Concurrent model calls crash the Metal backend on Apple Silicon (MPS);
    # use one thread there. Per-image results do not depend on the thread count.
    executor = ThreadPoolExecutor(max_workers=1 if _device == "mps" else 2)
    
    # JSON-based protocol over stdin/stdout
    while True:
        line = sys.stdin.readline()
        if not line: break
        
        try:
            task = json.loads(line)
            if task is None: break
            
            algo_code = task['code']
            test_ids = task['test_ids']
            test_data = [item for item in full_data if item['image_path'] in test_ids]
            
            def run_one(item):
                img_path = os.path.join(img_dir, item['image_path'])
                import re
                tq = re.sub(r'\s*\([^)]*\)', '', item['target_class']).strip()
                gt_count = item['count']
                gt_bboxes = item.get('bounding_boxes', [])

                lvars = {'image_path': img_path, 'text_query': tq, 'toolbox': toolbox}
                lvars.update({'np': __import__('numpy'), 'cv2': __import__('cv2'),
                              'os': __import__('os'), 'Image': __import__('PIL.Image').Image,
                              'torch': __import__('torch')})
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        w, h = img.size
                    exec(algo_code, lvars, lvars)
                    pred_bboxes = lvars.get('bboxes', [])
                    err_msg = None
                except Exception as e:
                    pred_bboxes = []
                    w, h = 640, 480
                    err_msg = traceback.format_exc()
                
                # Sanitize: convert any numpy/tensor elements to plain Python floats
                def _clean_bbox(b):
                    try:
                        return [float(x) for x in (b.tolist() if hasattr(b, 'tolist') else b)]
                    except Exception:
                        return []
                pred_bboxes = [_clean_bbox(b) for b in pred_bboxes if b is not None]

                pred_count = len(pred_bboxes)
                ae = abs(pred_count - gt_count)
                rel_ae = ae / max(1, gt_count)

                # Point-in-BBox grounded matching (Hungarian) — legacy "point_f1"
                match = grounded_matching(pred_bboxes, gt_bboxes)
                # Hungarian-matched mIoU (canonical) + strict F1@IoU=0.5
                from metrics.grounded_metrics import compute_miou, compute_f1_iou
                miou = compute_miou(pred_bboxes, gt_bboxes)
                f1_iou = compute_f1_iou(pred_bboxes, gt_bboxes, iou_threshold=0.5)

                signed_e = pred_count - gt_count  # positive = over-counting
                return {
                    # Full absolute path (analyzer code may open the image).
                    "image_path": img_path,
                    "gt_count": gt_count,
                    "pred_count": pred_count, "ae": ae, "rel_ae": rel_ae,
                    "signed_error": signed_e,
                    "image_width": w, "image_height": h, "error": err_msg,
                    # Target class of the query.
                    "target_class": tq,
                    # Lenient (point-in-bbox) — legacy
                    "hits": match["hits"], "misses": match["misses"],
                    "false_pos": match["false_pos"],
                    "precision": match["precision"], "recall": match["recall"],
                    "point_f1": match["point_f1"], "grounded_score": match["grounded_score"],
                    # Strict (IoU>=0.5) — new, standard detection F1
                    "f1_iou50":        f1_iou["f1"],
                    "precision_iou50": f1_iou["precision"],
                    "recall_iou50":    f1_iou["recall"],
                    "hits_iou50":      f1_iou["hits"],
                    # mIoU (already IoU-based, unchanged)
                    "miou": miou,
                    "pred_bboxes": pred_bboxes,
                    "gt_bboxes": gt_bboxes
                }

            results = list(executor.map(run_one, test_data))

            n = max(1, len(results))
            mae          = sum(r['ae'] for r in results) / n
            avg_rel_ae   = sum(r['rel_ae'] for r in results) / n
            signed_error = sum(r['signed_error'] for r in results) / n  # bias direction
            # Lenient (legacy, point-in-bbox)
            avg_point_f1      = sum(r["point_f1"] for r in results) / n
            avg_grounded = sum(r['grounded_score'] for r in results) / n
            avg_precision = sum(r['precision'] for r in results) / n
            avg_recall   = sum(r['recall'] for r in results) / n
            # Strict (IoU>=0.5) — new
            avg_f1_iou50    = sum(r["f1_iou50"]        for r in results) / n
            avg_prec_iou50  = sum(r["precision_iou50"] for r in results) / n
            avg_rec_iou50   = sum(r["recall_iou50"]    for r in results) / n
            # Other
            avg_miou     = sum(r["miou"] for r in results) / n
            avg_gt_count = sum(r["gt_count"] for r in results) / n
            total_hits      = sum(r['hits'] for r in results)
            total_misses    = sum(r['misses'] for r in results)
            total_false_pos = sum(r['false_pos'] for r in results)

            sys.stdout.write(json.dumps({
                "mae": mae, "avg_rel_ae": avg_rel_ae,
                "signed_error": round(signed_error, 3),
                "results": results,
                # Lenient (point-in-bbox) — legacy point_f1
                "point_f1": round(avg_point_f1, 4),
                "grounded_score": round(avg_grounded, 4),
                "avg_precision": round(avg_precision, 4),
                "avg_recall": round(avg_recall, 4),
                # Strict (IoU>=0.5) — standard detection F1
                "f1_iou50":        round(avg_f1_iou50, 4),
                "precision_iou50": round(avg_prec_iou50, 4),
                "recall_iou50":    round(avg_rec_iou50, 4),
                # Hungarian-matched mIoU + GT count avg (for composite/dual-rank scoring)
                "avg_miou": round(avg_miou, 4),
                "avg_gt_count": round(avg_gt_count, 2),
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_false_pos": total_false_pos
            }) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            sys.stderr.write(f"❌ [Worker Error] {traceback.format_exc()}\n")
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
