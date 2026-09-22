"""Seg toolbox + sandbox — extracted from the legacy seg_vtos_core for reuse by SegmentationEvaluator."""
from __future__ import annotations

from typing import Dict, List, Optional

SAM2_MODEL_ID = "facebook/sam2-hiera-large"
# CLIP backbone for the semantic patch filter (clip_select).
# ViT-L/14 — the stronger filter (better diseased-vs-healthy separation on small
# patch crops). Swap to the faster "openai/clip-vit-base-patch32" for a quicker,
# weaker filter.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# ── CLIP semantic filter (lazy, process-level singleton) ─────────────────────
# Loaded once per process and reused across every build_toolbox() call so the
# search loop does not re-instantiate CLIP per image.
_CLIP_CACHE: Dict = {}


def _get_clip():
    """Load CLIP once per process; return (model, processor, device)."""
    if "model" not in _CLIP_CACHE:
        import torch
        from transformers import AutoProcessor, CLIPModel
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(dev).eval()
        proc = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
        _CLIP_CACHE.update(model=model, proc=proc, dev=dev)
    return _CLIP_CACHE["model"], _CLIP_CACHE["proc"], _CLIP_CACHE["dev"]


def _clip_patch_text_sim(patches: List, prompts: List[str]):
    """Cosine-similarity matrix between PIL image patches and text prompts.
    Returns an (n_patches, n_prompts) float32 numpy array."""
    import numpy as _np
    import torch
    if not patches or not prompts:
        return _np.zeros((len(patches or []), len(prompts or [])),
                         dtype=_np.float32)
    model, proc, dev = _get_clip()
    with torch.no_grad():
        inputs = proc(text=list(prompts), images=list(patches),
                      return_tensors="pt", padding=True)
        out = model(**{k: v.to(dev) for k, v in inputs.items()})
        # logits_per_image = logit_scale * cosine(image, text); undo the scale
        sim = (out.logits_per_image
               / model.logit_scale.exp()).float().cpu().numpy()
    return sim


# ── inference-result cache (disk-backed: VTOS/assets/cache) ───────────────────
# Shared ImmutableEmbeddingCache so the VTOS search reuses DINO + SAM 2 inference
# results across candidates and iterations. Without it every candidate re-runs
# detect() + segment() from scratch — the dominant search cost on CPU.
_SEG_CACHE: Dict = {}


def _get_seg_cache():
    """Lazy shared disk-backed cache for foundation-model inference results."""
    if "cache" not in _SEG_CACHE:
        from vtos.inference_cache import ImmutableEmbeddingCache
        _SEG_CACHE["cache"] = ImmutableEmbeddingCache()
    return _SEG_CACHE["cache"]


# ── GroundingDINO detector (lazy, process-level singleton) ───────────────────
_DINO_CACHE: Dict = {}


def _get_dino():
    """Load GroundingDINO once per process; reuse it (detect() results cached)."""
    if "skill" not in _DINO_CACHE:
        from vtos.toolbox import GroundingDINOSkill
        skill = GroundingDINOSkill()
        skill.cache = _get_seg_cache()
        _DINO_CACHE["skill"] = skill
    return _DINO_CACHE["skill"]


# ── toolbox (plain or instrumented for EFE) ──────────────────────────────────
def build_toolbox(sam2, image_path: str, W: int, H: int,
                  exec_log: Optional[List[Dict]] = None,
                  gt_boxes: Optional[List] = None,
                  split: str = "test") -> Dict:
    """Build the seg toolbox exposed to LLM-emitted code.

    Tools: GroundingDINO detection (detect) + SAM 2 box refinement (segment);
    SAM 2 dense proposals (propose_masks); a CLIP semantic filter (clip_select)
    with descriptive-prompt helpers; polygon ops (merge / subtract / filter /
    clean). If exec_log is provided, every tool call appends a record to it.
    """
    from metrics.segmentation_metrics import mask_to_polygons

    # ── detection — GroundingDINO + SAM 2 box refinement ────────────────
    def detect(text: str, threshold: float = 0.30):
        """Open-vocabulary detection (GroundingDINO): returns bboxes
        [x, y, w, h] normalized to [0,1] for the text query."""
        out = []
        try:
            bbs_px, _, _ = _get_dino().detect(
                image_path=image_path, text_query=text, box_threshold=threshold)
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
            exec_log.append({"tool": "detect", "text": text, "n_out": len(out)})
        return out

    def segment(bboxes_xywh):
        """SAM 2 refines each bbox ([x,y,w,h] normalized) into a polygon mask."""
        polys = []
        try:
            xyxy_abs = []
            for b in (bboxes_xywh or []):
                if len(b) != 4:
                    continue
                x, y, w, h = b
                xyxy_abs.append([x * W, y * H, (x + w) * W, (y + h) * H])
            if xyxy_abs:
                for m in sam2.segment(image_path, xyxy_abs):
                    polys.extend(mask_to_polygons(m))
        except Exception:
            polys = []
        if exec_log is not None:
            exec_log.append({"tool": "segment", "n_in": len(bboxes_xywh or []),
                             "n_out": len(polys)})
        return polys

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

    def subtract_polygons(polygons_a, polygons_b):
        """Pixel-level subtraction: mask(polygons_a) AND NOT mask(polygons_b).
        Returns the resulting polygons. Extremely useful for finding anomalies
        by subtracting healthy regions from the global plant envelope."""
        import cv2
        from metrics.segmentation_metrics import polygons_to_mask
        if not polygons_a:
            if exec_log is not None:
                exec_log.append({"tool": "subtract_polygons", "n_in_a": 0, "n_in_b": len(polygons_b or []), "n_out": 0})
            return []
        if not polygons_b:
            if exec_log is not None:
                exec_log.append({"tool": "subtract_polygons", "n_in_a": len(polygons_a), "n_in_b": 0, "n_out": len(polygons_a)})
            return list(polygons_a)
        mask_a = polygons_to_mask(polygons_a, W, H).astype("uint8")
        mask_b = polygons_to_mask(polygons_b, W, H).astype("uint8")
        res_mask = cv2.bitwise_and(mask_a, cv2.bitwise_not(mask_b))
        out = mask_to_polygons(res_mask)
        if exec_log is not None:
            exec_log.append({"tool": "subtract_polygons", "n_in_a": len(polygons_a),
                             "n_in_b": len(polygons_b), "n_out": len(out)})
        return out

    def propose_masks(grid: int = 4):
        """GroundingDINO-free region proposer: prompt SAM 2 with a grid of
        foreground points and return one polygon mask per point, overlap-
        deduplicated. Proposes candidate regions WITHOUT naming the target —
        the algorithm then selects the diseased ones (e.g. color_anomaly /
        clip_select / area / texture). grid=N gives an N*N point grid (capped
        at 6)."""
        g = max(2, min(int(grid), 6))
        out = []
        try:
            # All g*g points in ONE SAM 2 call: the image is encoded once and
            # the decoder emits one mask per point. (Prompting SAM 2 per point
            # would re-encode the image g*g times — ~g*g x slower.)
            pts_abs = [[(i + 0.5) / g * W, (j + 0.5) / g * H]
                       for i in range(g) for j in range(g)]
            for m in sam2.segment_points(image_path, pts_abs):
                comps = mask_to_polygons(m)
                if comps:
                    # one point prompts one region — keep its largest component
                    # (drops tiny fragments that bloat the O(n^2) merge below)
                    out.append(max(comps, key=_poly_area_norm))
        except Exception:
            out = []
        if out:
            out = merge_polygons(out, iou_thresh=0.7)
        if exec_log is not None:
            exec_log.append({"tool": "propose_masks", "grid": g,
                             "n_out": len(out)})
        return out

    # ── semantic patch filter (CLIP image-text alignment) ───────────────
    def _poly_to_patch(poly, img):
        """Crop a polygon's padded bbox and gray-mask everything outside the
        polygon, so CLIP scores the patch itself rather than its background.
        Returns a PIL.Image, or None for a degenerate / tiny polygon."""
        from PIL import Image as _PI, ImageDraw as _ID
        xs = [float(v[0]) for v in poly]
        ys = [float(v[1]) for v in poly]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        pw = (x2 - x1) * 0.08 + 0.01
        ph = (y2 - y1) * 0.08 + 0.01
        cx1 = int(max(0.0, x1 - pw) * W)
        cy1 = int(max(0.0, y1 - ph) * H)
        cx2 = int(min(1.0, x2 + pw) * W)
        cy2 = int(min(1.0, y2 + ph) * H)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return None
        crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
        cw, ch = crop.size
        m = _PI.new("L", (cw, ch), 0)
        _ID.Draw(m).polygon(
            [(float(v[0]) * W - cx1, float(v[1]) * H - cy1) for v in poly],
            fill=255)
        return _PI.composite(crop, _PI.new("RGB", (cw, ch), (128, 128, 128)), m)

    def clip_select(polygons,
                    disease_prompt=("a photo of a diseased plant with "
                                    "lesions, spots or rot"),
                    healthy_prompt="a photo of healthy plant tissue",
                    top_frac: float = 0.3, min_disease_prob=None):
        """CLIP semantic filter — propose-then-SELECT. For each polygon patch,
        crop it and use CLIP to compare it against disease_prompt vs
        healthy_prompt; keeps the patches that look diseased. If min_disease_prob
        is set, keeps every patch whose P(disease) exceeds it; otherwise keeps
        the top `top_frac` fraction by P(disease). Pair with propose_masks.
        Returns the selected polygons."""
        kept = []
        try:
            import numpy as _np
            from PIL import Image as _PILImage
            polys = [p for p in (polygons or []) if p and len(p) >= 3]
            if polys:
                with _PILImage.open(image_path) as _im:
                    img = _im.convert("RGB")
                    valid = [(i, _poly_to_patch(polys[i], img))
                             for i in range(len(polys))]
                valid = [(i, pt) for i, pt in valid if pt is not None]
                if valid:
                    sim = _clip_patch_text_sim(
                        [pt for _, pt in valid],
                        [str(disease_prompt), str(healthy_prompt)])
                    logits = sim * 100.0  # CLIP temperature
                    ex = _np.exp(logits - logits.max(axis=1, keepdims=True))
                    p_dis = ex[:, 0] / ex.sum(axis=1)  # P(disease)
                    if min_disease_prob is not None:
                        sel = [k for k in range(len(valid))
                               if p_dis[k] > float(min_disease_prob)]
                    else:
                        n_keep = max(1, int(round(float(top_frac) * len(valid))))
                        sel = list(_np.argsort(-p_dis))[:n_keep]
                    kept = [polys[valid[k][0]] for k in sel]
        except Exception:
            kept = []
        return kept

    def plant_healthy_prompt(plant_name: str) -> str:
        """
        Returns a highly specific, visually descriptive CLIP prompt for the normal, healthy 
        morphology of the given plant species. To prevent data leakage and maintain strict 
        zero-shot OOD validation, custom descriptions are ONLY defined for training crops. 
        Validation and Test crops utilize a robust, dynamically generated generic template.
        """
        p = str(plant_name).lower()
        
        # val / test (zero-shot OOD) skip the train-only species descriptors
        # below and use the generic template — prevents disease/plant data leak.
        if split != "train":
            if any(x in p for x in ["blueberry", "citrus", "grape", "peach"]):
                return f"a photo of fresh, perfect, healthy clean green {plant_name} leaf or ripe fruit"
            return f"a photo of perfect, clean, healthy green leaf or plant tissue of a {plant_name}"

        # 1. Custom pristine descriptors strictly for Train Split crops (15 species)
        if "apple" in p:
            return "a photo of a perfect healthy green apple leaf or smooth shiny apple fruit"
        if "banana" in p:
            return "a photo of a healthy, clean, perfect large green banana leaf"
        if "basil" in p:
            return "a photo of healthy, aromatic, perfect smooth green basil leaves"
        if "bean" in p:
            return "a photo of a healthy green bean leaf with smooth veins"
        if "cabbage" in p:
            return "a photo of a healthy, dense green cabbage head or cabbage leaf"
        if "cauliflower" in p:
            return "a photo of a perfect healthy white cauliflower curd or green cauliflower leaf"
        if "cherry" in p:
            return "a photo of a healthy green cherry leaf or perfect shiny red cherry fruit"
        if "cucumber" in p:
            return "a photo of a healthy green cucumber leaf or long green cucumber fruit"
        if "garlic" in p:
            return "a photo of perfect healthy green garlic leaves or a clean white garlic bulb"
        if "ginger" in p:
            return "a photo of a healthy green ginger leaf or smooth clean ginger rhizome root"
        if "plum" in p:
            return "a photo of a healthy green plum leaf or perfect smooth round plum fruit"
        if "rice" in p:
            return "a photo of a healthy, bright green rice plant leaf or golden rice grains"
        if "strawberry" in p:
            return "a photo of a perfect healthy green strawberry leaf or ripe red strawberry fruit"
        if "tobacco" in p:
            return "a photo of a large, healthy, perfect broad green tobacco leaf"
        if "zucchini" in p:
            return "a photo of a healthy, large green zucchini leaf or smooth green zucchini fruit"
        
        # 2. Universal, dynamically generated zero-shot fallback for Val, Test, and unseen OOD crops
        if any(x in p for x in ["blueberry", "citrus", "grape", "peach"]):
            return f"a photo of fresh, perfect, healthy clean green {plant_name} leaf or ripe fruit"
        return f"a photo of perfect, clean, healthy green leaf or plant tissue of a {plant_name}"

    def plant_disease_prompt(plant_name: str, disease_name: str) -> str:
        """
        Returns a highly specific, visually descriptive CLIP prompt for the diseased morphology
        of the given plant species and target disease. To prevent data leakage, custom disease 
        mappings are strictly limited to Train Split diseases. Validation and Test splits 
        rely on a robust, dynamically generated template.
        """
        p = str(plant_name).lower()
        d = str(disease_name).lower()
        
        # Custom disease descriptors — TRAIN SPLIT ONLY. val / test (zero-shot
        # OOD) skip these so no researcher-written description of an unseen
        # disease leaks in; they fall through to the generic template below.
        desc = ""
        if split == "train":
            if "black rot" in d and "apple" in p:
                desc = "dark decaying brown or black rotten spots and circular lesions with yellow halos"
            elif "mosaic virus" in d or "mosaic" in d:
                desc = "mottled patches of light green, yellow, and dark green colors in a mosaic pattern"
            elif "rust" in d:
                desc = "powdery rusty orange, reddish-brown, or bright yellow spore pustules and spots"
            elif "scab" in d:
                desc = "dark olive-green, brown, or black velvety spots and scabby crusty lesions"
            elif "panama" in d:
                desc = "severe yellowing, brown wilting, and decaying dead areas on the leaf margins"
            elif "downy mildew" in d:
                desc = "white, gray, or pale yellow fuzzy mold and downy growth on the surface"
            elif "powdery mildew" in d:
                desc = "powdery white or light gray dusty coating, spots, and fuzzy patches of mold"
            elif "halo blight" in d:
                desc = "small dark brown water-soaked spots surrounded by a prominent wide pale-yellow ring or halo"
            elif "alternaria" in d or "cercospora" in d or "leaf spot" in d:
                desc = "circular dark brown or black spots, sometimes with concentric rings forming a target pattern"
            elif "leaf blight" in d or "blight" in d:
                desc = "large, dried, scorched brown or black dead patches on the foliage"
            elif "powdery" in d:
                desc = "powdery white or dusty gray spots and patches of mildew"
            elif "wilt" in d:
                desc = "wilting, drooping, shriveled, and pale green or yellow leaves"
            elif "pocket disease" in d:
                desc = "swollen, severely distorted, puckered, curled, and thickened pale yellow or reddish-pink leaves"
            elif "blast" in d:
                desc = "spindle-shaped, diamond-shaped, or eye-shaped gray spots with dark reddish-brown borders"
            elif "sheath blight" in d:
                desc = "greenish-gray, oval, water-soaked spots with dark reddish-brown margins"
            elif "anthracnose" in d:
                desc = "sunken, dark brown or black water-soaked decaying lesions and necrotic spots"
            elif "leaf scorch" in d:
                desc = "dry, brown, scorched leaf margins and burnt-looking tips"
        if not desc:
            # 2. Universal zero-shot dynamic fallback parser for Val, Test, and unseen OOD diseases
            features = []
            if "spot" in d or "lesion" in d:
                features.append("abnormal discolored spots or lesions")
            if "mildew" in d or "mold" in d or "powdery" in d:
                features.append("fuzzy mold patches or powdery mildew growth")
            if "rot" in d or "decay" in d:
                features.append("dark decaying rotten spots")
            if "blight" in d or "scorch" in d:
                features.append("dried scorched brown blighted patches")
            if "curl" in d or "roll" in d or "wilt" in d:
                features.append("distorted curled, rolled, or wilting drooping leaves")
            if "rust" in d:
                features.append("powdery rusty orange or brown spots")
            
            if features:
                desc = " showing " + " and ".join(features)
            else:
                desc = f" showing symptoms of {disease_name}"

        return f"a photo of a diseased {plant_name} with {disease_name},{desc}"

    tools = {
        "detect": detect,
        "segment": segment,
        "propose_masks": propose_masks,
        "clip_select": clip_select,
        "merge_polygons": merge_polygons,
        "subtract_polygons": subtract_polygons,
        "filter_polygons_by_area": filter_polygons_by_area,
        "clean_mask": clean_mask,
        "plant_healthy_prompt": plant_healthy_prompt,
        "plant_disease_prompt": plant_disease_prompt,
    }
    return tools


