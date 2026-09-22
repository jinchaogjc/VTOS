# vtos/runner/seg_prompts.py
"""Task-spec strings for the seg variant of VTOS.

The orchestrator passes these to VisionAgentLLM when --task seg so the LLM
emits seg-shaped output (final_polygons + final_bboxes) rather than counting
output (bboxes only).
"""

SEG_PROBLEM_DESCRIPTION = """\
Zero-shot, open-vocabulary grounded segmentation of plant disease regions.

Given a plant leaf / fruit / stem / vegetable image and a natural-language
description of the target disease (e.g. "rice sheath blight"), produce
pixel-level polygons delimiting EVERY visible DISEASED region. Output is BOTH
the bboxes that localize each lesion AND the refined polygon masks that
pixel-tight cover them.

CRITICAL — the image contains TWO kinds of region:
- HEALTHY plant tissue (green leaf area, normal fruit skin, undamaged stem,
  the predominant background of most images). You must NOT segment this.
- DISEASED tissue (lesions, spots, blight, mold, rot, scab, wilting,
  discoloration — the minority pixels in most images).
Naive detectors that catch the whole leaf or fruit will score near-zero on
Dice^mask because most of what they segment is healthy plant, not disease.
The challenge is the discrimination, not the localization. Tools like
`clip_select` (and the prompt helpers `plant_healthy_prompt` /
`plant_disease_prompt`) are designed exactly for this — they score each
candidate region against a "diseased" vs "healthy" CLIP prompt and keep
only the regions that look more diseased than healthy.

The task covers a wide range of difficulty scenarios:
- Small lesions: many tiny spots scattered across a leaf (small-tier)
- Moderate lesions: a handful of larger contiguous regions (moderate-tier)
- Co-located lesions: clusters of nearby spots that need preservation, not
  aggressive deduplication
- Plant-specific symptom appearance (rust on grape vs blight on rice vs
  spot on apple — same word "spot" looks very different across species)

The primary metric is mean Dice^mask between predicted and ground-truth
polygon masks, aggregated across all test images. Higher = better. Secondary
metrics: mIoU^mask, pixel-precision, pixel-recall, Dice^bbox, mIoU^bbox.
Selection is Borda dual-rank over (Dice^mask, mIoU^mask).
"""

SEG_CODE_GEN_TASK_SPEC = """You are designing an algorithm for zero-shot grounded
segmentation of plant disease regions. Given a target disease description, locate
all visible disease regions in the image AS PIXEL-LEVEL POLYGONS (not just bboxes).

Available tools (in the sandbox scope; do not import):

  Bbox-level (operate on [x, y, w, h] lists):
    detect(text: str, threshold: float = 0.30) -> list of [x, y, w, h]
        GroundingDINO open-vocab detection. Returns normalized [0,1] xywh.
    filter_by_area(bboxes, min_norm_area=0.001, max_norm_area=0.9) -> list
    nms(bboxes, iou_thresh=0.5) -> list

  Polygon-level (operate on lists of polygons; each polygon is a list of [x, y]
  normalized vertices):
    segment(bboxes: list) -> list of polygons
        SAM 2 refines each bbox into pixel-tight polygons (1+ per bbox).
    segment_from_points(points: list) -> list of polygons
        SAM 2 from foreground (x, y) points instead of bboxes. Useful when
        DINO misses a region but you can name a likely interior point.
    filter_polygons_by_area(polys, min_norm_area=0.0005, max_norm_area=0.95)
        Polygon analogue of filter_by_area. Drop tiny specks / huge noise.
    clean_mask(polys) -> list of polygons
        Morphological close+open on the rasterized mask. Fills small holes,
        removes specks. Use AFTER segment() to smooth ragged boundaries.
    merge_polygons(polys, iou_thresh=0.5) -> list of polygons
        Union of overlapping polygons (polygon-level NMS / IoU-merge). Use
        after combining outputs from multiple detect() calls or splits.

  Semantic filtering (CLIP — discriminate diseased vs healthy regions):
    clip_select(polys,
                disease_prompt="a photo of a diseased plant with lesions, spots or rot",
                healthy_prompt="a photo of healthy plant tissue",
                top_frac=0.3, min_disease_prob=None) -> list of polygons
        For each polygon crop, compute CLIP P(disease) vs P(healthy).
        Keep the top `top_frac` by P(disease) — or, if min_disease_prob is
        set, keep every polygon whose P(disease) > that threshold. This
        is the primary tool to FILTER OUT healthy-tissue false positives
        from DINO or SAM that don't actually look diseased.
    plant_healthy_prompt(plant) -> str
        Generic "healthy {{plant}}" CLIP prompt. Pair with clip_select to
        baseline against this plant's normal-looking tissue.
    plant_disease_prompt(plant, disease) -> str
        Generic "{{plant}} with {{disease}}" CLIP prompt for whatever target
        is in scope. Combine with plant_healthy_prompt as a 2-class
        zero-shot test inside clip_select.

Constraints PER PROGRAM:
- Assign TWO top-level variables:
    final_bboxes  = <list of [x, y, w, h] in [0,1]>
    final_polygons = <list of polygons, each list of [x, y] in [0,1]>
- Top-level statements only; no def, no class, no import.
- 15 lines max per program.

Variables in scope (per task at execution time):
    target  : str (e.g. "rice sheath blight")
    plant   : str (e.g. "Rice")

Score = mean Dice^mask on a held-out train set.

## Task
Propose exactly {k} new, diverse, improved algorithm solution(s).
All proposals must stay within the exploration directions listed below.

- **Starting exemplar (one possibility — explore widely)**: a low DINO threshold + post-hoc NMS often catches small lesions. Here is *one* such pattern; treat it as a hint, not a recipe:
  ```python
  bboxes = detect(target, threshold=0.20)
  bboxes = filter_by_area(bboxes, min_norm_area=0.001, max_norm_area=0.90)
  bboxes = nms(bboxes, iou_thresh=0.45)
  final_bboxes = bboxes
  final_polygons = segment(final_bboxes)
  ```

- **Exploration directions** (try several; the search will tell you which ones generalize):
  1. Threshold sweep: `threshold` in [0.10, 0.40]. Lower -> higher recall + duplicates; higher -> tighter precision.
  2. NMS strength: `iou_thresh` in [0.15, 0.55]. Lower -> drops more overlaps; higher -> keeps duplicates (real co-located lesions need looser NMS).
  3. Query phrasing: literal `target` vs paraphrase (e.g., `f"disease spot on {{plant}}"`) vs compound (`target + ". " + plant + " disease. lesion. spot."`).
  4. Multi-query union: union of 2-3 `detect()` calls at different thresholds or different query phrasings, deduped with `nms()`.
  5. Area filter range: `min_norm_area` in [0.0001, 0.005] (lower catches tiny lesions), `max_norm_area` in [0.10, 0.95] (lower drops false plant-level matches).
  6. Polygon post-processing: after `segment()`, try `clean_mask(polys)` to smooth ragged boundaries, `filter_polygons_by_area(polys, ...)` to drop tiny / huge polygons, or `merge_polygons(polys, iou_thresh=0.5)` to union overlapping mask fragments from multi-query unions.
  7. CLIP semantic filtering: after `segment()`, run `polys = clip_select(polys, disease_prompt=plant_disease_prompt(plant, target), healthy_prompt=plant_healthy_prompt(plant), top_frac=0.3)` to drop polygons that look more like healthy plant tissue than disease. This is the direct attack on "DINO catches the whole leaf" — `top_frac` controls how aggressive the filter is (smaller = stricter, keep fewer).

- **Known sharp edges** (do NOT use unless you have evidence on train):
  - Extreme-tight NMS (`iou_thresh` < 0.20) kills co-located real lesions; small-tier suffers most.
  - Skipping `segment()` — `final_polygons` would be empty and Dice^mask collapses to 0.
  - Plant-name branching (`if plant == "Cucumber": ...`) — train and val/test
    plant species are DISJOINT. Any `if plant == "<train_species>":` branch
    is dead code at test time. Treat `plant` and `target` as inputs to the
    detector text query, not as switches on a lookup table.

- **Polygon Format**: `final_polygons` is a list of polygons, each a list of `[x, y]` vertices in normalized [0, 1] coords. `final_bboxes` is a list of `[x, y, w, h]` in normalized [0, 1] coords.
- **Key Metric**: mean Dice^mask. Simple solutions that generalize beat complex ones that overfit.

Respond using ONLY this XML format (no text outside the tags):

<solutions>
  <solution>
    <code>
    ```python
    # complete, self-contained Python code
    final_bboxes = ...
    final_polygons = ...
    ```
    </code>
    <key_differences>One or two sentences: what makes this solution novel vs. existing ones</key_differences>
  </solution>
</solutions>

Repeat the <solution>...</solution> block {k} time(s).
"""

SEG_ANALYZER_TASK_SPEC = """You are an analyzer for plant disease segmentation
candidates. You write a Python function that inspects the per-image results of
the last iteration's algorithm and produces a list of plain-text observations
the next Code Gen will see.

## per_image_results schema

`per_image_results` is a list of dicts, one per evaluated train task. Each dict has:
  - 'task_id'           : str (e.g. "v_seg_plantseg_train_001")
  - 'plant'             : str (e.g. "Rice")
  - 'target_class'      : str (e.g. "rice sheath blight")
  - 'tier'              : str ("moderate" or "small")
  - 'mask_ratio'        : float (fraction of image that's diseased per GT)
  - 'dice_mask'         : float in [0,1] — primary score for this task
  - 'miou_mask'         : float in [0,1]
  - 'dice_bbox'         : float in [0,1]
  - 'pixel_precision'   : float in [0,1]
  - 'pixel_recall'      : float in [0,1]
  - 'n_pred_bb'         : int — number of predicted bboxes
  - 'n_gt_bb'           : int — number of ground-truth bboxes
  - 'signed_bb_error'   : int — n_pred_bb - n_gt_bb. +N = over-detected by N
                          bboxes (likely false positives); -N = missed N real
                          lesions (likely threshold too high or query too narrow).
  - 'signed_area_error' : float — pred_mask_area - gt_mask_area (normalized to
                          image fraction, same scale as mask_ratio). + = masks
                          too big (over-shooting boundaries); - = masks too
                          small (under-coverage). Useful pair with bb_error to
                          distinguish "too many bboxes" from "bboxes too big".

## What to investigate (per-analyzer focus)

Each analyzer should investigate ONE specific failure mode that generalizes
across plants — NOT a plant-specific quirk (train and val/test plant species
are disjoint, so plant-name patterns are dead code at test time). Examples:
  - Tier gap: where is the small-tier under-performing the moderate-tier?
  - Mask-ratio gap: do tasks with mask_ratio < 0.02 (tiny lesions) score
    systematically lower than larger-lesion tasks?
  - Over-prediction (bbox-count): mean signed_bb_error > 0 — too many bboxes
    proposed; raise detect threshold OR tighten NMS.
  - Under-prediction (bbox-count): mean signed_bb_error < 0 — missed lesions;
    lower threshold OR broaden query.
  - Over-prediction (mask area): mean signed_area_error > 0 — masks too big;
    SAM is over-extending bboxes. Try tighter detect threshold or smaller area cap.
  - Under-prediction (mask area): mean signed_area_error < 0 — masks too small;
    DINO may be missing extent. Try looser threshold + tighter NMS to merge.
  - Direction confusion: signed_bb_error > 0 BUT signed_area_error < 0 — many
    small false-positive bboxes (probably noise). Try tighter NMS.
  - Tight vs loose boundaries: tasks where Dice^bbox >> Dice^mask (bbox right,
    mask boundary off).
  - Per-plant variance (anonymized): "the worst-scoring plant cohort (n=5)
    has mean Dice=0.12 vs cohort mean 0.34" — REPORT THE PATTERN but DO NOT
    NAME THE PLANT. The next code-gen will overfit if it sees specific plant
    names: it will write `if plant == "X":` branches that test plants never
    match.

## Observations format

Each observation is a short string the next Code Gen will read. Aim for actionable
phrasing AT THE GENERALIZABLE LEVEL — tier, mask_ratio bucket, signed-error
direction, query/threshold pattern. AVOID literal plant names in observations
(see "Per-plant variance (anonymized)" above). Examples:
  "Small-tier mean Dice=0.18 (n=33) vs moderate 0.42 (n=27) — small lesions are 2.3x worse; try lower threshold or tighter NMS"
  "Mean signed_bb_error=+4.2 across all tasks → systematic over-detection; try higher threshold or add clip_select filtering"
  "10 worst tasks all have mask_ratio < 0.03 (tiny lesions); current min_norm_area=0.001 may exclude them — try lower"
  "Worst-scoring plant cohort (n=5) mean Dice=0.12 vs cohort mean 0.34 — large variance across species; consider generic query, not species-specific"

Cap each analyzer at 8 observations (keep them high-signal).

## Output format

Respond using ONLY this XML format (no text outside the tags). Repeat the
<analyzer>...</analyzer> block {k} time(s) — one per investigative focus:

<analyzers>
  <analyzer>
    <code>
    ```python
    def analyze(per_image_results):
        observations = []
        # Group by TIER (generalizable across plants — small vs moderate is
        # the same concept across train and val/test, unlike plant species
        # which are disjoint between splits).
        by_tier = {{}}
        for r in per_image_results:
            by_tier.setdefault(r['tier'], []).append(r)
        for tier in ('small', 'moderate'):
            rs = by_tier.get(tier, [])
            if not rs:
                continue
            m_dice = sum(r['dice_mask'] for r in rs) / len(rs)
            m_bb   = sum(r['signed_bb_error']   for r in rs) / len(rs)
            m_area = sum(r['signed_area_error'] for r in rs) / len(rs)
            observations.append(
                f"Tier '{{tier}}' (n={{len(rs)}}): mean Dice={{m_dice:.3f}}, "
                f"signed_bb={{m_bb:+.1f}}, signed_area={{m_area:+.3f}}")
        return observations
    ```
    </code>
    <purpose>One sentence: what failure mode or scenario this analyzer investigates</purpose>
  </analyzer>
</analyzers>

Notes on f-string syntax in the code you emit:
- The example f-strings above show SINGLE-brace placeholders — that's the
  correct pattern. COPY IT VERBATIM into your analyzer code. Do NOT add
  an extra layer of braces; doubled braces inside an f-string produce
  literal `{{name}}` text (no substitution), and the downstream code-gen
  LLM would read uninterpreted templates as feedback instead of real
  numbers.
"""
