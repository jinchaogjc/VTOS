# vtos/constants.py — single source of truth for all tier and run config values.
# Import from here everywhere; never hard-code these numbers in other files.
from __future__ import annotations

# ── Density tiers ─────────────────────────────────────────────────────────────
# Normal: 10–50 objects (moderate spatial overlap, non-trivial detection)
# Extreme: 51–100 objects (severe crowding, main contribution regime)
# Sparse tier (1–9) is excluded — trivially solved by all baselines.
TIER_BOUNDS: dict[str, tuple[int, int]] = {
    "normal":  (10, 50),
    "extreme": (51, 100),
}

# Maps the expert training slot name → canonical tier name in TIER_BOUNDS
EXPERT_TO_TIER: dict[str, str] = {
    "medium": "normal",
    "hard":   "extreme",
}

TIER_ORDER   = ["normal", "extreme"]   # ascending density
def count_to_tier(n: int) -> str:
    """Return the tier name for a ground-truth object count."""
    for tier, (lo, hi) in TIER_BOUNDS.items():
        if lo <= n <= hi:
            return tier
    raise ValueError(f"count {n} outside any defined tier (bounds: {TIER_BOUNDS})")


# ── PlantSeg segmentation tiers ───────────────────────────────────────────────
# The PlantSeg grounded-segmentation task classifies each image by mask_ratio
# (fraction of image pixels covered by the GT disease mask): "small" = a small
# diseased region (hard to localise), "moderate" = a larger region. This
# threshold is the SINGLE SOURCE — import seg_tier(); never hard-code 0.10
# anywhere else.
#
# Renamed 2026-05-23: previously "extreme" / "normal". The new labels are
# descriptive of lesion size (matching COCO-style small/medium/large
# convention) rather than editorial. Historical records / experiment files
# (records.json, summary.json, exp_results.md entries pre-2026-05-23) still
# contain "extreme" / "normal" — those are immutable snapshots, untouched.
SEG_SMALL_MASK_RATIO = 0.10


def seg_tier(mask_ratio: float) -> str:
    """Return the PlantSeg difficulty tier for a ground-truth mask_ratio.

    small    : mask_ratio <  SEG_SMALL_MASK_RATIO  (small diseased region)
    moderate : mask_ratio >= SEG_SMALL_MASK_RATIO
    """
    return "small" if mask_ratio < SEG_SMALL_MASK_RATIO else "moderate"


# ── Dataset split (v2: 30-category disjoint, 180 images) ──────────────────────
# images 001–060  → train  (15 categories, 30N + 30E = 60; 2N+2E per cat)
# images 061–080  → val    ( 5 categories, 10N + 10E = 20; 2N+2E per cat)
# images 081–180  → test   (10 categories, 50N + 50E = 100; 5N+5E per cat)
TRAIN_START = 1
TRAIN_END   = 60
VAL_START   = 61
VAL_END     = 80
TEST_START  = 81
TEST_END    = 180
N_TRAIN     = TRAIN_END - TRAIN_START + 1   # = 60
N_VAL       = VAL_END   - VAL_START   + 1   # = 20
N_TEST      = TEST_END  - TEST_START  + 1   # = 100

# ── Search loop ────────────────────────────────────────────────────────────────
N_ITER      = 10   # number of VTOS iterations per expert

# ── DINO threshold defaults ────────────────────────────────────────────────────
# BASELINE_BOX_THRESHOLD is the seed for sol_000_baseline in VTOS search:
# 0.10 was selected on the validation set as the point_f1-optimal threshold for
# plain GroundingDINO (see e003/e004/e005/e006 sweep, e007 test confirmation).
# VTOS therefore starts from our strongest non-search DINO baseline and must
# IMPROVE from there — not rediscover a known-good threshold.
BASELINE_BOX_THRESHOLD = 0.10
REJECTION_THRESHOLD  = 0.60  # solutions below this point_f1 score are rejected

# ── Val selection ──────────────────────────────────────────────────────────────
# A search solution is only chosen over baseline if it beats baseline by at least
# this margin on the held-out val set. Prevents overfitting to unrepresentative
# val images (e.g., a single GT=5 sparse image inflating multi-query scores).
MIN_VAL_IMPROVEMENT  = 0.02  # solution must beat baseline by ≥0.02 on val to be selected
