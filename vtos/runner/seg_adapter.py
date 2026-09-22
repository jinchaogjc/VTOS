"""Seg-task adapter for the VTOS orchestrator.

Pure-data module: loads PlantSeg v2 splits, builds workspace paths, exports
the task-spec strings used by the orchestrator when --task seg.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import List, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

SEG_DATA_ROOT = Path(_ROOT_DIR) / "data" / "tasklets" / "plantseg_ood"
SEG_IMAGE_DIR = SEG_DATA_ROOT / "images"
_WORKSPACE_ROOT = Path(_ROOT_DIR) / "logs" / "plantseg_vtos"


def load_split(split: str) -> List[Dict]:
    """Load a PlantSeg v2 split. Returns list of task dicts with bounding_boxes,
    segmentations, mask_ratio, plant, target_class, image_path, task_id.
    'val' is the held-out 20-task selection set — never seen by the search
    loop (the train sample is benchmark_train.json), used for picking the
    best candidate before scoring on test."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    path = SEG_DATA_ROOT / f"benchmark_{split}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def workspace_dir(exp_id: str, baseline: str = "vtos", stamp: str = "") -> Path:
    """Return the workspace path for a seg VTOS run."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    return _WORKSPACE_ROOT / f"{exp_id}_{baseline}_{stamp}"
