from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VisionEvaluator(ABC):
    """Interface for evaluating a generated vision algorithm.

    evaluate() must return a dict containing:
      - 'score'     : float, higher is better (primary ranking key)
      - '_per_image': List[Dict], per-image results used by the observer (optional but
                      required for evaluate_analyzer to do real work). Each dict must have
                      at minimum: image_path, pred_bboxes, gt_bboxes, ae.

    evaluate_analyzer() runs an analyzer against pre-computed per-image results from
    evaluate() and returns a list of observation strings. The default returns []; override
    when a real execution environment is available.
    """

    @abstractmethod
    def evaluate(self, code: str) -> Dict[str, Any]:
        ...

    def evaluate_analyzer(
        self, analyzer_code: str, per_image_results: List[Dict]
    ) -> List[str]:
        """Run analyzer code against per-image prediction results.

        analyzer_code must define:
            analyze(per_image_results: List[Dict]) -> List[str]
        where each dict in per_image_results contains at minimum:
            image_path  : str
            pred_bboxes : List[[x, y, w, h]]
            gt_bboxes   : List[[x, y, w, h]]
            ae          : int  (|len(pred_bboxes) - len(gt_bboxes)|)
        """
        return []


# ── Dummy evaluator ───────────────────────────────────────────────────────────

class DummyEvaluator(VisionEvaluator):
    """Placeholder evaluator for development and testing.

    evaluate()          — returns random aggregate metrics + synthetic per-image results.
    evaluate_analyzer() — returns canned observations (no real code is executed).

    Replace with SubprocessVisionEvaluator or a custom subclass once a real benchmark
    is available. The per-image structure produced here matches what a real evaluator
    should return so analyzer code can be written and tested against it.
    """

    SCENARIO_KEYS = ["small_objects", "crowded", "occluded", "night"]
    N_IMAGES = 10

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def evaluate(self, code: str) -> Dict[str, Any]:
        score = round(self._rng.uniform(0.3, 0.9), 4)
        metrics: Dict[str, Any] = {"score": score}
        for key in self.SCENARIO_KEYS:
            metrics[key] = round(self._rng.uniform(0.1, 0.95), 4)

        # Synthetic per-image results — enough structure for analyzer code to work with
        per_image = []
        for i in range(self.N_IMAGES):
            gt_count = self._rng.randint(1, 15)
            pred_count = max(0, gt_count + self._rng.randint(-3, 3))
            gt_bboxes = [[self._rng.randint(0, 400), self._rng.randint(0, 300),
                          self._rng.randint(10, 100), self._rng.randint(10, 100)]
                         for _ in range(gt_count)]
            pred_bboxes = [[self._rng.randint(0, 400), self._rng.randint(0, 300),
                            self._rng.randint(10, 100), self._rng.randint(10, 100)]
                           for _ in range(pred_count)]
            per_image.append({
                "image_path": f"test_images/img_{i:03d}.jpg",
                "pred_bboxes": pred_bboxes,
                "gt_bboxes": gt_bboxes,
                "ae": abs(pred_count - gt_count),
                "image_width": 640,
                "image_height": 480,
            })
        metrics["_per_image"] = per_image
        return metrics

    def evaluate_analyzer(
        self, analyzer_code: str, per_image_results: List[Dict]
    ) -> List[str]:
        return [
            "[DUMMY] No real analyzer execution performed.",
            f"[DUMMY] Received {len(per_image_results)} per-image result(s).",
        ]


# ── Subprocess-based evaluator ────────────────────────────────────────────────

