"""vision_agent.py — Concrete agent for VTOS.

Wires VisionSearchEngine to an LLM client from tools.llm_interface.

Usage:
    from tools.llm_interface import get_llm_client
    from vtos.vision_agent import VisionAgentLLM

    agent = VisionAgentLLM(evaluator=..., llm=get_llm_client("openrouter", model),
                           workspace_dir="./results", k=3, n_iterations=5)
    agent.run()
"""

from __future__ import annotations

from typing import Optional

from tools.llm_interface import LLMInterface

from .evaluator import VisionEvaluator, DummyEvaluator
from .search_engine import SYSTEM_PROMPT, VisionSearchEngine


# ── Default problem context (from Ref/code_agent.py) ─────────────────────────

DEFAULT_PROBLEM_DESCRIPTION = """\
Zero-shot, open-vocabulary object counting in images.

Given an image and a natural-language description of the target object class,
produce a list of bounding boxes (format [[x, y, w, h], ...]) for all instances
of that object in the image. The predicted count is len(bboxes).

The task covers a wide range of difficulty scenarios:
- Small objects: targets that occupy less than 1% of image area
- High-density / crowded scenes: many tightly packed instances
- Occlusion: partially hidden objects
- Low-light / night conditions
- Compound descriptions (e.g. "red car. sedan. vehicle") for better recall

The primary metric is Mean Absolute Error (MAE) between predicted and ground-truth
counts, aggregated across all test images. Lower MAE = better. The normalized
score exposed to the search engine is (1 - MAE / max_MAE), higher = better.
"""

DEFAULT_LIBRARY_DOC = """\
The `toolbox` instance is automatically available in the execution environment.
Do NOT import or instantiate ToolboxWrapper — just use `toolbox` directly.
The variable `image_path` (str) and `text_query` (str) are also pre-populated.

### Detection
- toolbox.grounding_dino_detect(image_path, text_query=<str>, box_threshold=0.15)
    -> List[[x, y, w, h]]
  Open-vocabulary detector using GroundingDINO. Lower box_threshold = higher recall
  but more false positives. Accepts keyword alias query= for text_query=.

- toolbox.slice_and_detect(image_path, text_query=<str>, grid=(2,2), box_threshold=0.20)
    -> List[[x, y, w, h]]
  Divides the image into a grid, runs grounding_dino_detect on each tile, then
  merges results back into full-image coordinates. Best for microscopic or
  dense scenes. grid=(3,3) is the gold standard for extreme density.
  Accepts keyword alias query= for text_query=.

### Refinement & Postprocessing
- toolbox.sam_refine_bboxes(image_path, bboxes) -> List[[x, y, w, h]]
  Runs SAM on each candidate bbox to clean up overlaps and return a refined list.
  Use after detection; especially useful when boxes fragment a single object.

- toolbox.nms_filter(bboxes, iou_threshold=0.5) -> List[[x, y, w, h]]
  Standard Non-Maximum Suppression. Use to remove duplicate detections.
  Lower iou_threshold = more aggressive merging.

### Standard Libraries (always available)
- import os, numpy as np, cv2
- from PIL import Image
- import scipy, skimage  (for clustering, morphology, etc.)

### Output Contract
The final result must be assigned to the variable `bboxes`:
    bboxes = [[x, y, w, h], ...]   # one entry per detected instance
The evaluator counts instances as len(bboxes). Do NOT assign a numeric `result`.
"""


# ── Concrete agent ────────────────────────────────────────────────────────────

class VisionAgentLLM(VisionSearchEngine):
    """VTOS agent backed by an LLM client (see tools.llm_interface)."""

    def __init__(
        self,
        evaluator: Optional[VisionEvaluator] = None,
        problem_description: str = DEFAULT_PROBLEM_DESCRIPTION,
        library_doc: str = DEFAULT_LIBRARY_DOC,
        workspace_dir: str = "./vision_search_workspace",
        k: int = 3,
        k_analyzer: int = 2,
        n_iterations: int = 10,
        run_analyzers: bool = True,
        propose_analyzers: bool = True,
        llm: Optional[LLMInterface] = None,
        task_spec: Optional[str] = None,
        analyzer_task_spec: Optional[str] = None,
        baseline_code: Optional[str] = None,
        task_family: str = "counting",
    ):
        if evaluator is None:
            evaluator = DummyEvaluator()
        self.llm = llm
        super().__init__(
            problem_description=problem_description,
            evaluator=evaluator,
            workspace_dir=workspace_dir,
            k=k,
            k_analyzer=k_analyzer,
            n_iterations=n_iterations,
            run_analyzers=run_analyzers,
            propose_analyzers=propose_analyzers,
            library_doc=library_doc,
            task_spec=task_spec,
            analyzer_task_spec=analyzer_task_spec,
            baseline_code=baseline_code,
            task_family=task_family,
        )

    def _propose_solutions(self, prompt: str) -> str:
        import time
        t0 = time.time()
        prompt_tokens_est = len(prompt) // 4  # rough estimate: 4 chars per token
        result = self.llm.complete(user=prompt, system=SYSTEM_PROMPT)
        elapsed = time.time() - t0
        result_tokens_est = len(result) // 4
        print(f"  ⏱ LLM propose call: {elapsed:.1f}s | prompt~{prompt_tokens_est}tok | response~{result_tokens_est}tok")
        _check_llm_response_or_die(result, "propose_solutions")
        return result

    def _update_thoughts(self, prompt: str) -> str:
        import time
        t0 = time.time()
        prompt_tokens_est = len(prompt) // 4
        result = self.llm.complete(user=prompt, system=SYSTEM_PROMPT)
        elapsed = time.time() - t0
        result_tokens_est = len(result) // 4
        print(f"  ⏱ LLM thoughts call: {elapsed:.1f}s | prompt~{prompt_tokens_est}tok | response~{result_tokens_est}tok")
        _check_llm_response_or_die(result, "update_thoughts")
        return result


# ── LLM-error guard: detect API failures and crash loudly ─────────────────────
# Background: when OpenRouter / Anthropic credits are exhausted (or any other
# upstream failure), the LLM client returns the error text as a regular string
# (e.g. "Error calling OpenRouter API: 402 - Insufficient credits …").  The
# downstream code in search_engine doesn't crash — it just parses zero solutions
# and runs an empty iteration, silently burning through the rest of n_iter.
# This guard makes that fail loud so the user notices immediately.

class LLMServiceError(RuntimeError):
    """Raised when the LLM returns an error string or an obviously-empty response."""


# Substrings that indicate the upstream API returned an error rather than content
_LLM_ERROR_MARKERS = (
    "Error calling OpenRouter API",
    "Error calling OpenAI API",
    "Error calling Anthropic API",
    "Error calling Poe API",
    "Error code: 401",   # auth
    "Error code: 402",   # insufficient credits
    "Error code: 403",   # forbidden
    "Error code: 429",   # rate limit
    "Error code: 500",   # provider failure
    "Error code: 502",
    "Error code: 503",
    "Insufficient credits",
)

# Responses below this character length are almost certainly errors or empty
# completions, not real model output.  A valid solution / thoughts response
# would include XML tags + code → hundreds of chars at minimum.
_LLM_MIN_RESPONSE_CHARS = 200


def _check_llm_response_or_die(response: str, call_kind: str) -> None:
    """Inspect an LLM response; raise LLMServiceError on detected failure.

    Detection rules:
      1. Response contains any known upstream error marker (e.g. "Error code: 402").
      2. Response is shorter than `_LLM_MIN_RESPONSE_CHARS` (~50 tokens), which
         is below the minimum viable size for any valid propose/thoughts output.

    Raises LLMServiceError with a human-readable message including a prompt to
    top up credits / check the API; the search loop will surface this as an
    abort rather than silently continuing.
    """
    if not isinstance(response, str):
        return  # let downstream parsers handle non-string returns
    # 1. Known error markers
    lowered_check = response[:400]  # only look at the prefix; error msgs are short
    for marker in _LLM_ERROR_MARKERS:
        if marker.lower() in lowered_check.lower():
            msg = (
                f"\n\n{'='*70}\n"
                f"❌ LLM API ERROR detected during {call_kind!r}.\n"
                f"   Raw response prefix: {response[:200]!r}\n"
                f"\n"
                f"   Likely causes: credit exhaustion (top up at\n"
                f"   https://openrouter.ai/settings/credits), expired key, or\n"
                f"   upstream provider outage / rate limit.\n"
                f"\n"
                f"   Aborting the search loop to avoid recording garbage iterations.\n"
                f"   Restart with --resume after fixing the API.\n"
                f"{'='*70}\n"
            )
            print(msg)
            raise LLMServiceError(f"LLM API failure during {call_kind}: {marker}")
    # 2. Suspiciously short response (likely empty completion or error stub)
    if len(response) < _LLM_MIN_RESPONSE_CHARS:
        msg = (
            f"\n\n{'='*70}\n"
            f"❌ LLM returned suspiciously short response during {call_kind!r}\n"
            f"   ({len(response)} chars; expected >= {_LLM_MIN_RESPONSE_CHARS}).\n"
            f"   Raw: {response!r}\n"
            f"\n"
            f"   This usually indicates an API error returned as a string,\n"
            f"   or a credit / rate-limit issue.\n"
            f"\n"
            f"   Aborting to avoid recording empty iterations.\n"
            f"   Restart with --resume after fixing the API.\n"
            f"{'='*70}\n"
        )
        print(msg)
        raise LLMServiceError(
            f"LLM returned {len(response)} chars during {call_kind} "
            f"(below threshold {_LLM_MIN_RESPONSE_CHARS})."
        )
