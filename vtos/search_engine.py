from __future__ import annotations

import json
import os
import re
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .data_structures import (
    AnalyzerSnapshot,
    HypothesisEntry,
    IterationRecord,
    VisionSearchState,
    VisionSolutionSnapshot,
    VisionThoughts,
)
from .constants import BASELINE_BOX_THRESHOLD
from .evaluator import VisionEvaluator
from .html_reporter import HTMLReporter


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return textwrap.dedent(code).strip()


def _parse_json_or_lines(raw: str) -> list:
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    items = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*[0123456789.]").strip().rstrip("]").strip()
        if line:
            items.append(line)
    return items


def _read_code(workspace_dir: str, code_file: str, max_chars: int = 1500) -> str:
    path = os.path.join(workspace_dir, code_file)
    if not os.path.exists(path):
        return "(file not found)"
    with open(path, encoding="utf-8") as f:
        code = f.read()
    if len(code) > max_chars:
        code = code[:max_chars] + "\n... (truncated)"
    return code


# ── Response parsers ──────────────────────────────────────────────────────────

def parse_algorithm_solutions(text: str) -> List[Tuple[str, str]]:
    """Return list of (code, key_differences) from LLM response."""
    if text.strip().startswith("Error:") or '"error":' in text.lower():
        return []

    solutions: List[Tuple[str, str]] = []
    raw_solutions = re.findall(r"<solution[^>]*>(.*?)</solution>", text, re.DOTALL)
    if not raw_solutions:
        code = _extract_code_block(text)
        if code and not (code.startswith("Error:") or '"error":' in code.lower()):
            solutions.append((code, "Single proposed solution"))
        return solutions
    for sol_text in raw_solutions:
        code_raw = _extract_tag(sol_text, "code") or sol_text
        code = _extract_code_block(code_raw)
        key_diffs = _extract_tag(sol_text, "key_differences") or "Not specified"
        if code and not (code.startswith("Error:") or '"error":' in code.lower()):
            solutions.append((code, key_diffs))
    return solutions


def parse_analyzers(text: str) -> List[Tuple[str, str]]:
    """Return list of (code, purpose) from LLM response."""
    analyzers = []
    raw = re.findall(r"<analyzer[^>]*>(.*?)</analyzer>", text, re.DOTALL)
    for ana_text in raw:
        code_raw = _extract_tag(ana_text, "code") or ana_text
        code = _extract_code_block(code_raw)
        purpose = _extract_tag(ana_text, "purpose") or "Not specified"
        if code:
            analyzers.append((code, purpose))
    return analyzers


def parse_thoughts(text: str) -> Optional[VisionThoughts]:
    """Extract a VisionThoughts object from LLM response."""
    thoughts_raw = _extract_tag(text, "thoughts")
    if not thoughts_raw:
        return None

    insights = _extract_tag(thoughts_raw, "insights") or ""
    analyzer_insights = _extract_tag(thoughts_raw, "analyzer_insights") or ""
    key_problems = _extract_tag(thoughts_raw, "key_problems") or ""
    status = _extract_tag(thoughts_raw, "exploration_status") or ""

    hypotheses: List[HypothesisEntry] = []
    hyp_raw = _extract_tag(thoughts_raw, "hypotheses") or ""
    if hyp_raw:
        try:
            hyp_list = json.loads(hyp_raw)
            if isinstance(hyp_list, list):
                hypotheses = [HypothesisEntry.from_dict(h) for h in hyp_list]
        except json.JSONDecodeError:
            for line in _parse_json_or_lines(hyp_raw):
                if isinstance(line, str) and len(line) > 5:
                    hypotheses.append(HypothesisEntry(hypothesis=line))

    directions: list = []
    dirs_raw = _extract_tag(thoughts_raw, "exploration_directions") or ""
    if dirs_raw:
        try:
            parsed = json.loads(dirs_raw)
            if isinstance(parsed, list):
                directions = parsed
        except json.JSONDecodeError:
            for line in dirs_raw.splitlines():
                line = line.strip().lstrip("-•*").strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                directions.append({
                    "direction": parts[0] if parts else line,
                    "status": parts[1] if len(parts) > 1 else "unknown",
                    "performance": parts[2] if len(parts) > 2 else "unknown",
                })

    return VisionThoughts(
        insights=insights,
        hypotheses=hypotheses,
        analyzer_insights=analyzer_insights,
        exploration_directions=directions,
        key_problems=key_problems,
        exploration_status=status,
    )


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert vision algorithm designer running an automated search for the best solution.
    Think carefully about the problem, existing solutions, observations, and current insights
    before proposing new algorithms or analyzers.
    Always respond in the exact XML format requested — no prose outside the tags.
""")


_CONTEXT_WINDOW = 3  # number of most-recent solutions to show in full


def _score_mode_key_metric_text() -> str:
    """Build the score-mode-aware 'Key Metric' line for propose-prompts.

    Returns a single sentence describing the optimisation target, with no
    point_f1-specific framing leaking into non-point_f1 modes. Mode is read from the
    VTOS_SCORE_MODE env var (set by orchestrator at run start).
    """
    sm = os.environ.get("VTOS_SCORE_MODE", "point_f1")
    if sm == "point_f1":
        return ("Optimise point_f1 = F1 with point-in-bbox hit criterion (higher = better; "
                "lenient — rewards any prediction whose centre lies in a GT box)")
    if sm == "dual_rank":
        return ("Optimise BOTH mIoU↑ AND MAE↓ jointly. The ranking uses "
                "Borda rank-sum over the two metrics (lower RankSum = better on both)")
    if sm == "dual_rank_seg":
        return ("Ranked by dual-rank over Dice^mask (↑) and mIoU^mask (↑) — "
                "both matter.")
    return f"Optimise (unknown score mode '{sm}')"


def _fmt_metrics(perf: dict) -> str:
    """Score-mode-aware one-line metric string for the search log.

    'dual_rank_seg' is ranked by Dice^mask and mIoU^mask, so show those — the
    scalar (dice+miou)/2 `score` is a display convenience, NOT the ranking key.
    Other modes keep the legacy `score=… mae=…` form.
    """
    sm = os.environ.get("VTOS_SCORE_MODE", "point_f1")
    if sm == "dual_rank_seg":
        d = perf.get("dice_mask")
        m = perf.get("miou_mask")
        if isinstance(d, (int, float)) and isinstance(m, (int, float)):
            return f"dice={d:.4f} miou={m:.4f}"
    sv = perf.get("score", "?")
    sv = f"{sv:.4g}" if isinstance(sv, (int, float)) else str(sv)
    return f"score={sv} mae={perf.get('mae', '?')}"


def build_propose_prompt(
    problem_description: str,
    state: VisionSearchState,
    k: int,
    workspace_dir: str,
    library_doc: Optional[str] = None,
    task_spec: Optional[str] = None,
    task_family: str = "counting",
) -> str:
    """... task_family: 'counting' (default) preserves the existing counting
    hint corpus. 'seg' suppresses all counting-tool examples (Sections A/B/C
    of the engine's prompt construction) so the LLM only sees the caller's
    seg-specific task_spec — no toolbox.X confusion."""
    ranked = state.ranked_snapshots()
    failed = [s for s in state.algorithm_snapshots if s.rank is None]

    lines = [f"## Problem\n{problem_description}\n"]

    if library_doc and task_family != "seg":
        # The default counting library_doc (`DEFAULT_LIBRARY_DOC` in vision_agent.py)
        # documents `toolbox.grounding_dino_detect`, `toolbox.nms_filter`,
        # `toolbox.slice_and_detect` etc. — the entire counting toolbox API.
        # Showing this to seg runs LITERALLY teaches the LLM the wrong API,
        # which is the root cause of the `toolbox.X(...)` hallucination
        # observed in psv2_010 (11 zero-score sols). Seg runs use a different
        # sandbox (bare `detect()`/`segment()` closures, NOT a `toolbox`
        # namespace), so the entire library_doc section is omitted.
        #
        lines.append(f"## Available Library Functions\n{library_doc}\n")

    lines.append(f"## Current Knowledge\n{state.thoughts.format()}\n")

    if ranked:
        # Sliding window: always show best (rank 1) + last CONTEXT_WINDOW solutions in full.
        # Older solutions shown as one-line summaries to keep context bounded.
        all_snaps = [s for s in state.algorithm_snapshots if s.rank is not None]
        recent_snaps = all_snaps[-_CONTEXT_WINDOW:]  # last K by insertion order
        best_snap = ranked[0]

        full_show = {best_snap.id}
        for s in recent_snaps:
            full_show.add(s.id)

        summary_snaps = [s for s in ranked if s.id not in full_show]

        lines.append("## Existing Solutions")
        if summary_snaps:
            lines.append("### History (summary — older solutions)")
            for snap in summary_snaps:
                diffs = (snap.key_differences or "")[:80]
                # Display format depends on score mode
                _sm = os.environ.get("VTOS_SCORE_MODE", "point_f1")
                if _sm == "dual_rank":
                    rs = snap.scenario_metrics.get('dual_rank_sum', '?')
                    mir = snap.scenario_metrics.get('miou_rank', '?')
                    mar = snap.scenario_metrics.get('mae_rank', '?')
                    lines.append(f"  - {snap.id}: RankSum={rs} (mIoU r={mir}, MAE r={mar}) | {diffs}")
                elif _sm == "dual_rank_seg":
                    rs = snap.scenario_metrics.get('dual_rank_sum', '?')
                    dir_ = snap.scenario_metrics.get('dice_rank', '?')
                    mir = snap.scenario_metrics.get('miou_rank', '?')
                    lines.append(f"  - {snap.id}: RankSum={rs} (Dice r={dir_}, mIoU r={mir}) | {diffs}")
                else:
                    lines.append(f"  - {snap.id}: point_f1={snap.score:.4f} | {diffs}")
            lines.append("")

        # Seed (sol_000_*) was being shown only during iter 1, then dropped
        # from recent_snaps as new proposals pushed it past _CONTEXT_WINDOW.
        # Always include it as a permanent anchor — the LLM is supposed to
        # build ON the seed, so it needs to keep seeing the seed's code.
        seed_snap = next(
            (s for s in state.algorithm_snapshots if s.id.startswith('sol_000_')),
            None,
        )
        lines.append("### Recent Experiments (full code — seed + last 3 + best)")
        shown_ids = set()
        show_order = ([seed_snap] if seed_snap is not None else []) \
                     + [best_snap] + recent_snaps
        for snap in show_order:
            if snap.id in shown_ids:
                continue
            shown_ids.add(snap.id)
            if snap.id.startswith('sol_000_'):
                label = " 🌱 SEED (always shown — refine vs this baseline)"
            elif snap.id == best_snap.id:
                label = " ⭐ BEST"
            else:
                label = ""
            code_text = _read_code(workspace_dir, snap.code_file)
            # Defense against the self-reinforcing counting-API hallucination
            # loop in seg-mode runs: once an iter emits `toolbox.X(...)` (a
            # counting-only API the seg sandbox doesn't have), its code gets
            # stored AND rendered to future iters as "past attempts" — the LLM
            # then copies the pattern, perpetuating zero-score candidates.
            # When task_family=seg, replace such code with a clear rejection
            # marker so the next LLM learns NOT to retry the pattern.
            if task_family == "seg" and re.search(r"\btoolbox\.\w+\s*\(", code_text):
                lines.append(f"\n#### {snap.format_brief()}{label}")
                lines.append(
                    "```\n[REJECTED — this past attempt used the counting-only "
                    "`toolbox.X(...)` API which does not exist in the seg "
                    "sandbox. Do NOT retry this pattern. Use the bare-name "
                    "functions documented in the Task section instead.]\n```")
                continue
            lines.append(f"\n#### {snap.format_brief()}{label}")
            if snap.analyzer_observations:
                obs_str = "\n".join(f"  - {o}" for o in snap.analyzer_observations[:5])
                lines.append(f"Analyzer observations:\n{obs_str}")
            lines.append(f"```python\n{code_text}\n```")

        if failed:
            lines.append(f"\n{len(failed)} failed solution(s) omitted.")
    else:
        lines.append("## Existing Solutions\nNone yet — this is the first iteration.")

    if task_spec:
        # Task-spec override: a caller-supplied `## Task` block (e.g. the
        # segmentation task routed through this engine). The literal {k} token
        # is substituted with the per-iteration proposal count. Counting is
        # unaffected — when task_spec is None the original block below runs.
        lines.append("\n## Task\n" + task_spec.replace("{k}", str(k)))
    else:
        lines.append(f"""
## Task
Propose exactly {k} new, diverse, improved algorithm solution(s).
All proposals must stay within the proven improvement directions listed below.
{"Use only functions listed in the Available Library Functions section." if library_doc else ""}
- **Starting exemplar (one possibility — explore widely)**: a low DINO threshold + post-hoc NMS often wins. Here is *one* such pattern; treat it as a hint, not a recipe:
  ```python
  import numpy as np
  bboxes = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold=0.20)
  bboxes = toolbox.nms_filter(bboxes, iou_threshold=0.40)
  ```
- **Exploration directions** (try several; the search will tell you which ones generalize):
  1. Threshold sweep: `box_threshold` ∈ [0.10, 0.40]. Lower → higher recall + duplicates; higher → tighter precision.
  2. NMS strength: `iou_threshold` ∈ [0.20, 0.55]. Lower → drops more overlaps; higher → keeps duplicates.
  3. Density-adaptive post-processing: detect high-density regions (e.g., grid-based cell counts) and apply tighter NMS only there.
  4. Multi-threshold union: union of two or three detect() calls, deduped with NMS.
  5. Tiling: `toolbox.slice_and_detect(image_path, text_query, grid=(R, C))` helps in extreme-density scenes (51-100 objects) but creates duplicates on normal-density images (10-50 objects); use selectively.
- **Note about queries**: keep the user-provided `text_query` literal; synonym expansion has historically hurt precision on lower-density scenes.

- **Known sharp edges** (do NOT use unless you have strong evidence on val):
  - `toolbox.sam_refine_bboxes` — degraded val performance in prior runs.
  - Aggressive area / size filters — fail catastrophically on diverse object scales.
  - Running DINO at 4+ thresholds and unioning everything — typically adds more FPs than TPs.
- **BBox Format**: All bboxes are `[x1, y1, x2, y2]` in normalized [0, 1] coords.
- **Key Metric**: {_score_mode_key_metric_text()}. Simple solutions that generalize beat complex ones that overfit.

Respond using ONLY this XML format (no text outside the tags):

<solutions>
  <solution>
    <code>
    ```python
    # complete, self-contained Python code
    ```
    </code>
    <key_differences>One or two sentences: what makes this solution novel vs. existing ones</key_differences>
  </solution>
</solutions>

Repeat the <solution>...</solution> block {k} time(s).
""")

    return "\n".join(lines)


def build_analyzer_prompt(
    problem_description: str,
    state: VisionSearchState,
    k_analyzer: int,
    workspace_dir: str,
    library_doc: Optional[str] = None,
    analyzer_task_spec: Optional[str] = None,
) -> str:
    ranked = state.ranked_snapshots()

    lines = [f"## Problem\n{problem_description}\n"]

    if library_doc:
        lines.append(f"## Available Library Functions\n{library_doc}\n")

    if ranked:
        lines.append("## Current Best Algorithm Solutions")
        for snap in ranked[:3]:
            lines.append(f"\n### {snap.format_brief()}")
            if snap.scenario_metrics:
                metrics_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in snap.scenario_metrics.items())
                lines.append(f"Scenario metrics: {metrics_str}")
            lines.append(f"```python\n{_read_code(workspace_dir, snap.code_file)}\n```")
    else:
        lines.append("## Current Best Algorithm Solutions\nNone evaluated yet.")

    if state.analyzer_snapshots:
        lines.append("\n## Existing Analyzers")
        for ana in state.analyzer_snapshots:
            status = "ERROR" if ana.error else "OK"
            lines.append(f"\n### {ana.id} [{status}] | purpose: {ana.purpose}")
            if ana.observations:
                obs_str = "\n".join(f"  - {o}" for o in ana.observations[:5])
                lines.append(f"Observations:\n{obs_str}")
    else:
        lines.append("\n## Existing Analyzers\nNone yet.")

    task_section = textwrap.dedent(f"""
        ## Insight for Building Analyzer
        {state.thoughts.analyzer_insights or '(none yet)'}

        ## Task
        Propose {k_analyzer} new or improved analyzer program(s).
        An analyzer investigates how and where the current best algorithms fail.
        It must define a function with this exact signature:

            def analyze(per_image_results: list) -> list:
                ...

        Each element of per_image_results is a dict with these keys:
            image_path   : str   — path to the image file
            pred_bboxes  : list  — predicted boxes [[x1, y1, x2, y2], ...] (normalized)
            gt_bboxes    : list  — ground-truth boxes [[gx, gy, gw, gh], ...] (normalized)
            hits         : int   — number of predicted boxes that spatially hit a GT bbox
            false_pos    : int   — number of predicted boxes that are 'blind counts'
            ae           : int   — absolute error
            image_width  : int
            image_height : int

        ## Guidelines for Writing Analyzer Code
        - **No External Functions**: Do NOT use functions that you haven't defined in the block.
        - **IOU Utility**: Use `toolbox.compute_iou(box1, box2)` (where both are [x1, y1, x2, y2]) if you need to calculate overlap. Note: you may need to convert `gt_bboxes` from `[x, y, w, h]` to `[x1, y1, x2, y2]` first.
        - **Focus**: Compare the location of `pred_bboxes` centers vs `gt_bboxes` to explain WHY count or hits are low.




        ## Slicing Cost-Benefit Estimation (REQUIRED per image)
        For each image, estimate whether a tiled/sliced detection would help using the
        following reward criterion — slicing is only worth doing if `2*dH - dFP > 0`:

            dH  = estimated additional hits from slicing (how many new GTs get covered)
            dFP = estimated additional false positives from slicing
                  (use: n_boundary_tiles_with_GT × 1.5 + n_overcrowded_tiles × 0.5)
            reward = 2 * dH - dFP   # positive = slicing helps; negative = slicing hurts

        Output per image:
            - `estimated_dh`  : float — expected new hits from slicing
            - `estimated_dfp` : float — expected new FPs from slicing
            - `should_slice`  : bool  — True only if (2 * dH - dFP) > 0

        Respond using ONLY this XML format (no text outside the tags):

        <analyzers>
          <analyzer>
            <code>
            ```python
            def analyze(per_image_results):
                observations = []
                for res in per_image_results:
                    if res['false_pos'] > 0:
                        # Use double braces to escape for f-string
                        observations.append(f"Image {{res['image_path']}} has {{res['false_pos']}} blind counts.")
                return observations
            ```
            </code>
            <purpose>One sentence: what failure mode or scenario this analyzer investigates</purpose>

          </analyzer>
        </analyzers>

        Repeat the <analyzer>...</analyzer> block {k_analyzer} time(s).
    """)
    task_section_final = task_section

    if analyzer_task_spec:
        # Task-spec override: a caller-supplied analyzer `## Task` block (e.g.
        # the segmentation task routed through this engine). It supplies the
        # task-appropriate per_image schema + analyzer guidance + response
        # format, replacing the counting-specific per_image schema and the
        # counting-only slicing cost-benefit section in `task_section` above.
        # The `## Insight
        # for Building Analyzer` block is kept (task-agnostic). The literal {k}
        # token is substituted with the per-iteration analyzer count. Counting
        # is unaffected — when analyzer_task_spec is None the block above runs.
        task_section_final = textwrap.dedent(f"""
            ## Insight for Building Analyzer
            {state.thoughts.analyzer_insights or '(none yet)'}

            ## Task
            """) + analyzer_task_spec.replace("{k}", str(k_analyzer))

    lines.append(task_section_final)
    return "\n".join(lines)


def build_update_thoughts_prompt(
    problem_description: str,
    state: VisionSearchState,
    new_algo_snapshots: List[VisionSolutionSnapshot],
    new_ana_observations: List[Tuple[str, List[str]]],
) -> str:
    new_results = "\n".join(snap.format_brief() for snap in new_algo_snapshots) or "(none)"
    top_ranked = state.ranked_snapshots()[:5]
    top_str = "\n".join(s.format_brief() for s in top_ranked) or "(none yet)"

    ana_section = ""
    if new_ana_observations:
        parts = []
        for purpose, obs in new_ana_observations:
            obs_str = "\n".join(f"  - {o}" for o in obs[:8])
            parts.append(f"Analyzer ({purpose}):\n{obs_str}")
        ana_section = "\n\n## New Analyzer Observations This Iteration\n" + "\n\n".join(parts)
    else:
        ana_section = "\n\n## New Analyzer Observations This Iteration\n(none)"

    return f"""## Problem
{problem_description}

## Previous Knowledge
{state.thoughts.format()}

## New Algorithm Results This Iteration
{new_results}
{ana_section}

## Overall Top Solutions (after re-ranking)
{top_str}

## Task
Update the knowledge base based on new evidence. **IMPORTANT — keep it compact:**
- Update insights: distil into 3–5 concise bullet points max. Remove outdated ones.
- Re-assess each hypothesis: confirm, refute, or partially support based on new results.
  Add NEW hypotheses only if genuinely novel. **Keep at most 6 hypotheses total** —
  merge related ones; drop unverified hypotheses with no supporting evidence after 2+ iters.
- Update analyzer_insights: 1–2 sentences only.
- Revise exploration directions: **keep at most 6** — drop abandoned/low-value ones.
- Identify the single most important bottleneck.
- Summarize current status in 1–2 sentences.

Respond using ONLY this XML format (no text outside the tags):

<thoughts>
  <insights>Updated insights from experimental evidence</insights>
  <hypotheses>
    [
      {{"hypothesis": "...", "status": "confirmed|refuted|partial|unverified", "evidence": "..."}},
      ...
    ]
  </hypotheses>
  <analyzer_insights>What types of analyses have been most informative for improving the algorithm</analyzer_insights>
  <exploration_directions>
    [{{"direction": "...", "status": "tried|planned|abandoned", "performance": "..."}}]
  </exploration_directions>
  <key_problems>Core bottlenecks and challenges to address for a breakthrough</key_problems>
  <exploration_status>High-level state: improving / stuck / converging; what to focus on next</exploration_status>
</thoughts>
"""


# ── Abstract search engine ────────────────────────────────────────────────────

class VisionSearchEngine(ABC):
    """Core VTOS search loop.

    Subclasses implement _propose_solutions and _update_thoughts using their
    preferred LLM backend.
    """

    def __init__(
        self,
        problem_description: str,
        evaluator: VisionEvaluator,
        workspace_dir: str = "./vision_search_workspace",
        k: int = 3,
        k_analyzer: int = 2,
        n_iterations: int = 10,
        run_analyzers: bool = True,
        propose_analyzers: bool = True,
        library_doc: Optional[str] = None,
        task_spec: Optional[str] = None,
        analyzer_task_spec: Optional[str] = None,
        baseline_code: Optional[str] = None,
        task_family: str = "counting",
    ):
        self.problem_description = problem_description
        self.evaluator = evaluator
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.k = k
        self.k_analyzer = k_analyzer
        self.n_iterations = n_iterations
        self.run_analyzers = run_analyzers
        self.propose_analyzers = propose_analyzers
        self.library_doc = library_doc
        # Optional task-spec override + seed program. When None (counting,
        # default), the engine uses its built-in counting `## Task` block and
        # the val-tuned DINO seed; when set (e.g. segmentation), they replace
        # those — see build_propose_prompt() and run()'s seeding block.
        self.task_spec = task_spec
        # Optional analyzer task-spec override. When None (counting, default),
        # build_analyzer_prompt uses its built-in counting analyzer `## Task`
        # block (counting per_image schema + slicing section); when set (e.g. segmentation), it supplies a task-aware
        # per_image schema + analyzer guidance — see build_analyzer_prompt().
        self.analyzer_task_spec = analyzer_task_spec
        self.baseline_code = baseline_code
        # Task family: "counting" (default) preserves existing behavior; "seg"
        # gates out counting-only hint sections AND filters past-iter snapshot
        # code containing `toolbox.X(...)` from the rendered prompt (breaks
        # the self-reinforcing hallucination loop observed in psv2_010).
        self.task_family = task_family

        os.makedirs(os.path.join(self.workspace_dir, "solutions"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "analyzers"), exist_ok=True)

        self.state = self._load_state()
        self.reporter = HTMLReporter(self.workspace_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> VisionSearchState:
        print(f"VTOS started. Workspace: {self.workspace_dir}")
        print(f"Running {self.n_iterations} iterations, {self.k} proposals each.\n")

        # --- SEEDING (Iteration 0 ONLY) ---
        if self.state.iteration == 0 and not self.state.algorithm_snapshots:
            print("🌱 Seeding search with a simple baseline...")
            # Seed with our val-tuned best plain DINO threshold (see vtos/constants.py).
            # VTOS must IMPROVE over our strongest non-search baseline, not rediscover it.
            # A caller may override the seed via self.baseline_code (e.g. the
            # segmentation task supplies a Grounded-SAM 2 seed); when unset the
            # counting default below is used unchanged.
            baseline_code = self.baseline_code if self.baseline_code else (
                "# Plain DINO at val-tuned best threshold (our strongest non-search baseline)\n"
                f"bboxes = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold={BASELINE_BOX_THRESHOLD})\n"
            )
            sol_id = "sol_000_baseline"
            rel_path = os.path.join("solutions", f"{sol_id}.py")
            abs_path = os.path.join(self.workspace_dir, rel_path)
            with open(abs_path, "w", encoding="utf-8") as f: f.write(baseline_code)
            try:
                perf = self.evaluator.evaluate(baseline_code)
                snap = VisionSolutionSnapshot(
                    id=sol_id, code_file=rel_path,
                    key_differences="Simple direct detection baseline",
                    performance=perf, iteration=0, timestamp=datetime.now().isoformat()
                )
                self.state.algorithm_snapshots.append(snap)
                self._rank_snapshots()
                print(f"  [Baseline] {_fmt_metrics(perf)}")
            except Exception as e: print(f"  ⚠ Baseline seeding failed: {e}")
            # Multi-seed: optionally seed extra iteration-0 paradigms (set as
            # `engine.extra_seeds` = list of (name, code)) so the search
            # recombines paradigms rather than inventing from one anchor.
            for _sname, _scode in (getattr(self, "extra_seeds", None) or []):
                _sid = f"sol_000_{_sname}"
                _srel = os.path.join("solutions", f"{_sid}.py")
                with open(os.path.join(self.workspace_dir, _srel), "w",
                          encoding="utf-8") as _sf:
                    _sf.write(_scode)
                try:
                    _sperf = self.evaluator.evaluate(_scode)
                    self.state.algorithm_snapshots.append(VisionSolutionSnapshot(
                        id=_sid, code_file=_srel,
                        key_differences=f"Seed paradigm: {_sname}",
                        performance=_sperf, iteration=0,
                        timestamp=datetime.now().isoformat()))
                    self._rank_snapshots()
                    print(f"  [Seed:{_sname}] {_fmt_metrics(_sperf)}")
                except Exception as _se:
                    print(f"  ⚠ Seed '{_sname}' failed: {_se}")
            # Advance iteration counter so LLM proposals start at iter 1 (baseline is iter 0 only)
            self.state.iteration = 1

        for _ in range(self.n_iterations):
            best = self.state.ranked_snapshots()
            if best and best[0].score >= 1.0:
                print(f"\n🎯 Perfect solution found ({best[0].id}). Early exit.")
                break

            iter_num = self.state.iteration
            # state.iteration is already 1-indexed after the baseline-seed advance
            # at line 555 (sets state.iteration = 1 so sol IDs start at sol_001_*).
            # So iter_num itself is the user-facing iteration number; don't add +1.
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}")
            print(f"{'='*60}")
            thoughts_before = VisionThoughts.from_dict(self.state.thoughts.to_dict())
            new_algo, new_ana, new_obs = self._run_iteration()

            record = IterationRecord(
                iteration=iter_num,
                thoughts_before=thoughts_before.to_dict(),
                thoughts_after=self.state.thoughts.to_dict(),
                new_algorithm_snapshots=[s.to_dict() for s in new_algo],
                new_analyzer_snapshots=[a.to_dict() for a in new_ana],
                timestamp=datetime.now().isoformat(),
            )
            self.state.history.append(record)
            self.state.iteration += 1
            self._save_state()
            self.reporter.update(self.state, self.problem_description)
            html_path = self.reporter.update(self.state, self.problem_description)
            top = self.state.ranked_snapshots()
            if top:
                _sm = os.environ.get("VTOS_SCORE_MODE", "point_f1")
                best = top[0]
                if _sm == "dual_rank":
                    rs   = best.scenario_metrics.get('dual_rank_sum', '?')
                    mir  = best.scenario_metrics.get('miou_rank', '?')
                    mar  = best.scenario_metrics.get('mae_rank', '?')
                    miou = best.performance.get('miou', best.performance.get('avg_miou', '?'))
                    mae  = best.performance.get('mae', '?')
                    miou_s = f"{miou:.4f}" if isinstance(miou, (int, float)) else str(miou)
                    mae_s  = f"{mae:.2f}"  if isinstance(mae,  (int, float)) else str(mae)
                    best_str = (f"  best=#{best.id} RankSum={rs} "
                                f"(mIoU r={mir}, MAE r={mar}; mIoU={miou_s}, MAE={mae_s})")
                elif _sm == "dual_rank_seg":
                    rs = best.scenario_metrics.get('dual_rank_sum', '?')
                    best_str = (f"  best=#{best.id} {_fmt_metrics(best.performance)} "
                                f"rank_sum={rs}")
                else:
                    score_val = best.score
                    score_str = f"{score_val:.4g}" if isinstance(score_val, (int, float)) else str(score_val)
                    best_str = f"  best score={score_str}"
            else:
                best_str = ""
            print(f"\nHTML report: {html_path}{best_str}")

        print("\nSearch complete.")
        top = self.state.ranked_snapshots()
        if top:
            print(f"Best solution: {top[0].format_brief()}")
        return self.state

    # ── Iteration logic ───────────────────────────────────────────────────────

    def _run_iteration(self):
        import time
        iter_start = time.time()

        def _elapsed(t0):
            return f"{time.time() - t0:.1f}s"

        # Step 1: Propose algorithm solutions
        t1 = time.time()
        print(f"\n[1/{self._n_steps()}] Proposing {self.k} algorithm solution(s)...")
        t1_prompt = time.time()
        propose_prompt = build_propose_prompt(
            self.problem_description, self.state, self.k,
            self.workspace_dir, self.library_doc,
            task_spec=self.task_spec,
            task_family=self.task_family,
        )
        t1_prompt_done = time.time()
        print(f"  ⏱ Step 1 prompt construction: {t1_prompt_done - t1_prompt:.2f}s | len={len(propose_prompt)} chars ~{len(propose_prompt)//4}tok")
        raw = self._propose_solutions(propose_prompt)  # LLM call timing printed inside vision_agent.py
        proposals = parse_algorithm_solutions(raw)
        if not proposals:
            print(f"  FAILED to parse any solutions. LLM raw output: {raw[:200]}...")
        else:
            print(f"  Parsed {len(proposals)} solution(s)")
        print(f"  ⏱ Step 1 total (prompt+LLM): {_elapsed(t1)}")

        # Step 2: Evaluate algorithm solutions
        t2 = time.time()
        print(f"[2/{self._n_steps()}] Evaluating solutions...")
        new_algo: List[VisionSolutionSnapshot] = []
        iter_idx = self.state.iteration

        for i, (code, key_diffs) in enumerate(proposals):
            sol_id = f"sol_{iter_idx:03d}_{i:02d}"
            rel_path = os.path.join("solutions", f"{sol_id}.py")
            abs_path = os.path.join(self.workspace_dir, rel_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(code)
            t_eval = time.time()
            try:
                perf = self.evaluator.evaluate(code)
                scenario_metrics = {
                    k: v for k, v in perf.items()
                    if k not in ("score", "_per_image")
                }
                error = None
                print(f"  [{i+1}/{len(proposals)}] {sol_id}: {_fmt_metrics(perf)} ({_elapsed(t_eval)})")
            except Exception as exc:
                perf = {"score": float("-inf")}
                scenario_metrics = {}
                error = str(exc)
                print(f"  [{i+1}/{len(proposals)}] {sol_id}: FAILED — {error[:80]}")
            snap = VisionSolutionSnapshot(
                id=sol_id,
                code_file=rel_path,
                key_differences=key_diffs,
                performance=perf,
                scenario_metrics=scenario_metrics,
                iteration=iter_idx,
                timestamp=datetime.now().isoformat(),
                error=error,
            )
            new_algo.append(snap)

        self.state.algorithm_snapshots.extend(new_algo)
        self._rank_snapshots()
        print(f"  ⏱ Step 2 (vision eval): {_elapsed(t2)}")

        top = self.state.ranked_snapshots()
        if top:
            best = top[0]
            extra = ""
            if os.environ.get("VTOS_SCORE_MODE") == "dual_rank_seg":
                _rs = best.scenario_metrics.get("dual_rank_sum")
                if _rs is not None:
                    extra = f"  rank_sum={_rs}"
            print(f"\n  ⭐ Best so far: {best.id}  {_fmt_metrics(best.performance)}{extra} (iter {best.iteration})")

        step = 3
        new_obs: List[Tuple[str, List[str]]] = []

        # Step 3: Run existing analyzers against top solutions
        if self.run_analyzers and self.state.analyzer_snapshots:
            t3 = time.time()
            print(f"[{step}/{self._n_steps()}] Running analyzers...")
            analyzer_obs = self._run_analyzer_step(top[:3])
            if analyzer_obs:
                new_obs.extend(analyzer_obs)
            print(f"  ⏱ Step 3 (analyzers): {_elapsed(t3)}")
            step += 1

        # Step 4: Propose new analyzers
        new_ana: List[AnalyzerSnapshot] = []
        if self.propose_analyzers:
            t4 = time.time()
            print(f"[{step}/{self._n_steps()}] Proposing {self.k_analyzer} analyzer(s)...")
            new_ana = self._propose_analyzer_step()
            print(f"  ⏱ Step 4 (LLM analyzers): {_elapsed(t4)}")
            step += 1

        # Step 5: Update thoughts
        t5 = time.time()
        print(f"[{step}/{self._n_steps()}] Updating thoughts...")
        t5_prompt = time.time()
        update_prompt = build_update_thoughts_prompt(
            self.problem_description, self.state, new_algo, new_obs,
        )
        t5_prompt_done = time.time()
        print(f"  ⏱ Step 5 prompt construction: {t5_prompt_done - t5_prompt:.2f}s | len={len(update_prompt)} chars ~{len(update_prompt)//4}tok")
        raw_thoughts = self._update_thoughts(update_prompt)  # LLM call timing printed inside vision_agent.py
        new_thoughts = parse_thoughts(raw_thoughts)
        if new_thoughts:
            self.state.thoughts = new_thoughts
            print("  Thoughts updated.")
        else:
            print("  WARNING: could not parse thoughts — keeping previous version")
        print(f"  ⏱ Step 5 total (prompt+LLM): {_elapsed(t5)}")

        total = time.time() - iter_start
        print(f"\n  ⏱ TOTAL iteration: {total:.1f}s  |  propose_prompt={len(propose_prompt)//4}tok  thoughts_prompt={len(update_prompt)//4}tok")

        return new_algo, new_ana, new_obs

    def _n_steps(self) -> int:
        return (
            3
            + int(self.run_analyzers and bool(self.state.analyzer_snapshots))
            + int(self.propose_analyzers)
        )

    def _run_analyzer_step(
        self, top_snaps: List[VisionSolutionSnapshot]
    ) -> List[Tuple[str, List[str]]]:
        collected: List[Tuple[str, List[str]]] = []
        for ana in self.state.analyzer_snapshots:
            if ana.error:
                continue
            ana_code = _read_code(self.workspace_dir, ana.code_file, max_chars=99999)
            for snap in top_snaps:
                per_image = snap.performance.get("_per_image", [])
                if not per_image:
                    continue
                try:
                    obs = self.evaluator.evaluate_analyzer(ana_code, per_image)
                    if obs:
                        # Fortified: Deep-convert to string to prevent unhashable dict errors
                        clean_obs = []
                        for o in obs:
                            if isinstance(o, dict):
                                clean_obs.append(json.dumps(o))
                            else:
                                clean_obs.append(str(o))

                        snap.analyzer_observations.extend(clean_obs)
                        ana.observations.extend(clean_obs)
                        collected.append((ana.purpose, clean_obs))
                        print(f"  {ana.id} → {len(obs)} observation(s) for {snap.id}")
                except Exception as exc:
                    print(f"  {ana.id} failed on {snap.id}: {exc}")
        return collected

    def _propose_analyzer_step(self) -> List[AnalyzerSnapshot]:
        iter_idx = self.state.iteration
        top_snaps = self.state.ranked_snapshots()
        # Use per-image results from the current best solution as the initial test bed
        top_per_image: List[Dict] = []
        if top_snaps:
            top_per_image = top_snaps[0].performance.get("_per_image", [])

        ana_prompt = build_analyzer_prompt(
            self.problem_description, self.state, self.k_analyzer,
            self.workspace_dir, self.library_doc,
            analyzer_task_spec=self.analyzer_task_spec,
        )
        raw = self._propose_solutions(ana_prompt)
        parsed = parse_analyzers(raw)
        print(f"  Parsed {len(parsed)} analyzer(s)")

        new_ana: List[AnalyzerSnapshot] = []
        for i, (code, purpose) in enumerate(parsed):
            ana_id = f"ana_{iter_idx:03d}_{i:02d}"
            rel_path = os.path.join("analyzers", f"{ana_id}.py")
            abs_path = os.path.join(self.workspace_dir, rel_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                obs = self.evaluator.evaluate_analyzer(code, top_per_image)
                # Fortified: Deep-convert
                obs = [json.dumps(o) if isinstance(o, dict) else str(o) for o in obs]
                error = None
                print(f"  {ana_id}: OK ({len(obs)} initial observation(s))")
            except Exception as exc:
                obs = []
                error = str(exc)
                print(f"  {ana_id}: FAILED — {error[:80]}")
            snap = AnalyzerSnapshot(
                id=ana_id,
                code_file=rel_path,
                purpose=purpose,
                observations=obs,
                iteration=iter_idx,
                timestamp=datetime.now().isoformat(),
                error=error,
            )
            new_ana.append(snap)

        self.state.analyzer_snapshots.extend(new_ana)
        return new_ana

    # ── Ranking ───────────────────────────────────────────────

    def _rank_snapshots(self):
        """Rank algorithm snapshots.

        Sort strategy depends on VTOS_SCORE_MODE env var:
          'point_f1' (default) : sort by (score=point_f1 desc, mae asc)
          'dual_rank'     : Borda count over (mIoU desc, MAE asc) — scale-invariant
                            rank-sum ranking. Lower rank-sum = better.
        """
        score_mode = os.environ.get("VTOS_SCORE_MODE", "point_f1")
        valid = [s for s in self.state.algorithm_snapshots if s.error is None]

        if score_mode == "dual_rank":
            # Borda count over two independent rankings (mIoU desc, MAE asc).
            # Each snap gets miou_rank + mae_rank; lower is better.
            n = len(valid)
            if n == 0:
                pass
            else:
                # Fall back to performance dict for the actual numerical values
                def _miou(s): return s.performance.get('miou',
                              s.performance.get('avg_miou',
                              s.scenario_metrics.get('miou', 0.0)))
                def _mae(s):  return s.performance.get('mae',
                              s.scenario_metrics.get('mae', float('inf')))
                # rank 1 = best (highest mIoU / lowest MAE)
                by_miou = sorted(valid, key=_miou, reverse=True)
                by_mae  = sorted(valid, key=_mae)
                miou_rank = {id(s): i + 1 for i, s in enumerate(by_miou)}
                mae_rank  = {id(s): i + 1 for i, s in enumerate(by_mae)}
                # Final rank-sum (lower = better)
                valid_with_rs = [
                    (s, miou_rank[id(s)] + mae_rank[id(s)],
                     miou_rank[id(s)], mae_rank[id(s)]) for s in valid
                ]
                valid_with_rs.sort(key=lambda x: x[1])  # ascending rank-sum
                valid = [t[0] for t in valid_with_rs]
                # Stash rank-sum into scenario_metrics so prompt can show it
                for s, rs, mir, mar in valid_with_rs:
                    s.scenario_metrics['dual_rank_sum'] = rs
                    s.scenario_metrics['miou_rank']     = mir
                    s.scenario_metrics['mae_rank']      = mar
        elif score_mode == "dual_rank_seg":
            # Segmentation dual-rank: Borda count over two independent
            # rankings — Dice^mask desc and mIoU^mask desc (BOTH higher = better,
            # unlike counting's dual_rank where MAE is lower=better). Each snap
            # gets dice_rank + miou_rank; lower rank-sum = better.
            n = len(valid)
            if n == 0:
                pass
            else:
                # Fall back to scenario_metrics for the actual numerical values
                def _dice(s): return s.performance.get('dice_mask',
                              s.scenario_metrics.get('dice_mask', 0.0))
                def _miou(s): return s.performance.get('miou_mask',
                              s.scenario_metrics.get('miou_mask', 0.0))
                # rank 1 = best (highest Dice^mask / highest mIoU^mask)
                by_dice = sorted(valid, key=_dice, reverse=True)
                by_miou = sorted(valid, key=_miou, reverse=True)
                dice_rank = {id(s): i + 1 for i, s in enumerate(by_dice)}
                miou_rank = {id(s): i + 1 for i, s in enumerate(by_miou)}
                # Final rank-sum (lower = better)
                valid_with_rs = [
                    (s, dice_rank[id(s)] + miou_rank[id(s)],
                     dice_rank[id(s)], miou_rank[id(s)]) for s in valid
                ]
                valid_with_rs.sort(key=lambda x: x[1])  # ascending rank-sum
                valid = [t[0] for t in valid_with_rs]
                # Stash rank-sum into scenario_metrics so prompt can show it
                for s, rs, dir_, mir in valid_with_rs:
                    s.scenario_metrics['dual_rank_sum'] = rs
                    s.scenario_metrics['dice_rank']     = dir_
                    s.scenario_metrics['miou_rank']     = mir
        else:
            # 'point_f1' — the value is stored in s.score
            valid = sorted(
                valid,
                key=lambda s: (s.score, -s.scenario_metrics.get('mae', float('inf'))),
                reverse=True,
            )

        for rank, snap in enumerate(valid, 1):
            snap.rank = rank
        for snap in self.state.algorithm_snapshots:
            if snap.error is not None:
                snap.rank = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> VisionSearchState:
        path = os.path.join(self.workspace_dir, "records.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"Resuming from existing state (iteration {data.get('iteration', 0)})")
            return VisionSearchState.from_dict(data)
        return VisionSearchState()

    def _save_state(self):
        path = os.path.join(self.workspace_dir, "records.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _propose_solutions(self, prompt: str) -> str:
        """Call the LLM with the given prompt; return raw response text."""

    @abstractmethod
    def _update_thoughts(self, prompt: str) -> str:
        """Call the LLM with the thoughts-update prompt; return raw response text."""
