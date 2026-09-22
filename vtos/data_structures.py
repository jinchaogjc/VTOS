from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HypothesisEntry:
    hypothesis: str
    status: str = "unverified"  # "unverified" | "confirmed" | "refuted" | "partial"
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"hypothesis": self.hypothesis, "status": self.status, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HypothesisEntry:
        return cls(
            hypothesis=d.get("hypothesis", ""),
            status=d.get("status", "unverified"),
            evidence=d.get("evidence", ""),
        )


@dataclass
class VisionThoughts:
    insights: str = ""
    hypotheses: List[HypothesisEntry] = field(default_factory=list)
    analyzer_insights: str = ""
    exploration_directions: List[Dict[str, str]] = field(default_factory=list)
    key_problems: str = ""
    exploration_status: str = "initial - no solutions explored yet"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insights": self.insights,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "analyzer_insights": self.analyzer_insights,
            "exploration_directions": self.exploration_directions,
            "key_problems": self.key_problems,
            "exploration_status": self.exploration_status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VisionThoughts:
        return cls(
            insights=d.get("insights", ""),
            hypotheses=[HypothesisEntry.from_dict(h) for h in d.get("hypotheses", [])],
            analyzer_insights=d.get("analyzer_insights", ""),
            exploration_directions=d.get("exploration_directions", []),
            key_problems=d.get("key_problems", ""),
            exploration_status=d.get("exploration_status", "initial"),
        )

    def format(self, max_hypotheses: int = 6, max_directions: int = 6) -> str:
        # Prioritise confirmed/refuted/partial over unverified (then newest = highest index first)
        priority = {"confirmed": 0, "partial": 1, "refuted": 2, "unverified": 3}
        sorted_hyps = sorted(
            enumerate(self.hypotheses),
            key=lambda x: (priority.get(x[1].status, 4), -x[0])
        )
        shown_hyps = [h for _, h in sorted_hyps[:max_hypotheses]]
        hyp_lines = "\n".join(
            f"  [{i+1}] {h.hypothesis} | status: {h.status} | evidence: {h.evidence}"
            for i, h in enumerate(shown_hyps)
        ) or "  (none yet)"
        if len(self.hypotheses) > max_hypotheses:
            hyp_lines += f"\n  ... ({len(self.hypotheses) - max_hypotheses} older hypotheses pruned)"

        shown_dirs = self.exploration_directions[-max_directions:]
        dirs = "\n".join(
            f"  [{i+1}] {e.get('direction', '?')} | status: {e.get('status', '?')} | perf: {e.get('performance', 'unknown')}"
            for i, e in enumerate(shown_dirs)
        ) or "  (none yet)"
        if len(self.exploration_directions) > max_directions:
            dirs = f"  ... ({len(self.exploration_directions) - max_directions} older directions pruned)\n" + dirs

        return (
            f"### Insights (from experimental evidence)\n{self.insights or '(none yet)'}\n\n"
            f"### Hypotheses (tool/operation capabilities to verify)\n{hyp_lines}\n\n"
            f"### Insight for Building Analyzer\n{self.analyzer_insights or '(none yet)'}\n\n"
            f"### Exploration Directions\n{dirs}\n\n"
            f"### Key Problems & Bottlenecks\n{self.key_problems or '(none yet)'}\n\n"
            f"### Exploration Status\n{self.exploration_status}"
        )


@dataclass
class VisionSolutionSnapshot:
    id: str
    code_file: str
    key_differences: str
    performance: Dict[str, Any]
    scenario_metrics: Dict[str, float] = field(default_factory=dict)
    analyzer_observations: List[str] = field(default_factory=list)
    rank: Optional[int] = None
    iteration: int = 0
    timestamp: str = ""
    error: Optional[str] = None

    @property
    def score(self) -> float:
        return float(self.performance.get("score", float("-inf")))

    def to_dict(self) -> Dict[str, Any]:
        # Sanitize performance and metrics to ensure they only contain hashable values
        def _clean(d):
            if not isinstance(d, dict): return d
            return {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}

        return {
            "id": self.id,
            "code_file": self.code_file,
            "key_differences": self.key_differences,
            "performance": _clean(self.performance),
            "scenario_metrics": _clean(self.scenario_metrics),
            "analyzer_observations": self.analyzer_observations,
            "rank": self.rank,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VisionSolutionSnapshot:
        return cls(
            id=d["id"],
            code_file=d["code_file"],
            key_differences=d.get("key_differences", ""),
            performance=d.get("performance", {"score": float("-inf")}),
            scenario_metrics=d.get("scenario_metrics", {}),
            analyzer_observations=d.get("analyzer_observations", []),
            rank=d.get("rank"),
            iteration=d.get("iteration", 0),
            timestamp=d.get("timestamp", ""),
            error=d.get("error"),
        )

    def format_brief(self) -> str:
        """Compact one-line summary for LLM prompts.

        Score-mode aware:
          - 'point_f1'       → "score=point_f1 | mae | grounded"
          - 'dual_rank' → "RankSum (mIoU r/ MAE r) | mIoU | mae"

        All other scenario metrics (avg_rel_ae, total_hits, total_misses,
        avg_precision, avg_recall) are EXCLUDED from the LLM context to prevent
        metric overload that leads to over-engineering and regression.
        """
        import os as _os
        _sm = _os.environ.get("VTOS_SCORE_MODE", "point_f1")
        rank_str = f"#{self.rank}" if self.rank is not None else "N/A"
        if self.error:
            perf_str = f"ERROR: {self.error[:80]}"
        else:
            mae = self.scenario_metrics.get('mae')
            if _sm == "dual_rank":
                rs   = self.scenario_metrics.get('dual_rank_sum')
                mir  = self.scenario_metrics.get('miou_rank', '?')
                mar  = self.scenario_metrics.get('mae_rank', '?')
                miou = self.scenario_metrics.get('miou')
                rs_str = (str(int(rs)) if isinstance(rs, (int, float)) else str(rs))
                perf_str = f"RankSum={rs_str} (mIoU r={mir}, MAE r={mar})"
                if isinstance(miou, (int, float)): perf_str += f" | mIoU={miou:.3g}"
                if isinstance(mae,  (int, float)): perf_str += f" | mae={mae:.3g}"
            elif _sm == "composite":
                score_val = self.score
                miou = self.scenario_metrics.get('miou')
                score_str = f"{score_val:.4g}" if isinstance(score_val, (int, float)) else str(score_val)
                perf_str = f"Composite={score_str}"
                if isinstance(miou, (int, float)): perf_str += f" | mIoU={miou:.3g}"
                if isinstance(mae,  (int, float)): perf_str += f" | mae={mae:.3g}"
            elif _sm == "f1":
                score_val = self.score
                score_str = f"{score_val:.4g}" if isinstance(score_val, (int, float)) else str(score_val)
                perf_str = f"F1@0.5={score_str}"
                if isinstance(mae, (int, float)): perf_str += f" | mae={mae:.3g}"
            else:  # 'point_f1' (default) — historical format
                score_val = self.score
                score_str = f"{score_val:.4g}" if isinstance(score_val, (int, float)) else str(score_val)
                perf_str = f"score={score_str}"
                gs = self.scenario_metrics.get('grounded_score')
                if isinstance(mae, (int, float)): perf_str += f" | mae={mae:.3g}"
                if isinstance(gs,  (int, float)): perf_str += f" | grounded={gs:.3g}"
        return (
            f"[{self.id}] rank={rank_str} | {perf_str} | "
            f"iter={self.iteration} | {self.key_differences[:100]}"
        )


@dataclass
class AnalyzerSnapshot:
    id: str
    code_file: str
    purpose: str
    observations: List[str] = field(default_factory=list)
    iteration: int = 0
    timestamp: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code_file": self.code_file,
            "purpose": self.purpose,
            "observations": [json.dumps(o) if isinstance(o, dict) else str(o) for o in self.observations],
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnalyzerSnapshot:
        return cls(
            id=d["id"],
            code_file=d["code_file"],
            purpose=d.get("purpose", ""),
            observations=d.get("observations", []),
            iteration=d.get("iteration", 0),
            timestamp=d.get("timestamp", ""),
            error=d.get("error"),
        )


@dataclass
class IterationRecord:
    iteration: int
    thoughts_before: Dict[str, Any]
    thoughts_after: Dict[str, Any]
    new_algorithm_snapshots: List[Dict[str, Any]]
    new_analyzer_snapshots: List[Dict[str, Any]]
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thoughts_before": self.thoughts_before,
            "thoughts_after": self.thoughts_after,
            "new_algorithm_snapshots": self.new_algorithm_snapshots,
            "new_analyzer_snapshots": self.new_analyzer_snapshots,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> IterationRecord:
        return cls(
            iteration=d.get("iteration", 0),
            thoughts_before=d.get("thoughts_before", {}),
            thoughts_after=d.get("thoughts_after", {}),
            new_algorithm_snapshots=d.get("new_algorithm_snapshots", []),
            new_analyzer_snapshots=d.get("new_analyzer_snapshots", []),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class VisionSearchState:
    thoughts: VisionThoughts = field(default_factory=VisionThoughts)
    algorithm_snapshots: List[VisionSolutionSnapshot] = field(default_factory=list)
    analyzer_snapshots: List[AnalyzerSnapshot] = field(default_factory=list)
    history: List[IterationRecord] = field(default_factory=list)
    iteration: int = 0

    def ranked_snapshots(self) -> List[VisionSolutionSnapshot]:
        """Return snapshots sorted by rank.

        Baseline anchoring (counting score modes only):
        - Find the snapshot whose id contains 'baseline' and record its score.
        - Any non-baseline snapshot with score < baseline_score * 0.95 is:
            (a) still kept in algorithm_snapshots (history / debugging),
            (b) excluded from the returned list so it never enters LLM top-k context,
            (c) its key_differences field is prefixed with [REJECTED: below baseline]
                so the marker is persisted when state is saved to JSON.
        - If no baseline exists yet (iteration 0 before seeding), behaves as before.

        Segmentation ('dual_rank_seg') is EXEMPT from baseline anchoring. That
        gate compares the scalar (dice+miou)/2 `score`; on a hard task where no
        candidate beats the baseline it rejects EVERY candidate, leaving the LLM
        only the baseline as a code exemplar — the search goes memoryless and
        cannot refine its own best attempts. In that mode the full rank-ordered
        list (Borda dual-rank over Dice^mask and mIoU^mask) is returned as-is.
        """
        ranked = sorted(
            [s for s in self.algorithm_snapshots if s.rank is not None],
            key=lambda s: s.rank,
        )

        # Segmentation dual-rank: never gate on the (dice+miou)/2 scalar — the
        # search must always see its top candidates in order to refine them.
        if os.environ.get("VTOS_SCORE_MODE") == "dual_rank_seg":
            return ranked

        # ── Baseline anchoring (counting score modes) ───────────────────────
        # Locate baseline score (id assigned as 'sol_000_baseline' in search_engine.py)
        baseline_score: Optional[float] = None
        for s in self.algorithm_snapshots:
            if "baseline" in s.id:
                baseline_score = s.score
                break

        if baseline_score is not None and baseline_score > 0:
            rejection_threshold = baseline_score * 0.95
            accepted: List[VisionSolutionSnapshot] = []
            for s in ranked:
                if "baseline" in s.id or s.score >= rejection_threshold:
                    accepted.append(s)
                else:
                    # Mark rejected in key_differences (idempotent)
                    if not s.key_differences.startswith("[REJECTED:"):
                        s.key_differences = (
                            f"[REJECTED: below baseline "
                            f"(score={s.score:.4f} < threshold={rejection_threshold:.4f})] "
                            + s.key_differences
                        )
            return accepted

        return ranked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thoughts": self.thoughts.to_dict(),
            "algorithm_snapshots": [s.to_dict() for s in self.algorithm_snapshots],
            "analyzer_snapshots": [a.to_dict() for a in self.analyzer_snapshots],
            "history": [h.to_dict() for h in self.history],
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VisionSearchState:
        return cls(
            thoughts=VisionThoughts.from_dict(d.get("thoughts", {})),
            algorithm_snapshots=[
                VisionSolutionSnapshot.from_dict(s) for s in d.get("algorithm_snapshots", [])
            ],
            analyzer_snapshots=[
                AnalyzerSnapshot.from_dict(a) for a in d.get("analyzer_snapshots", [])
            ],
            history=[IterationRecord.from_dict(h) for h in d.get("history", [])],
            iteration=d.get("iteration", 0),
        )
