"""vtos/runner/seg_orchestrator.py — seg-native entry point for the full VTOS framework.

Sibling to vtos/runner/orchestrator.py (which stays counting-only and is NOT touched).
This entry point wires the SHARED core engine (vtos/search_engine.py,
vtos/vision_agent.py) to the seg-specific evaluator, prompts, and data
(PlantSeg v2). It produces the same artifact set as orchestrator.py — records.json,
solutions/, analyzers/ — and runs the full VisionThoughts/hypothesis machinery.

Usage:
    python -m vtos.runner.seg_orchestrator \\
        --exp-id psv2_009 \\
        --seed-code vtos/runner/seg_seed_grounded_sam2.py \\
        --n-iter 15 --k-proposals 3 \\
        --analyzer-mode off \\
        --llm-provider openrouter --llm-model anthropic/claude-sonnet-4.6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Shared core (untouched, task-agnostic):
# NOTE: VisionAgentLLM INHERITS from VisionSearchEngine — call agent.run() directly
# (there is no agent.engine attribute).
from vtos.vision_agent import VisionAgentLLM

# Seg-specific pieces (Tasks 1-3):
from vtos.runner.seg_evaluator import SegPersistentEvaluator, SegEvaluatorAdapter
from vtos.runner.seg_prompts import (
    SEG_CODE_GEN_TASK_SPEC, SEG_ANALYZER_TASK_SPEC, SEG_PROBLEM_DESCRIPTION,
)

# LLM client factory (shared with the rest of the project — lives in tools/, NOT vtos/).
from tools.llm_interface import get_llm_client


WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "logs/plantseg_vtos"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VTOS seg orchestrator (PlantSeg v2)")
    # Identity
    ap.add_argument("--exp-id", required=True,
                    help="Experiment ID prefix (e.g. psv2_009).")
    ap.add_argument("--description", default="",
                    help="Free-form one-line description recorded in workspace metadata.")
    # Search budget
    ap.add_argument("--n-iter", type=int, default=5,
                    help="Number of search iterations. Default 5 is a quick "
                         "iteration budget for rapid experiments; bump to 15 "
                         "(prior default) for full-budget runs intended to "
                         "match psv2_009..012b's wall-clock.")
    ap.add_argument("--k-proposals", type=int, default=3,
                    help="Candidate proposals per iteration.")
    ap.add_argument("--n-train-eval", type=int, default=60,
                    help="Train tasks used to score each candidate.")
    # Analyzer
    ap.add_argument("--analyzer-mode", choices=["on", "off"], default="off",
                    help="If on, analyzer proposes diagnostics per iter (more cost, richer thoughts).")
    # Seed
    ap.add_argument("--seed-code", default="",
                    help="Path to a .py file used as iter-0 baseline_code.")
    # LLM
    ap.add_argument("--llm-model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--llm-provider", default="openrouter")
    # Split (for the test-eval phase that runs AFTER search)
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="Split used for the FINAL test eval after search converges.")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Cap on test tasks (0 = full split).")
    # Val-selection (the DEFAULT now). After search, take top-K candidates by
    # train Borda, re-evaluate each on the held-out val split (20 tasks the
    # search loop never touches), pick the val-best by Dice^mask, then run
    # that one on test. Demonstrated to win +0.011 Dice^mask on psv2_011
    # (sol_015_02 vs train-best sol_013_01) without extra LLM cost.
    # Set --val-select-k 0 to fall back to pure train-selection (legacy).
    ap.add_argument("--val-select-k", type=int, default=5,
                    help="Top-K train-Borda candidates re-evaluated on val to "
                         "pick the test candidate. 0 disables (train-select only).")
    ap.add_argument("--resume-from", default="",
                    help="Path to an EXISTING workspace dir (e.g. "
                         "logs/plantseg_vtos/psv2_012c_seg_orch_<stamp>). The "
                         "engine loads records.json and continues from "
                         "state.iteration, running --n-iter MORE iters on top. "
                         "Use to extend a finished n_iter=5 run to 15 without "
                         "re-evaluating iters 1-5. Seed code is ignored on "
                         "resume (the loaded state already has iter-0 seed).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    t_start = time.time()

    # Workspace
    if args.resume_from:
        # Resume mode: reuse the existing workspace as-is. The engine's
        # _load_state() will pick up records.json and continue the iter
        # counter; the for-loop then runs --n-iter additional iters.
        workspace = Path(args.resume_from).expanduser().resolve()
        if not (workspace / "records.json").is_file():
            sys.stderr.write(f"ERROR: --resume-from path missing records.json: "
                             f"{workspace}\n")
            return 2
        print(f"workspace (resume): {workspace}", file=sys.stderr)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        workspace = WORKSPACE_ROOT / f"{args.exp_id}_seg_orch_{stamp}"
        workspace.mkdir(parents=True, exist_ok=True)
        print(f"workspace: {workspace}", file=sys.stderr)

    # Score-mode env flag (Borda dual-rank over Dice^mask and mIoU^mask) — used by
    # VisionSolutionSnapshot.format_brief and by VisionSearchState.ranked_snapshots
    # (dual_rank_seg exempts from baseline anchoring).
    os.environ["VTOS_SCORE_MODE"] = "dual_rank_seg"

    # Seed code (optional). Skipped on resume — the loaded state already
    # contains the iter-0 seed snapshot, and re-injecting would either
    # duplicate it or get silently ignored by the engine's "fresh state"
    # check at search_engine.py:1814. Either way, passing --seed-code on
    # resume is a no-op; warn and proceed.
    baseline_code_text = None
    if args.seed_code and args.resume_from:
        print(f"warning: --seed-code ignored on resume (state already has seed)",
              file=sys.stderr)
    elif args.seed_code:
        seed_path = Path(args.seed_code)
        if not seed_path.is_file():
            sys.stderr.write(f"ERROR: --seed-code path not found: {args.seed_code}\n")
            return 2
        baseline_code_text = seed_path.read_text(encoding="utf-8")
        print(f"seed: {args.seed_code} ({len(baseline_code_text)} chars)",
              file=sys.stderr)

    # Evaluator (seg-native)
    seg_eval = SegPersistentEvaluator()
    evaluator = SegEvaluatorAdapter(seg_eval, n_train_eval=args.n_train_eval)

    # LLM client (OpenRouter / Poe / etc.)
    llm = get_llm_client(provider=args.llm_provider, model=args.llm_model)

    # Agent (full VTOS engine via shared VisionAgentLLM)
    run_analyzers = (args.analyzer_mode == "on")
    agent = VisionAgentLLM(
        evaluator=evaluator,
        llm=llm,
        workspace_dir=str(workspace),
        k=args.k_proposals,
        n_iterations=args.n_iter,
        run_analyzers=run_analyzers,
        propose_analyzers=run_analyzers,    # Gate analyzer GENERATION; was missing
                                            # so engine default (True) created
                                            # analyzers even with --analyzer-mode off.
        problem_description=SEG_PROBLEM_DESCRIPTION,  # override counting default —
                                            # without this every seg run since
                                            # psv2_009 had the counting problem
                                            # statement at the top of every
                                            # propose prompt + the HTML report.
        task_spec=SEG_CODE_GEN_TASK_SPEC,
        analyzer_task_spec=SEG_ANALYZER_TASK_SPEC,
        baseline_code=baseline_code_text,
        task_family="seg",   # gates counting-only hint sections + filters past
                              # snapshot toolbox.X code (breaks hallucination loop)
    )

    # Run the search
    print(f"=== seg_orchestrator: n_iter={args.n_iter}, "
          f"k={args.k_proposals}, n_train_eval={args.n_train_eval}, "
          f"analyzer={args.analyzer_mode} ===",
          file=sys.stderr)
    state = agent.run()

    # Rank by train Borda — the search loop's own scoring criterion.
    ranked = state.ranked_snapshots()
    if not ranked:
        sys.stderr.write("ERROR: no candidates produced — search failed.\n")
        return 1
    train_best = ranked[0]

    # ── Val-selection (default) ────────────────────────────────────────
    # Train-Borda overfits to the n_train_eval=60 train sample. The held-out
    # val split (benchmark_val.json, 20 tasks, never seen by the search loop)
    # picks better — verified on psv2_011 where val-selected sol_015_02 won
    # test by +0.011 Dice^mask over train-selected sol_013_01.
    # `--val-select-k 0` disables this and falls back to pure train-selection.
    val_eval_results = {}
    if args.val_select_k > 0 and len(ranked) > 1:
        top_k = ranked[: args.val_select_k]
        print(f"\n=== Val-selection: re-evaluating top-{len(top_k)} on val ===",
              file=sys.stderr)
        for snap in top_k:
            code_str = (workspace / snap.code_file).read_text(encoding="utf-8")
            r = seg_eval.evaluate(code_str, split="val")
            val_eval_results[snap.id] = {
                k: v for k, v in r.items() if k != "_per_image"
            }
            print(f"  VAL {snap.id:14s} dice={r['dice_mask']:.4f}  "
                  f"miou={r['miou_mask']:.4f}  n={r['n']}", file=sys.stderr)
        val_best_id = max(val_eval_results,
                          key=lambda sid: val_eval_results[sid]["dice_mask"])
        best = next(s for s in top_k if s.id == val_best_id)
        with open(workspace / "val_eval.json", "w") as f:
            json.dump(val_eval_results, f, indent=2)
        print(f"val-best: {best.id} "
              f"(val dice={val_eval_results[best.id]['dice_mask']:.4f}, "
              f"train_borda_rank=#{top_k.index(best)+1})", file=sys.stderr)
    else:
        best = train_best
        print(f"train-best: {best.id} (train score={best.score:.4f}) "
              f"[val-selection disabled]", file=sys.stderr)

    best_code_path = workspace / "best_solution.py"
    best_code = (workspace / best.code_file).read_text(encoding="utf-8")
    best_code_path.write_text(
        f"# {best.id} — train score={best.score:.4f}\n\n{best_code}\n")

    # Test eval on the selected best
    test_perf = seg_eval.evaluate(best_code, split=args.split,
                                  n_tasks=args.max_tasks or None)
    with open(workspace / "test_eval.json", "w") as f:
        json.dump(test_perf, f, indent=2)

    # Session metadata — on resume, record total_iters_completed (from state.history
    # length) so the post-hoc reader can tell this was a continuation vs a
    # fresh n_iter=N run. n_iter still reflects the iters added THIS invocation.
    metadata = {
        "exp_id": args.exp_id,
        "description": args.description,
        "split": args.split,
        "n_iter": args.n_iter,
        "resumed_from": args.resume_from or None,
        "total_iters_completed": len(state.history),
        "k_proposals": args.k_proposals,
        "n_train_eval": args.n_train_eval,
        "score_mode": "dual_rank",
        "analyzer_mode": args.analyzer_mode,
        "llm_model": args.llm_model,
        "llm_provider": args.llm_provider,
        "seed_code_path": args.seed_code,
        "workspace": str(workspace),
        "wall_time_sec": round(time.time() - t_start, 1),
        "best_id": best.id,
        "best_train_score": float(best.score),
        "test_score": float(test_perf.get("score", 0.0)),
        # Val-selection lineage. train_best_id is what pure train-Borda would
        # have picked (== best_id when val-selection is disabled, OR when the
        # val-best happens to also be train-Borda #1).
        "val_select_k": args.val_select_k,
        "train_best_id": train_best.id,
        "selected_by": "val" if (args.val_select_k > 0 and val_eval_results
                                  and best.id != train_best.id) else "train",
        "val_dice_of_selected": (val_eval_results[best.id]["dice_mask"]
                                  if val_eval_results and best.id in val_eval_results
                                  else None),
    }
    with open(workspace / "metadata_session.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {workspace}/metadata_session.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
