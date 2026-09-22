#!/usr/bin/env python3
"""
eval/vtos_runner.py — CLI entry point for running VTOS on LVIS-Count.

Delegates to `vtos/runner/orchestrator.py::main()` for the search loop.

Usage:
    # Smoke test (3 iterations, val split)
    python -m eval.vtos_runner --exp_id e018 --split val --n-iter 3 \
        --analyzer-mode off --k-proposals 1

    # Full ablation run (10 iter, K=3, analyzer ON)
    python -m eval.vtos_runner --exp_id e020 --split test --n-iter 10 \
        --analyzer-mode on --k-proposals 3 --llm-model claude-sonnet-4-6

    # Continue training from a previous checkpoint
    python -m eval.vtos_runner --exp_id e021 --resume \
        --load-experts logs/.../demo_results_..._vtos_single_expert_analyzer_on
"""
from __future__ import annotations

import argparse
import os
import sys

# Project root on sys.path so vtos.* / tools.* / eval.* resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_parser() -> argparse.ArgumentParser:
    """CLI for a VTOS search run on LVIS-Count."""
    p = argparse.ArgumentParser(
        description="Run VTOS on LVIS-Count.")
    # Standard eval interface
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--exp_id", default="",
                   help="Experiment ID prefix (e.g. e018) used in output dir name")
    p.add_argument("--llm-model", default="gpt-4o-mini",
                   help="Model name (Poe: 'claude-sonnet-4-6'; OpenRouter: 'anthropic/claude-sonnet-4.6')")
    p.add_argument("--provider", default="poe",
                   choices=["poe", "openai", "openrouter"],
                   help="LLM provider (default: poe). Use 'openrouter' when Poe credits are low.")
    p.add_argument("--score-mode", default="point_f1",
                   choices=["point_f1", "dual_rank"],
                   help="VTOS search signal: point_f1 (lenient point-in-box F1, default) "
                        "or dual_rank (Borda count over mIoU↑ + MAE↓)")
    # VTOS-specific knobs
    p.add_argument("--n-iter", type=int, default=None,
                   help="Number of VTOS search iterations (default: N_ITER from constants)")
    p.add_argument("--k-proposals", type=int, default=3,
                   help="Solutions proposed per iteration (default: 3)")
    p.add_argument("--analyzer-mode", choices=["on", "off"], default="on",
                   help="Toggle the analyzer step (default: on)")
    p.add_argument("--load-experts", default=None,
                   help="Path to existing demo_results dir to load pre-trained experts")
    p.add_argument("--resume", action="store_true",
                   help="Continue training from existing workspace checkpoint")
    p.add_argument("--method-id", default=None,
                   help="Override the auto-generated method_id used in output folder")
    return p


def main():
    args = _build_parser().parse_args()

    # The orchestrator reads sys.argv via its own argparse — rebuild sys.argv.
    forwarded = [
        sys.argv[0],
        "--llm-model", args.llm_model,
        "--analyzer-mode", args.analyzer_mode,
        "--k-proposals", str(args.k_proposals),
        "--exp-id", args.exp_id,
        "--split", args.split,
        "--provider", args.provider,
        "--score-mode", args.score_mode,
    ]
    if args.n_iter is not None:
        forwarded += ["--n-iter", str(args.n_iter)]
    if args.load_experts:
        forwarded += ["--load-experts", args.load_experts]
    if args.resume:
        forwarded += ["--resume"]
    if args.method_id:
        forwarded += ["--method-id", args.method_id]
    elif args.exp_id:
        # Use exp_id as method_id prefix so output folders follow the e0NN convention.
        forwarded += ["--method-id",
                      f"{args.exp_id}_vtos_single_expert_{args.split}"]

    sys.argv = forwarded
    print(f"📦 VTOS runner — split={args.split}  exp_id={args.exp_id or '(none)'}")
    print(f"   forwarded args = {' '.join(forwarded[1:])}")

    from vtos.runner.orchestrator import main as run_orchestrator
    run_orchestrator()


if __name__ == "__main__":
    main()
