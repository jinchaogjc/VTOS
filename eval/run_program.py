"""Score a fixed VTOS program on a benchmark split (no search, no LLM).

    python -m eval.run_program --task lvis_count --code-file programs/lvis_count_sol_012_00.py
    python -m eval.run_program --task plantseg_ood --code-file programs/plantseg_ood_sol_004_01.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["lvis_count", "plantseg_ood"])
    ap.add_argument("--code-file", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-images", type=int, default=0, help="0 = full split")
    args = ap.parse_args()
    code = open(args.code_file, encoding="utf-8").read()

    if args.task == "lvis_count":
        from vtos.runner import orchestrator as o
        bench = {"train": o.BENCH_TRAIN, "val": o.BENCH_VAL, "test": o.BENCH_TEST}[args.split]
        ids = [r["image_path"] for r in json.load(open(bench))]
        ids = ids[:args.max_images] if args.max_images else ids
        engine = o.PersistentEvaluator.get_instance()
        res = engine.evaluate(code, ids)
        engine.shutdown()
        out = {"n": len(ids), "MAE": round(res["mae"], 2), "mIoU%": round(100 * res["avg_miou"], 2)}
    else:
        from vtos.runner.seg_evaluator import SegPersistentEvaluator
        res = SegPersistentEvaluator().evaluate(code, split=args.split, n_tasks=args.max_images or None)
        out = {"n": res["n"], "Dice_mask%": round(100 * res["dice_mask"], 2),
               "mIoU_mask%": round(100 * res["miou_mask"], 2)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
