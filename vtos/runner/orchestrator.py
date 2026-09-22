"""
LVIS-Count search driver: single-expert search on the train split, val-based
solution selection, and test evaluation.
"""
import sys, os, json, argparse, subprocess

# ── Paths ─────────────────────────────────────────────────────────────
code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, code_root)

from vtos.constants import (
    TIER_BOUNDS, TIER_ORDER, count_to_tier,
    N_ITER, N_TRAIN, N_VAL, N_TEST, TEST_START, TEST_END,
    REJECTION_THRESHOLD, MIN_VAL_IMPROVEMENT,
)
_N_LO, _N_HI = TIER_BOUNDS["normal"]
_E_LO, _E_HI = TIER_BOUNDS["extreme"]

from vtos.vision_agent import VisionAgentLLM

# ── Single LLM entry point: Poe via centralised keystore ──────────────
from tools.llm_interface import get_llm_client

# ── Real Evaluation Harness ──────────────────────────────────────────
# LVIS-Count — three split files + shared images directory.
DATA_ROOT = os.path.join(code_root, "data/tasklets/lvis_count")
BENCH_TRAIN  = os.path.join(DATA_ROOT, "benchmark_train.json")
BENCH_VAL    = os.path.join(DATA_ROOT, "benchmark_val.json")
BENCH_TEST   = os.path.join(DATA_ROOT, "benchmark_test.json")
img_dir      = os.path.join(DATA_ROOT, "images")
# eval_worker.py needs a single JSON containing ALL tasks (train+val+test);
# we synthesise one in a tmp dir so the worker can look up GT by image_path.
def _build_merged_benchmark() -> str:
    """Concatenate all three v2 split JSONs into a single tmp file for the worker."""
    import tempfile
    merged = []
    for p in (BENCH_TRAIN, BENCH_VAL, BENCH_TEST):
        with open(p) as f:
            merged.extend(json.load(f))
    out = os.path.join(tempfile.gettempdir(), "lvis_count_merged_benchmark.json")
    with open(out, "w") as f:
        json.dump(merged, f)
    return out
json_path = _build_merged_benchmark()

def get_python_exe():
    """Resolve the Python interpreter for subprocess workers.

    Priority:
      1. $VTOS_PYTHON if set and exists
      2. sys.executable (current interpreter)
    """
    env_py = os.environ.get("VTOS_PYTHON")
    if env_py and os.path.exists(env_py):
        return env_py
    return sys.executable

# ── Persistent Worker Process (Standard Subprocess) ──────────────────
class PersistentEvaluator:
    # Singleton used for the final test evaluation.
    # Expert training uses per-expert instances via make_instance().
    _instance = None
    _proc = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._start_worker()
        return cls._instance

    @classmethod
    def make_instance(cls):
        """Create an independent worker process (for parallel expert training)."""
        inst = cls.__new__(cls)
        inst._proc = None
        inst._start_worker()
        return inst

    def _start_worker(self):
        print("🔧 [System] Starting persistent execution worker via eval_worker.py...")
        python_exe = get_python_exe()
        worker_script = os.path.join(os.path.dirname(__file__), "eval_worker.py")
        
        import subprocess
        self._proc = subprocess.Popen(
            [python_exe, worker_script, code_root, img_dir, json_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr, # Pass worker stderr to terminal for debugging
            text=True,
            bufsize=1
        )

    def evaluate(self, code, test_ids):
        task = {"code": code, "test_ids": test_ids}
        try:
            self._proc.stdin.write(json.dumps(task) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            print("  ⚠ Worker pipe broken — restarting worker and retrying...")
            self._start_worker()
            self._proc.stdin.write(json.dumps(task) + "\n")
            self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            print("  ⚠ Worker process died unexpectedly — restarting and retrying...")
            self._start_worker()
            try:
                self._proc.stdin.write(json.dumps(task) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except Exception:
                pass
        if not line:
            return {"score": 0.0, "mae": 999}
            
        res = json.loads(line)
        if "error" in res:
            print(f"  ⚠ Evaluator returned error: {res['error']}")
            return {"score": 0.0, "mae": 999}

        # ── Primary training score ──────────────────────────────────────
        # 'point_f1'       : F1 with point-in-bbox hit criterion (lenient)
        # 'dual_rank' : stores point_f1 here too; the actual ranking happens
        #               downstream in search_engine._rank_snapshots()
        point_f1_val   = res.get('point_f1', res.get('grounded_score',
                                           max(0.0, 1.0 - res['avg_rel_ae'])))
        miou_val  = res.get('avg_miou', 0.0)
        avg_gt    = res.get('avg_gt_count', 30.0)
        score = point_f1_val
        return {
            "score": score,
            "mae": res['mae'],
            "signed_error": res.get('signed_error', 0),  # +over-counting, -under-counting
            "avg_rel_ae": res['avg_rel_ae'],
            "point_f1": point_f1_val,
            "miou": miou_val,
            "avg_miou": miou_val,
            "avg_gt_count": avg_gt,
            "grounded_score": res.get('grounded_score', 0),
            "avg_precision": res.get('avg_precision', 0),
            "avg_recall": res.get('avg_recall', 0),
            "total_hits": res.get('total_hits', 0),
            "total_misses": res.get('total_misses', 0),
            "total_false_pos": res.get('total_false_pos', 0),
            "_per_image": res['results']
        }
    
    def shutdown(self):
        if self._proc:
            self._proc.stdin.write(json.dumps(None) + "\n")
            self._proc.stdin.flush()
            self._proc.terminate()

# ── Evaluator (accepts configurable test image IDs) ──────────────────
# ── Evaluator (now uses persistent worker for 10x speed) ─────────────
class RealEvaluator:
    def __init__(self, test_ids, shared=False):
        self.test_ids = test_ids
        # shared=True: reuse the singleton worker; shared=False: own worker process
        self.engine = PersistentEvaluator.get_instance() if shared else PersistentEvaluator.make_instance()

    def evaluate(self, code):
        """Pass code to the persistent worker engine."""
        return self.engine.evaluate(code, self.test_ids)

    def evaluate_analyzer(self, code, results):
        """Execute an analyzer's `analyze(per_image_results) -> list[str]`.

        Args:
            code: analyzer Python source.
            results: per_image_results dict list.
        """
        import numpy as np

        # Provide a minimal toolbox shim so analyzers can call toolbox.compute_iou
        class _SimpleToolbox:
            @staticmethod
            def compute_iou(boxA, boxB):
                xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                aA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
                aB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
                return inter / float(aA + aB - inter + 1e-6)

        local_vars = {
            "per_image_results": results,
            "toolbox": _SimpleToolbox(),
            "np": np
        }
        try:
            exec(code, local_vars, local_vars)
            if 'analyze' in local_vars:
                obs = local_vars['analyze'](results)
                return [json.dumps(o) if isinstance(o, dict) else str(o) for o in obs]
            return ["Error: 'analyze' function not found."]
        except Exception as e:
            return [f"Analyzer Error: {e}"]


# ── Expert Training Helper ────────────────────────────────────────────
# Neutral problem description — no domain hints injected.
# The LLM must discover the right strategy from MAE/Score feedback alone.
EXPERT_PROBLEM_DESCS = {
    "all": (
        "Count every visible object of the target class. The training set covers two density tiers:\n"
        f"  - Normal   ({_N_LO}–{_N_HI} objects, moderate density).\n"
        f"  - Extreme  ({_E_LO}–{_E_HI} objects, severe crowding).\n"
        "\n"
        "## SEED\n"
        "The starting baseline (sol_000_baseline) is a single `grounding_dino_detect` call at\n"
        "the validation-tuned best threshold (box_threshold≈0.10). It already over-predicts on\n"
        "extreme scenes (high recall, lower precision). Your goal is to keep that recall while\n"
        "improving precision — both fewer spurious boxes AND boxes that land more tightly on\n"
        "the actual objects. The per-iteration prompt will spell out the exact ranking criterion.\n"
        "\n"
        "## RULES\n"
        "1. A single strategy must work across BOTH tiers — do NOT hard-code per-image logic.\n"
        "2. BBox format is [x1, y1, x2, y2] normalized [0,1]. Area = (x2-x1)*(y2-y1).\n"
        f"3. Solutions scoring below {REJECTION_THRESHOLD} are REJECTED.\n"
        "\n"
        "## EXPLORATION IDEAS (try several; the search will tell you what generalizes)\n"
        "- A. Add `toolbox.nms_filter(bboxes, iou_threshold=0.40)` after the detect call to deduplicate.\n"
        "- B. Vary `iou_threshold` ∈ [0.20, 0.55] in the NMS step.\n"
        "- C. Multi-threshold ensemble around the seed: e.g. union of detect(0.08) + detect(0.12), then NMS.\n"
        "- D. Density-adaptive post-NMS: tighter NMS only on dense regions (grid-based cell counts).\n"
        "- E. (extreme tier only) `toolbox.slice_and_detect(image_path, text_query, grid=(R, C))` — but be\n"
        "     cautious: slicing on normal-density scenes inflates duplicates.\n"
        "\n"
        "## REFERENCE STARTING POINT (one possibility; explore widely)\n"
        "```python\n"
        "import numpy as np\n"
        "bboxes = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold=0.10)\n"
        "bboxes = toolbox.nms_filter(bboxes, iou_threshold=0.40)\n"
        "```"
    ),
}

def _pick_val_best(items, score_mode, baseline_id_match=lambda x: 'baseline' in str(x).lower()):
    """Score-mode-aware val selection helper.

    Args:
        items: list of dicts with keys:
            'id'          : solution id string (for log/baseline detection)
            'code'        : code text (returned with chosen)
            'snap'        : original snapshot object (returned with chosen)
            'val_perf'    : evaluator output dict (has 'point_f1','miou','mae','avg_gt_count'...)
            'train_score' : float; only for logging
        score_mode: 'point_f1' | 'dual_rank'

    Returns:
        (chosen_dict, val_metric_for_display)
        — chosen_dict is one of `items`
        — val_metric_for_display is point_f1 (point_f1 mode) or rank_sum (dual_rank
          mode; lower is better)

    Side effects: prints per-candidate val scores; applies baseline fallback in
    'point_f1' mode where MIN_VAL_IMPROVEMENT is meaningful. In
    'dual_rank' mode, Borda count itself includes the baseline as a candidate,
    so no separate margin check is applied — if the baseline has the lowest
    rank-sum, it naturally wins.
    """
    if not items:
        return None, 0.0

    if score_mode == "dual_rank":
        def _miou(it): return it['val_perf'].get('miou', it['val_perf'].get('avg_miou', 0.0))
        def _mae(it):  return it['val_perf'].get('mae', float('inf'))
        by_miou = sorted(items, key=_miou, reverse=True)
        by_mae  = sorted(items, key=_mae)
        miou_rank_map = {id(it): i + 1 for i, it in enumerate(by_miou)}
        mae_rank_map  = {id(it): i + 1 for i, it in enumerate(by_mae)}
        for it in items:
            it['_val_rank_sum'] = miou_rank_map[id(it)] + mae_rank_map[id(it)]
            it['_val_miou_rank'] = miou_rank_map[id(it)]
            it['_val_mae_rank']  = mae_rank_map[id(it)]
        scored = sorted(items, key=lambda x: x['_val_rank_sum'])
        for it in scored:
            print(f"  [val] {it['id']}: train={it['train_score']:.4f}  "
                  f"val_RankSum={it['_val_rank_sum']} "
                  f"(mIoU r={it['_val_miou_rank']}, MAE r={it['_val_mae_rank']}; "
                  f"val_mIoU={_miou(it):.4f}, val_MAE={_mae(it):.2f})")
        best = scored[0]
        print(f"  ✅ Val-best (dual_rank): [{best['id']}] val_RankSum={best['_val_rank_sum']} "
              f"(val_mIoU={_miou(best):.4f}, val_MAE={_mae(best):.2f})")
        return best, float(best['_val_rank_sum'])

    else:  # 'point_f1' (default) — historical behaviour
        for it in items:
            it['_val_point_f1'] = it['val_perf'].get('point_f1', it['val_perf'].get('score', 0.0))
            print(f"  [val] {it['id']}: train={it['train_score']:.4f}  val={it['_val_point_f1']:.4f}")
        scored = sorted(items, key=lambda x: x['_val_point_f1'], reverse=True)
        best = scored[0]
        baseline = next((it for it in items if baseline_id_match(it['id'])), None)
        if baseline is not None and not baseline_id_match(best['id']):
            margin = best['_val_point_f1'] - baseline['_val_point_f1']
            if margin < MIN_VAL_IMPROVEMENT:
                print(f"  ↩ Margin {margin:.4f} < {MIN_VAL_IMPROVEMENT}, reverting to baseline "
                      f"(search={best['_val_point_f1']:.4f} baseline={baseline['_val_point_f1']:.4f})")
                best = baseline
        print(f"  ✅ Val-best: [{best['id']}] val_point_f1={best['_val_point_f1']:.4f}")
        return best, float(best['_val_point_f1'])


def _val_select(snaps, workspace, val_ids, top_k=5):
    """Re-evaluate top-k training candidates on held-out val set.

    Score-mode-aware: respects VTOS_SCORE_MODE env var. In 'point_f1' mode keeps the
    historical behaviour (select by val point_f1, baseline-margin fallback). In
    'dual_rank' mode selects by val Borda rank-sum.

    Always includes sol_000_baseline as a candidate even if it falls outside
    top-k by training score.

    Returns: (snap_data, code, val_score_for_display)
    """
    score_mode = os.environ.get("VTOS_SCORE_MODE", "point_f1")
    sorted_snaps = sorted(snaps, key=lambda s: s.get('performance', {}).get('score', 0), reverse=True)
    baseline_snap = next((s for s in snaps if 'baseline' in s.get('id', '').lower()), None)
    candidates = sorted_snaps[:top_k]
    if baseline_snap and baseline_snap not in candidates:
        candidates = candidates + [baseline_snap]
    engine = PersistentEvaluator.make_instance()
    items = []
    for snap_data in candidates:
        code_path = os.path.join(workspace, snap_data.get('code_file', ''))
        if not os.path.exists(code_path):
            continue
        with open(code_path) as f:
            code = f.read()
        val_perf = engine.evaluate(code, val_ids)
        items.append({
            'id':          snap_data.get('id', '?'),
            'snap':        snap_data,
            'code':        code,
            'val_perf':    val_perf,
            'train_score': snap_data.get('performance', {}).get('score', 0.0),
        })
    engine.shutdown()
    if not items:
        return snaps[0], None, 0.0
    best, val_score = _pick_val_best(items, score_mode)
    return best['snap'], best['code'], val_score


def train_expert(name, train_ids, workspace, llm, n_iter=3, force_retrain=False, use_analyzers=True, val_ids=None, k=3, resume=False):
    """Train one expert agent on the given image subset.

    Args:
        val_ids: Optional held-out images for solution selection.  If provided, top-5 train
                 candidates are re-evaluated on val_ids and the best by val point_f1 is returned,
                 preventing overfitting to the training images.
        use_analyzers: If True (default), run full VTOS loop with analyzer proposal/execution.
    """
    print(f"\n{'='*60}")
    print(f"🎓 Training [{name.upper()}] Expert")
    print(f"   Images : {train_ids}")
    print(f"   Iters  : {n_iter}")
    if val_ids:
        print(f"   Val    : {val_ids}")
    print(f"{'='*60}")
    problem_desc = EXPERT_PROBLEM_DESCS.get(name, "Count objects accurately in images.")

    records_path = os.path.join(workspace, 'records.json')

    # If force_retrain, wipe workspace and start fresh
    if force_retrain and os.path.exists(workspace):
        import shutil
        shutil.rmtree(workspace)
        print(f"  🗑 Cleared old workspace for {name} expert (force retrain)")

    # If already trained, skip training but still apply val-based selection if requested
    # Skip this early-exit when resume=True — search engine will continue from saved state
    if os.path.exists(records_path) and not resume:
        with open(records_path, 'r') as f:
            saved = json.load(f)
        snaps = saved.get('algorithm_snapshots', [])
        if snaps:
            if val_ids:
                best_snap_data, best_code, val_score = _val_select(snaps, workspace, val_ids)
                score = val_score
                mae   = best_snap_data.get('performance', {}).get('mae', '?')
                _sm_label = {"point_f1": "val_point_f1",
                             "dual_rank": "val_RankSum"}.get(
                    os.environ.get("VTOS_SCORE_MODE", "point_f1"), "val_score")
                print(f"  ⏭ Already trained — best by val [{best_snap_data['id']}] "
                      f"{_sm_label}={score:.4f} | mae={mae}")
            else:
                best_snap_data = max(snaps, key=lambda s: s.get('performance', {}).get('score', 0))
                best_code_path = os.path.join(workspace, best_snap_data['code_file'])
                if os.path.exists(best_code_path):
                    with open(best_code_path, 'r') as f:
                        best_code = f.read()
                score = best_snap_data.get('performance', {}).get('score', 0)
                mae   = best_snap_data.get('performance', {}).get('mae', '?')
                print(f"  ⏭ Already trained — loading [{best_snap_data['id']}] score={score:.4f} | mae={mae}")
            class _Snap:
                pass
            snap = _Snap()
            snap.id = best_snap_data['id']
            snap.score = score
            snap.performance = best_snap_data.get('performance', {})
            return best_code, snap

    os.makedirs(workspace, exist_ok=True)
    evaluator = RealEvaluator(test_ids=train_ids)
    if not use_analyzers:
        print(f"  ⚡ Analyzer mode: OFF (speed mode, ~50s/iter)")
    else:
        print(f"  🔬 Analyzer mode: ON (full VTOS, ~2-2.5min/iter)")
    agent = VisionAgentLLM(
        evaluator=evaluator,
        problem_description=problem_desc,
        workspace_dir=workspace,
        k=k, n_iterations=n_iter,
        run_analyzers=use_analyzers,
        propose_analyzers=use_analyzers,
        llm=llm,
    )
    agent.run()

    ranked = agent.state.ranked_snapshots()
    if not ranked:
        print(f"  ⚠ No ranked solutions for {name} expert.")
        return None, None

    if val_ids:
        # Re-rank top-5 by val using current score_mode; reuse this expert's worker.
        # Always include sol_000_baseline: it may rank below top-5 by train score but often generalises best.
        score_mode = os.environ.get("VTOS_SCORE_MODE", "point_f1")
        baseline_snap = next((r for r in ranked if 'baseline' in r.id.lower()), None)
        top5 = ranked[:5]
        ranked_for_val = top5 + ([baseline_snap] if baseline_snap and baseline_snap not in top5 else [])
        items = []
        for r in ranked_for_val:
            rpath = os.path.join(workspace, r.code_file)
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                code = f.read()
            val_perf = evaluator.engine.evaluate(code, val_ids)
            items.append({
                'id':          r.id,
                'snap':        r,
                'code':        code,
                'val_perf':    val_perf,
                'train_score': r.score,
            })
        if items:
            chosen, val_metric = _pick_val_best(items, score_mode)
            best, best_code = chosen['snap'], chosen['code']
            # Stash val metric on the snap object for downstream metadata.
            # In point_f1 mode, val_metric is the score (higher better).
            # In dual_rank mode, val_metric is the rank-sum (lower better) — store both
            # the rank-sum and the val point_f1 (for human-readable reporting).
            best.performance["val_score_mode"] = score_mode
            if score_mode == "dual_rank":
                best.performance["val_rank_sum"] = val_metric
                best.performance["val_point_f1"] = chosen['val_perf'].get('point_f1', 0.0)
            else:
                best.performance["val_point_f1"] = val_metric
        else:
            best = ranked[0]
            best_code_path = os.path.join(workspace, best.code_file)
            with open(best_code_path, 'r') as f:
                best_code = f.read()
    else:
        best = ranked[0]
        best_code_path = os.path.join(workspace, best.code_file)
        with open(best_code_path, 'r') as f:
            best_code = f.read()
        print(f"\n  ✅ Best: [{best.id}] score={best.score:.4f} | mae={best.performance.get('mae','?')}")

    return best_code, best



from datetime import datetime, timezone

# ── Tee Logger: mirrors stdout to a file ─────────────────────────────
class _TeeLogger:
    """Writes all print() output to both stdout AND a log file."""
    def __init__(self, log_path: str):
        import sys
        self._log = open(log_path, 'w', buffering=1, encoding='utf-8')
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        import sys
        sys.stdout = self._stdout
        self._log.close()


_LLM_SHORTNAMES = {
    "gpt-4o-mini": "4omini",
    "gpt-4o": "4o",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "claude-opus-4-6": "opus",
    "qwen/qwen3-vl-8b-instruct": "qwen3vl8b",
    "qwen/qwen3-vl-32b-instruct": "qwen3vl32b",
}


def _build_config_tag(settings: dict) -> str:
    """Compose a compact, sortable tag from VTOS settings.

    Format: se_a{on|off}[_kN][_nN][_<llm-short>]
    Defaults (k=3, n=10) are omitted from the tag to keep it short.
    """
    parts = ["se"] if settings.get("single_expert") else ["me"]
    parts.append("aon" if settings.get("analyzer_mode") == "on" else "aoff")
    k = settings.get("k_proposals", 3)
    n = settings.get("n_iter", 10)
    if k != 3:
        parts.append(f"k{k}")
    if n != 10:
        parts.append(f"n{n}")
    llm = settings.get("llm_model", "")
    short = _LLM_SHORTNAMES.get(llm, llm.split("/")[-1].replace("-", "")[:10])
    parts.append(short)
    return "_".join(parts)


def save_standardized_demo_results(
    method_id: str,
    method_name: str,
    test_per_image: list,
    test_ids: list,
    benchmark_records: list,
    settings: dict,
    extra_meta: dict,
    run_ts: str,
    exp_id: str = "",
    split: str = "test",
):
    """Write the standardized results layout:

        logs/lvis_count/{exp_id}_{split}_vtos_{config_tag}_{ts}/

    Inside: metadata.json, session_metadata.md, results.csv, results.json,
            predictions/*.json.
    """
    timestamp = run_ts[:16].replace(":", "-")  # 2026-05-14T22-43
    # Compact "config_tag" (e.g. "se_aoff_sonnet").
    cfg_tag = _build_config_tag(settings)
    folder_prefix = f"{exp_id}_" if exp_id else ""
    folder_name   = f"{folder_prefix}{split}_vtos_{cfg_tag}_{timestamp}"
    out_dir   = os.path.join(code_root, "logs", "lvis_count", folder_name)
    pred_dir  = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Key by BARE filename (benchmark JSON convention). The per_image dicts may
    # have either bare or full-path image_path; we normalise via basename when
    # looking up (see bench lookup below).
    bench_by_image = {r["image_path"]: r for r in benchmark_records if r["image_path"] in test_ids}

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=code_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "method_id":     method_id,
        "method_name":   method_name,
        "settings":      settings,
        "git_commit":    git_commit,
        "run_timestamp": run_ts,
        "test_set":      f"test_{TEST_START:03d}–{TEST_END:03d}",
        "n_images":      len(test_per_image),
        **extra_meta,
    }
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # ── Human-readable session metadata ──
    s = settings
    test_agg = extra_meta.get("test_aggregate", {})
    expert_best = (extra_meta.get("expert") or {}).get("all", {}).get("best_id", "?")
    md_lines = [
        "# VTOS Eval Session Metadata",
        "",
        f"- **Experiment ID:** {method_id}",
        f"- **Method:** {method_name}",
        f"- **Timestamp:** {run_ts}",
        f"- **Git commit:** {git_commit}",
        "",
        "## LLM Configuration",
        f"- **Model:** {s.get('llm_model', '?')}",
        f"- **Analyzer:** {s.get('analyzer_mode', '?').upper()}",
        "",
        "## Search Configuration",
        f"- **Single Expert:** {s.get('single_expert', False)}",
        f"- **N iterations:** {s.get('n_iter', '?')}",
        f"- **K proposals per iter:** {s.get('k_proposals', '?')}",
        f"- **Resumed from checkpoint:** {s.get('resumed', False)}",
    ]
    if s.get('loaded_experts_from'):
        md_lines.append(f"- **Loaded experts from:** {s['loaded_experts_from']}")
    md_lines += [
        "",
        "## Results",
        f"- **Best solution ID:** {expert_best}",
        f"- **Test point_f1:** {test_agg.get('routing_score', '?')}",
        f"- **Test MAE:** {test_agg.get('routing_mae', '?')}",
        f"- **# test images:** {len(test_per_image)}",
        f"- **Wall time:** {extra_meta.get('wall_time_seconds', '?')}s",
    ]
    with open(os.path.join(out_dir, "session_metadata.md"), 'w') as mf:
        mf.write("\n".join(md_lines) + "\n")

    # ── Build per-task list, recomputing canonical Hungarian metrics ─────
    # This guarantees every row in results.csv / results.json uses the SAME
    # metric (metrics/grounded_metrics.compute_*) throughout.
    import csv as _csv
    sys.path.insert(0, code_root)
    from metrics.grounded_metrics import (compute_point_f1_metrics, compute_miou,
                                          compute_count_metrics)

    per_task_rows = []
    for r in test_per_image:
        # Normalise image_path to bare filename for bench lookup. Defensive: if
        # upstream code already normalised (the recent fix in main()) this is a no-op.
        raw_img  = r["image_path"]
        img_name = os.path.basename(raw_img) if raw_img else raw_img
        bench    = bench_by_image.get(img_name) or bench_by_image.get(raw_img) or {}
        gt_cnt   = r.get("gt_count", bench.get("count", 0))
        gt_boxes = r.get("gt_bboxes") or bench.get("bounding_boxes", [])   # xywh
        pred_count = r.get("pred_count", 0)
        pred_xyxy  = r.get("pred_bboxes", [])                           # xyxy
        tier     = count_to_tier(gt_cnt) if gt_cnt > 0 else "unknown"

        # Per-image JSON (legacy format, kept for backward compat).
        # img_name is the bare filename (basename-normalised above), so the
        # output predictions/<image>.json path is well-formed regardless of
        # whether eval_worker stored bare or full paths.
        payload = {
            "image_path":    img_name,
            "pred_bboxes":   pred_xyxy,
            "gt_bboxes":     gt_boxes,
            "pred_count":    pred_count,
            "gt_count":      gt_cnt,
            "target_class":  bench.get("target_class", ""),
            "density_tier":  tier,
            "inference_time": r.get("time", None),
            "settings":      settings,
            "run_timestamp": run_ts,
        }
        with open(os.path.join(pred_dir, img_name + ".json"), 'w') as f:
            json.dump(payload, f, indent=2)

        # Canonical per-task metrics row
        point_f1_d = compute_point_f1_metrics(pred_xyxy, gt_boxes)
        miou  = compute_miou(pred_xyxy, gt_boxes)
        cnt_d = compute_count_metrics(pred_count, gt_cnt)
        per_task_rows.append({
            "task_id":         img_name.replace(".jpg", ""),
            "category":        bench.get("target_class", "unknown"),
            "density_tier":    tier,
            "gt_count":        gt_cnt,
            "pred_count":      pred_count,
            "mae":             cnt_d["ae"],
            "squared_error":   cnt_d["squared_error"],
            "bias":            cnt_d["bias"],
            "miou":            miou,
            "point_f1":             point_f1_d["point_f1"],
            "precision":       point_f1_d["precision"],
            "recall":          point_f1_d["recall"],
            "predicted_boxes": pred_xyxy,
        })

    # results.csv (no list columns)
    if per_task_rows:
        csv_rows = [{k: v for k, v in r.items() if k != "predicted_boxes"}
                    for r in per_task_rows]
        with open(os.path.join(out_dir, "results.csv"), 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        # results.json (full — includes predicted_boxes for downstream cost/metric scripts)
        with open(os.path.join(out_dir, "results.json"), 'w') as f:
            json.dump(per_task_rows, f, indent=2)

    return out_dir


def print_config():
    """Print all run constants to stdout so every log is self-documenting."""
    print("─" * 60)
    print("⚙  Run configuration (from vtos/constants.py)")
    print(f"   Tier bounds : " + "  ".join(f"{t}={TIER_BOUNDS[t]}" for t in TIER_ORDER))
    print(f"   Train split : {N_TRAIN} images (benchmark_train.json)")
    print(f"   Val   split : {N_VAL} images (benchmark_val.json)")
    print(f"   Test  split : {N_TEST} images (benchmark_test.json)")
    print(f"   N_ITER      : {N_ITER}")
    print(f"   Reject < point_f1: {REJECTION_THRESHOLD}")
    print("─" * 60)


# ── Main: single-expert search + val selection + test evaluation ─────
def main():
    import time
    main_start = time.time()

    parser = argparse.ArgumentParser(description="VTOS single-expert search with optional analyzer ablation.")
    parser.add_argument("--analyzer-mode", choices=["on", "off"], default="on",
                        help="Toggle the analyzer step in the search loop (default: on)")
    parser.add_argument("--method-id", default=None,
                        help="Identifier for the standardized results dir; defaults to vtos_with/no_analyzer")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="Poe model name (e.g. gpt-4o-mini, claude-sonnet-4-6)")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Override N_ITER from constants (default: use N_ITER from vtos.constants)")
    parser.add_argument("--load-experts", default=None,
                        help="Path to an existing run dir whose expert_all/ workspace is loaded")
    parser.add_argument("--k-proposals", type=int, default=3,
                        help="Number of solution proposals per iteration (default: 3); use 1 to compare k=1 vs k=3 cost")
    parser.add_argument("--resume", action="store_true",
                        help="Continue training from existing workspace checkpoint (requires --load-experts)")
    parser.add_argument("--exp-id", default="",
                        help="Experiment ID prefix used in output dir name (e.g. e018)")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="Which split to TEST on at the end (default: test)")
    parser.add_argument("--provider", default="poe",
                        choices=["poe", "openai", "openrouter"],
                        help="LLM provider (default: poe). Use 'openrouter' when Poe credits are low.")
    parser.add_argument("--score-mode", default="point_f1",
                        choices=["point_f1", "dual_rank"],
                        help="Primary search signal:\n"
                             "  'point_f1'       = F1 with point-in-bbox hit (lenient; default)\n"
                             "  'dual_rank' = Borda count over (mIoU desc, MAE asc)")
    args = parser.parse_args()

    ENABLE_ANALYZERS = (args.analyzer_mode == "on")
    # Expose score-mode to evaluator + ranking via env var (avoids threading 5 layers)
    os.environ["VTOS_SCORE_MODE"] = args.score_mode
    print(f"  📊 Score mode: {args.score_mode}")
    n_iter_run       = args.n_iter if args.n_iter is not None else N_ITER
    k_proposals      = args.k_proposals
    _ana_tag  = "analyzer_on" if ENABLE_ANALYZERS else "no_analyzer"
    method_id   = args.method_id or f"vtos_single_expert_{_ana_tag}"
    method_name = f"VTOS Single-Expert ({'analyzer ON' if ENABLE_ANALYZERS else 'analyzer OFF'})"

    # ── Timestamped Workspace ─────────────────────────────────────────
    # Workspaces (intermediate solutions, HTML report, execution.log) live
    # outside the source tree at logs/vtos_workspace/.  This keeps vtos/ clean.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_root = os.path.join(code_root, "logs", "vtos_workspace")
    os.makedirs(workspace_root, exist_ok=True)
    base_dir = os.path.join(workspace_root, f"{method_id}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    log_path = os.path.join(base_dir, "execution.log")
    _tee = _TeeLogger(log_path)

    print_config()
    print("🚀 VTOS single-expert search + val selection + test")
    print(f"🔬 Ablation arm  : {method_id}  (analyzer={'ON' if ENABLE_ANALYZERS else 'OFF'})")
    print(f"📍 Output dir    : {base_dir}")
    print(f"📝 Execution log: {log_path}")

    llm = get_llm_client(provider=args.provider, model=args.llm_model)
    print(f"  ✅ LLM client: {type(llm).__name__} ({args.provider} / {args.llm_model})")

    # ── Dataset split (explicit train/val/test JSONs) ─────────────────
    # Tier membership is in each task's `density_tier` field (or derivable
    # from `count` via count_to_tier).
    def _load_split(path):
        with open(path) as f:
            return json.load(f)
    _train_records = _load_split(BENCH_TRAIN)
    _val_records   = _load_split(BENCH_VAL)
    _test_records  = _load_split(BENCH_TEST)

    def _tier_of(rec):
        return rec.get("density_tier") or count_to_tier(rec.get("count", 0))

    # Image-path lists (filenames only — eval_worker resolves against img_dir).
    train_image_paths = [r["image_path"] for r in _train_records]
    val_image_paths   = [r["image_path"] for r in _val_records]
    test_ids          = [r["image_path"] for r in _test_records]

    # Per-tier breakdown (for logging).
    _by_tier_train = {t: [r["image_path"] for r in _train_records if _tier_of(r) == t]
                      for t in TIER_ORDER}
    _by_tier_val   = {t: [r["image_path"] for r in _val_records   if _tier_of(r) == t]
                      for t in TIER_ORDER}

    # GT tier labels for test images (used for reporting).
    test_tier_labels = {r["image_path"]: _tier_of(r) for r in _test_records}

    # Train on benchmark_train.json (60 images), select on benchmark_val.json
    # (20 images), test on benchmark_test.json (100 images). All splits use
    # disjoint categories.
    single_train_ids = train_image_paths
    single_val_ids   = val_image_paths
    print(f"\n  📍 Single-expert train : {len(single_train_ids)} images  "
          f"(per tier: " + ", ".join(f"{t}={len(_by_tier_train[t])}" for t in TIER_ORDER) + ")")
    print(f"  📍 Single-expert val   : {len(single_val_ids)} images  "
          f"(per tier: " + ", ".join(f"{t}={len(_by_tier_val[t])}" for t in TIER_ORDER) + ")")
    print(f"  📍 Test                : {len(test_ids)} images (benchmark_test.json)")
    print(f"  🔬 Analyzer mode: {'ON' if ENABLE_ANALYZERS else 'OFF (speed mode)'}")
    print(f"  🔁 N_ITER        : {n_iter_run}")

    # ── Single-expert search ──────────────────────────────────────────
    single_workspace = os.path.join(base_dir, "expert_all")
    if args.load_experts:
        src = os.path.join(args.load_experts, "expert_all")
        if os.path.exists(src) and not os.path.exists(single_workspace):
            import shutil; shutil.copytree(src, single_workspace)
            print(f"  📂 Loaded expert_all from {src}")
    print("\n🚀 Training single expert on all density tiers...")
    single_code, single_snap = train_expert(
        "all", single_train_ids, single_workspace, llm,
        n_iter=n_iter_run,
        force_retrain=not bool(args.load_experts),
        use_analyzers=ENABLE_ANALYZERS,
        val_ids=single_val_ids,
        k=k_proposals,
        resume=args.resume,
    )
    print(f"\n{'='*60}\n📊 Single-Expert Training Summary\n{'='*60}")
    if single_snap:
        p = single_snap.performance
        point_f1_v = p.get('point_f1', p.get('grounded_score', '?'))
        print(f"  [all] best={single_snap.id} | point_f1={point_f1_v:.4f} | mae={p.get('mae','?'):.1f}")

    # Evaluate: apply single expert code to all test images directly
    engine = PersistentEvaluator.get_instance()
    t_test = time.time()
    code_to_run = single_code or "bboxes = []"
    test_perf = engine.evaluate(code_to_run, test_ids)
    print(f"  ⏱ Test evaluation: {time.time() - t_test:.1f}s")

    point_f1_score = test_perf.get('point_f1', 0.0)
    mae_score = test_perf.get('mae', float('nan'))
    miou_score = test_perf.get('avg_miou', test_perf.get('miou', float('nan')))
    f1_iou50  = test_perf.get('f1_iou50', float('nan'))
    # eval_worker.py returns per-image results under '_per_image' (not 'results').
    per_image = test_perf.get('_per_image', test_perf.get('results', []))
    # Canonical test metrics: mIoU + MAE + F1@IoU=0.5; point_f1 kept as secondary.
    _miou_pct = miou_score * 100 if isinstance(miou_score,(int,float)) and miou_score <= 1 else miou_score
    _f1_pct   = f1_iou50  * 100 if isinstance(f1_iou50,(int,float))  and f1_iou50  <= 1 else f1_iou50
    print(f"\n  🎯 TEST mIoU       : {_miou_pct:.2f}" if isinstance(_miou_pct,(int,float)) else f"\n  🎯 TEST mIoU       : {_miou_pct}")
    print(f"  📉 TEST MAE        : {mae_score:.2f}")
    print(f"  🎯 TEST F1@IoU=0.5 : {_f1_pct:.2f}" if isinstance(_f1_pct,(int,float)) else f"  🎯 TEST F1@IoU=0.5 : {_f1_pct}")
    print(f"  (legacy) TEST point_f1  : {point_f1_score:.4f}")
    print(f"\n  Per-image breakdown:")
    for r in per_image:
        tier = test_tier_labels.get(r['image_path'], '?')
        prec_s = f"{r.get('precision',0):.0%}"
        rec_s  = f"{r.get('recall',0):.0%}"
        print(f"    {r['image_path']:20s} | tier={tier:6s} | gt={r['gt_count']:3d} | pred={r['pred_count']:3d} | hits={r.get('hits','?')} | prec={prec_s} | rec={rec_s}")

    total_wall = time.time() - main_start
    print(f"\n  ⏱ Total wall time: {total_wall:.1f}s")
    print(f"📝 Log saved to: {log_path}")

    # Build per-image list in the format save_standardized_demo_results expects
    with open(json_path) as f:
        benchmark_records = json.load(f)
    test_per_image_out = []
    for r in per_image:
        # eval_worker may store image_path as a bare filename or a full path;
        # benchmark records are keyed by bare filename, so normalise via basename.
        r_img = r.get('image_path', '')
        r_basename = os.path.basename(r_img) if r_img else ''
        bm = next((b for b in benchmark_records
                   if b['image_path'] == r_basename
                   or b['image_path'] == r_img), {})
        test_per_image_out.append({
            "image_path":  r_basename or r_img,  # store bare for downstream uniformity
            "gt_count":    r.get('gt_count', 0),
            "pred_count":  r.get('pred_count', 0),
            "hits":        r.get('hits', 0),
            "misses":      r.get('misses', 0),
            "false_pos":   r.get('false_pos', 0),
            "precision":   r.get('precision', 0.0),
            "recall":      r.get('recall', 0.0),
            "point_f1":         r.get('point_f1', 0.0),
            "ae":          r.get('ae', 0),
            "signed_error": r.get('signed_error', 0),
            "pred_bboxes": r.get('pred_bboxes', []),
            # Prefer eval_worker's gt_bboxes (always correct); fall back to bench.
            "gt_bboxes":   r.get('gt_bboxes') or bm.get('bounding_boxes', []),
        })
    settings = {
        "analyzer_mode": "on" if ENABLE_ANALYZERS else "off",
        "n_iter":        n_iter_run,
        "k_proposals":   k_proposals,
        "llm_model":     args.llm_model,
        "single_expert": True,
        "resumed":       bool(args.resume),
        "loaded_experts_from": args.load_experts,
    }
    out_dir = save_standardized_demo_results(
        method_id=method_id, method_name=method_name,
        test_per_image=test_per_image_out, test_ids=test_ids,
        benchmark_records=benchmark_records, settings=settings,
        extra_meta={"expert": {"all": {"best_id": getattr(single_snap, 'id', '?')}},
                    "test_aggregate": {"routing_score": point_f1_score, "routing_mae": mae_score},
                    "wall_time_seconds": round(total_wall, 1)},
        run_ts=datetime.now(timezone.utc).isoformat(),
        exp_id=args.exp_id, split=args.split,
    )
    print(f"✅  Standardized results: {out_dir}/")
    engine.shutdown(); _tee.close()

if __name__ == "__main__":
    main()

