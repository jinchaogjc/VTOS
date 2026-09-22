from __future__ import annotations

import html
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_structures import VisionSearchState


def _esc(s: object) -> str:
    return html.escape(str(s))


def _is_seg_snapshot(perf: dict) -> bool:
    """Return True if the snapshot's performance dict has seg metric keys.
    Seg snapshots populate 'dice_mask'; counting snapshots populate 'mae'.
    Lets the renderer auto-route between counting and seg column schemes
    without an external flag — the data itself signals which it is."""
    return "dice_mask" in perf or "miou_mask" in perf


def _score_mode() -> str:
    """Read VTOS_SCORE_MODE env var; one of 'point_f1', 'composite', 'dual_rank',
    'dual_rank_seg' (the seg-variant of dual_rank, ranks Dice^mask + mIoU^mask)."""
    return os.environ.get("VTOS_SCORE_MODE", "point_f1")


def _is_dual_rank_mode(mode: str | None = None) -> bool:
    """True for any Borda dual-rank mode — counting's 'dual_rank' (mIoU+MAE)
    or seg's 'dual_rank_seg' (Dice^mask+mIoU^mask). Several rendering paths
    below special-case dual-rank (rank-sum is an int, lower-is-better); they
    must accept BOTH mode strings, otherwise the seg variant falls through
    to the point_f1 branch and the primary cell renders empty."""
    m = mode or _score_mode()
    return m in ("dual_rank", "dual_rank_seg")


def _primary_label(mode: str | None = None) -> str:
    """Human-readable column header for the primary ranking metric."""
    mode = mode or _score_mode()
    return {
        "point_f1":           "point_f1 ↑",
        "f1":            "F1@0.5 ↑",
        "composite":     "Composite ↑",
        "dual_rank":     "RankSum ↓",
        "dual_rank_seg": "Borda ↓",   # seg dual-rank over Dice^mask + mIoU^mask
    }.get(mode, "Score")


def _is_seg_state(state) -> bool:
    """True if any snapshot in the state has a seg performance dict.
    Used to pick seg-appropriate column headers in the Search Progress Summary."""
    for snap in getattr(state, "algorithm_snapshots", []):
        perf = snap.performance if hasattr(snap, "performance") else snap.get("performance", {})
        if isinstance(perf, dict) and _is_seg_snapshot(perf):
            return True
    return False


def _progress_headers_html(state) -> str:
    """Render the Search Progress Summary thead row, seg-aware.
    For seg states: MAE → Dice^mask, mIoU → mIoU^mask, Prec/Rec → Px-P/Px-R.
    Bias column is DROPPED for seg — there is no signed_count_error in seg
    snapshots (it's a counting-only metric), so the column was always empty."""
    primary = _primary_label()
    if _is_seg_state(state):
        return (
            "<th>Iteration</th><th>New Solution</th>"
            "<th title='Mean Dice over masks'>Dice^mask ↑</th>"
            "<th title='Mean IoU over masks'>mIoU^mask ↑</th>"
            f"<th title='Primary ranking metric for the active VTOS_SCORE_MODE'>{primary}</th>"
            "<th title='Pixel precision / Pixel recall on masks'>Px-P / Px-R</th>"
            "<th>Per-Image (task_id, Dice)</th><th>Cumul. Best</th><th>Key Change</th>"
        )
    return (
        "<th>Iteration</th><th>New Solution</th><th>MAE</th><th>mIoU</th>"
        "<th title='Bias: +over, -under'>Bias</th>"
        f"<th title='Primary ranking metric for the active VTOS_SCORE_MODE'>{primary}</th>"
        "<th>Prec / Rec</th>"
        "<th>Per-Image (gt→pred, metric)</th><th>Cumul. Best</th><th>Key Change</th>"
    )


def _primary_value(snap, mode: str | None = None):
    """Return the value displayed in the primary-score column for a snapshot.

    - 'point_f1'       → snap.score (which holds the point_f1 value)
    - 'composite' → snap.score (which holds the composite value)
    - 'dual_rank' → scenario_metrics['dual_rank_sum'] (lower is better)
    """
    mode = mode or _score_mode()
    if snap.error:
        return None
    if _is_dual_rank_mode(mode):
        return snap.scenario_metrics.get('dual_rank_sum', None)
    return snap.score


def _fmt_primary(v, mode: str | None = None) -> str:
    """Format a primary value: ints for dual_rank (rank-sum), 4dp for floats."""
    mode = mode or _score_mode()
    if v is None or v == "":
        return "—"
    if _is_dual_rank_mode(mode):
        return str(int(v)) if isinstance(v, (int, float)) else str(v)
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def _score_badge(snap) -> str:
    """Score-mode aware badge: shows the primary ranking metric.

    'point_f1'             → "point_f1 0.5278"
    'composite'       → "Composite 0.4492"
    'dual_rank'       → "RankSum 9 (MAE r=2, mIoU r=7)"           ← counting
    'dual_rank_seg'   → "RankSum 5 (Dice r=4, mIoU r=1)"          ← seg
    """
    if snap.error:
        return '<span class="badge badge-error">ERROR</span>'
    mode = _score_mode()
    primary = _primary_value(snap, mode)

    # Color coding
    if _is_dual_rank_mode(mode):
        # Lower rank-sum is better; baseline ~ N+1 each = ~2*(n_snaps+1)/2 = mid.
        # Treat ≤ 6 (top quartile of typical 15-30 candidates) as good.
        good = isinstance(primary, (int, float)) and primary <= 6
    else:
        good = isinstance(primary, (int, float)) and primary >= 0.4
    color = "badge-good" if good else "badge-neutral"

    if _is_dual_rank_mode(mode):
        sm = snap.scenario_metrics
        mir = sm.get('miou_rank', '?')
        # Seg's Borda is over (Dice^mask, mIoU^mask) — sm has 'dice_rank'.
        # Counting's Borda is over (MAE, mIoU) — sm has 'mae_rank'.
        # Detect by which field is populated and label accordingly. Note:
        # `mae_rank` does NOT exist in seg snapshots, so the previous
        # unconditional "MAE r=?" rendered as a stale "?" for every seg row.
        if 'dice_rank' in sm:
            partner_label, partner_val = 'Dice r', sm['dice_rank']
        else:
            partner_label, partner_val = 'MAE r', sm.get('mae_rank', '?')
        rs_str = _fmt_primary(primary, mode)
        return (f'<span class="badge {color}">RankSum {rs_str} '
                f'({partner_label}={partner_val}, mIoU r={mir})</span>')
    label = {"point_f1": "point_f1", "f1": "F1@0.5", "composite": "Composite"}.get(mode, "Score")
    return f'<span class="badge {color}">{label} {_fmt_primary(primary, mode)}</span>'


def _hyp_badge(status: str) -> str:
    cls = {
        "confirmed": "badge-good",
        "refuted": "badge-error",
        "partial": "badge-warn",
        "unverified": "badge-muted",
    }.get(status, "badge-muted")
    return f'<span class="badge {cls}">{_esc(status)}</span>'


def _dir_badge(status: str) -> str:
    cls = {
        "tried": "badge-neutral",
        "planned": "badge-planned",
        "abandoned": "badge-error",
    }.get(status, "badge-neutral")
    return f'<span class="badge {cls}">{_esc(status)}</span>'


def _progress_row(rec, state: "VisionSearchState") -> str:
    """Generate one row for the Search Progress Summary table.

    Shows per-iteration new solution MAE, mIoU, primary-mode metric, and
    per-image breakdown. The primary-metric column auto-switches between
    point_f1 / Composite / RankSum based on VTOS_SCORE_MODE.
    """
    mode = _score_mode()
    if not rec.new_algorithm_snapshots:
        return f"<tr><td>{rec.iteration}</td><td colspan='8'>—</td></tr>"

    # Show the BEST proposal of this iteration.
    # In point_f1/composite mode: highest snap.score wins.
    # In dual_rank mode: lowest dual_rank_sum wins (but it's set AFTER ranking,
    # so during the early-iteration display it may be missing — fall back to
    # score-then-mae).
    def _snap_rankkey(s):
        sm = s.get('scenario_metrics', {})
        if _is_dual_rank_mode(mode) and 'dual_rank_sum' in sm:
            return (sm['dual_rank_sum'], -s.get('performance', {}).get('miou', 0.0))
        # higher score = better; flip sign for sort-ascending
        sc = s.get('performance', {}).get('score', float('-inf'))
        return (-sc if isinstance(sc, (int, float)) else float('inf'),
                s.get('performance', {}).get('mae', float('inf')))
    new_snap = min(rec.new_algorithm_snapshots, key=_snap_rankkey)
    perf = new_snap.get('performance', {})
    sm   = new_snap.get('scenario_metrics', {})
    # Counting populates {mae, miou, signed_error, avg_precision, avg_recall};
    # seg populates {dice_mask, miou_mask, pixel_precision, pixel_recall}.
    # Auto-detect and map seg keys onto the same column slots so the existing
    # table layout shows real numbers in both modes. The "MAE" column shows
    # Dice^mask for seg; "Bias" is left as '—' for seg (no signed_count error
    # in this evaluator).
    is_seg = _is_seg_snapshot(perf)
    if is_seg:
        new_mae   = perf.get('dice_mask', '—')
        new_miou  = perf.get('miou_mask', '—')
        new_bias  = perf.get('signed_count_error', '—')
        new_prec  = perf.get('pixel_precision', '—')
        new_rec   = perf.get('pixel_recall', '—')
    else:
        new_mae   = perf.get('mae', '—')
        new_miou  = perf.get('miou', perf.get('avg_miou', '—'))
        new_bias  = perf.get('signed_error', '—')
        new_prec  = perf.get('avg_precision', '—')
        new_rec   = perf.get('avg_recall', '—')
    new_score = perf.get('score', '—')
    new_point_f1   = perf.get('point_f1', '—')
    new_id    = new_snap.get('id', '—')
    new_rank_sum = sm.get('dual_rank_sum', '—')
    key_diff  = new_snap.get('key_differences', '—')[:100]

    # Per-image breakdown — schema differs by task family.
    # Seg per_image dicts contain: task_id, dice_mask, miou_mask, plant, tier, ...
    # Counting per_image dicts contain: image_path, gt_count, pred_count, ae, point_f1/miou.
    # Render the seg view as compact "#NNN: 0.XXXX (Δ±0.NNN)" sorted worst-first
    # (lowest Dice^mask first) so the most diagnostic tasks lead the cell.
    # The delta is current_iter_dice - prev_iter_dice for the SAME task_id, so
    # the reader can spot which tasks regressed or improved vs the last iter.
    # Seed iter (iter 0) and tasks not present in the previous iter render
    # without a delta. Counting view stays unchanged.
    per_image = perf.get('_per_image', [])
    img_parts = []
    if is_seg:
        # Build {task_id -> dice_mask} for the previous iter's BEST snapshot.
        # Pull from state.algorithm_snapshots (which holds ALL snaps from ALL
        # iters, including the iter-0 seed `sol_000_baseline`) keyed by id
        # prefix. state.history starts at iter 1 — it does NOT contain the
        # seed — so a previous lookup that only checked history left iter 1
        # without any deltas. Using algorithm_snapshots fixes that.
        prev_iter = rec.iteration - 1
        prev_per_task_dice = {}
        if prev_iter >= 0:
            prefix = f'sol_{prev_iter:03d}_'
            # algorithm_snapshots are VisionSolutionSnapshot dataclasses; their
            # .performance + .scenario_metrics fields are dicts, so we can hand
            # a {dict-shaped} view to _snap_rankkey (which only does .get()).
            prev_candidates = [
                {'id': s.id, 'performance': s.performance,
                 'scenario_metrics': s.scenario_metrics}
                for s in state.algorithm_snapshots
                if s.id.startswith(prefix)
            ]
            if prev_candidates:
                prev_best = min(prev_candidates, key=_snap_rankkey)
                prev_per = prev_best['performance'].get('_per_image', [])
                prev_per_task_dice = {
                    im.get('task_id'): im.get('dice_mask')
                    for im in prev_per
                    if isinstance(im.get('dice_mask'), (int, float))
                       and im.get('task_id') is not None
                }

        def _seg_key(im):
            d = im.get('dice_mask', None)
            return d if isinstance(d, (int, float)) else float('inf')
        for img in sorted(per_image, key=_seg_key):
            tid = img.get('task_id', '?')
            # Compact task tag: "v_seg_plantseg_train_001" → "#001"
            short = tid.rsplit('_', 1)[-1] if isinstance(tid, str) else str(tid)
            dice  = img.get('dice_mask', '?')
            dice_str = f"{dice:.4f}" if isinstance(dice, (int, float)) else str(dice)
            # Signed delta vs the same task_id in the previous iter's best.
            # 3-decimal Δ keeps cells compact (60 entries per cell already).
            delta_str = ""
            if isinstance(dice, (int, float)):
                prev_d = prev_per_task_dice.get(tid)
                if isinstance(prev_d, (int, float)):
                    delta_str = f" (Δ{(dice - prev_d):+.3f})"
            img_parts.append(f"#{short}: {dice_str}{delta_str}")
    else:
        for img in per_image:
            gt   = img.get('gt_count', '?')
            pred = img.get('pred_count', '?')
            ae   = img.get('ae', '?')
            img_name = img.get('image_path', '?').split('/')[-1]
            if _is_dual_rank_mode(mode):
                miou_i = img.get('miou', '?')
                metric_str = f"mIoU={miou_i:.2f}" if isinstance(miou_i, (int, float)) else f"mIoU={miou_i}"
            else:
                point_f1_i = img.get('point_f1', img.get('grounded_score', '?'))
                point_f1_str = f"{point_f1_i:.2f}" if isinstance(point_f1_i, (int, float)) else str(point_f1_i)
                metric_str = f"point_f1={point_f1_str}"
            img_parts.append(f"{img_name}: {gt}→{pred}(Δ{ae} {metric_str})")
    per_img_str = " | ".join(img_parts) if img_parts else "—"

    # Cumulative best at this iteration — show the appropriate metric.
    # For seg, the "MAE" column actually holds Dice^mask, so read dice_mask
    # from the cum-best snapshot too (otherwise the comparison color is broken).
    ranked = state.ranked_snapshots()
    cum_best_mae = "—"
    cum_best_id = "—"
    cum_best_primary = "—"  # mode-aware primary metric of cumulative best
    if ranked:
        best_snap = ranked[0]
        perf_best = best_snap.performance
        sm_best   = best_snap.scenario_metrics
        if isinstance(perf_best, dict):
            if is_seg:
                cum_best_mae = perf_best.get('dice_mask', '—')
            else:
                cum_best_mae = perf_best.get('mae', '—')
        cum_best_id = best_snap.id
        if _is_dual_rank_mode(mode):
            cum_best_primary = sm_best.get('dual_rank_sum', '—')
        else:
            cum_best_primary = perf_best.get('score', '—')

    def _fmt_score(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    def _fmt_mae(v):   return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    def _fmt_dice(v):  return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    def _fmt_miou(v):  return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    # For seg, the "MAE" column is really Dice^mask — render with 4 decimals
    # (matching the headline mIoU^mask column) instead of the 1-decimal MAE form.
    new_mae_str  = _fmt_dice(new_mae) if is_seg else _fmt_mae(new_mae)
    new_miou_str = _fmt_miou(new_miou)
    cum_mae_str  = _fmt_dice(cum_best_mae) if is_seg else _fmt_mae(cum_best_mae)
    # primary cell: rank-sum (int) in dual_rank, otherwise score (float)
    if _is_dual_rank_mode(mode):
        primary_cell = (str(int(new_rank_sum)) if isinstance(new_rank_sum, (int, float))
                        else str(new_rank_sum))
        cum_primary_str = (str(int(cum_best_primary)) if isinstance(cum_best_primary, (int, float))
                           else str(cum_best_primary))
    else:
        primary_cell = _fmt_score(new_score if mode == 'composite' else new_point_f1)
        cum_primary_str = _fmt_score(cum_best_primary)

    bias_str = (f"+{new_bias:.1f}" if isinstance(new_bias, (int, float)) and new_bias > 0
                else f"{new_bias:.1f}" if isinstance(new_bias, (int, float)) else str(new_bias))
    prec_str = f"{new_prec:.0%}" if isinstance(new_prec, (int, float)) else str(new_prec)
    rec_str  = f"{new_rec:.0%}"  if isinstance(new_rec,  (int, float)) else str(new_rec)

    # MAE column color: counting wants lower (MAE) is better; seg wants higher
    # (Dice^mask) is better. Flip the comparison for seg or the colors invert.
    mae_color = ""
    if isinstance(new_mae, (int, float)) and isinstance(cum_best_mae, (int, float)):
        better = (new_mae >= cum_best_mae) if is_seg else (new_mae <= cum_best_mae)
        mae_color = "style='color:var(--good)'" if better else "style='color:var(--bad)'"

    # Bias column is counting-only (signed_count_error). For seg, drop the
    # cell entirely so the row aligns with the seg header that omits Bias.
    bias_color = ""
    if isinstance(new_bias, (int, float)):
        bias_color = "style='color:var(--bad)'" if abs(new_bias) > 5 else "style='color:var(--good)'"
    bias_cell = ("" if is_seg else
                 f"<td {bias_color} title='Bias: +over-counting, -under-counting'>{bias_str}</td>")

    # primary-cell color: green if better-or-equal to cum-best, red if worse.
    # Dual-rank (counting + seg) uses rank-sum where LOWER is better; other
    # modes use score/point_f1 where HIGHER is better.
    is_dr = _is_dual_rank_mode(mode)
    primary_color = ""
    if isinstance(new_rank_sum if is_dr else (new_score if mode == "composite" else new_point_f1), (int, float)) \
            and isinstance(cum_best_primary, (int, float)):
        new_p = new_rank_sum if is_dr else (new_score if mode == "composite" else new_point_f1)
        better = (new_p <= cum_best_primary) if is_dr else (new_p >= cum_best_primary)
        primary_color = "style='color:var(--good)'" if better else "style='color:var(--bad)'"

    return f"""<tr>
      <td>{rec.iteration}</td>
      <td class='mono'>{_esc(new_id)}</td>
      <td class='rank' {mae_color}>{new_mae_str}</td>
      <td>{new_miou_str}</td>
      {bias_cell}
      <td {primary_color}><b>{primary_cell}</b></td>
      <td class='small'>{prec_str} / {rec_str}</td>
      <td class='small mono'>{_esc(per_img_str)}</td>
      <td class='small'>#{cum_best_id} ({'dice' if is_seg else 'mae'}={cum_mae_str}, {_primary_label(mode).split()[0]}={cum_primary_str})</td>
      <td class='small'>{_esc(key_diff)}</td>
    </tr>"""


def generate_html(state: "VisionSearchState", problem_description: str) -> str:
    ranked = state.ranked_snapshots()
    failed = [s for s in state.algorithm_snapshots if s.rank is None]

    # Collect all scenario metric keys in ranked order of first appearance
    scenario_keys: list = []
    for snap in state.algorithm_snapshots:
        for k in snap.scenario_metrics:
            if k not in scenario_keys:
                scenario_keys.append(k)

    # ── Solution rankings table ───────────────────────────────────────────────
    header_extras = "".join(f"<th>{_esc(k)}</th>" for k in scenario_keys)
    rows = []
    for snap in ranked:
        scenario_cells = ""
        for k in scenario_keys:
            val = snap.scenario_metrics.get(k, "—")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) and abs(val) < 100 else (f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
            scenario_cells += f'<td class="small">{val_str}</td>'
        first_obs = _esc(snap.analyzer_observations[0][:100]) if snap.analyzer_observations else "—"
        # Per-snap headline metrics: lets the reader verify rank ordering
        # without inspecting records.json. For seg: Dice^mask + mIoU^mask
        # (higher=better, 4 decimals). For counting: MAE + mIoU (lower MAE
        # is better — keep the existing convention).
        perf = snap.performance or {}
        if _is_seg_snapshot(perf):
            _v1 = perf.get('dice_mask', '—')
            _v2 = perf.get('miou_mask', '—')
            _v1_fmt = f"{_v1:.4f}" if isinstance(_v1, (int, float)) else str(_v1)
            _v2_fmt = f"{_v2:.4f}" if isinstance(_v2, (int, float)) else str(_v2)
        else:
            _v1 = perf.get('mae', '—')
            _v2 = perf.get('miou', perf.get('avg_miou', '—'))
            _v1_fmt = f"{_v1:.4f}" if isinstance(_v1, (int, float)) else str(_v1)
            _v2_fmt = f"{_v2:.4f}" if isinstance(_v2, (int, float)) else str(_v2)
        metric_cells = (f'<td class="small mono">{_v1_fmt}</td>'
                        f'<td class="small mono">{_v2_fmt}</td>')
        rows.append(f"""
        <tr>
          <td class="rank">#{snap.rank}</td>
          <td class="mono">{_esc(snap.id)}</td>
          <td>{_score_badge(snap)}</td>
          {metric_cells}
          <td class="mono small">{_esc(snap.code_file)}</td>
          <td class="small">{_esc(snap.key_differences[:120])}</td>
          {scenario_cells}
          <td class="small obs-cell">{first_obs}</td>
          <td class="small">iter {snap.iteration}</td>
        </tr>""")
    for snap in failed:
        scenario_cells = "".join(f'<td class="small">—</td>' for _ in scenario_keys)
        # Failed snaps don't have valid metrics — render dash placeholders so
        # column alignment matches the success-row schema (rank, id, badge,
        # 2 metric cells, file, key_diff, scenario_cells, obs, iter).
        rows.append(f"""
        <tr class="row-error">
          <td>—</td>
          <td class="mono">{_esc(snap.id)}</td>
          <td><span class="badge badge-error">ERROR</span></td>
          <td class="small mono">—</td><td class="small mono">—</td>
          <td class="mono small">{_esc(snap.code_file)}</td>
          <td class="small">{_esc((snap.error or '')[:120])}</td>
          {scenario_cells}
          <td class="small">—</td>
          <td class="small">iter {snap.iteration}</td>
        </tr>""")
    table_rows = "".join(rows) if rows else "<tr><td colspan='99' class='empty'>No solutions yet</td></tr>"

    # ── Hypotheses panel ──────────────────────────────────────────────────────
    hyp_html = ""
    for h in state.thoughts.hypotheses:
        hyp_html += f"""
        <div class="hyp-item">
          {_hyp_badge(h.status)}
          <span class="hyp-text">{_esc(h.hypothesis)}</span>
          <div class="small hyp-evidence">{_esc(h.evidence) if h.evidence else ''}</div>
        </div>"""
    hyp_html = hyp_html or "<p class='empty'>No hypotheses yet</p>"

    # ── Exploration directions panel ──────────────────────────────────────────
    dirs_html = ""
    for d in state.thoughts.exploration_directions:
        status = d.get("status", "unknown")
        dirs_html += f"""
        <div class="direction-item">
          {_dir_badge(status)}
          <span class="dir-text">{_esc(d.get('direction', '?'))}</span>
          <span class="dir-perf">{_esc(d.get('performance', 'unknown'))}</span>
        </div>"""
    dirs_html = dirs_html or "<p class='empty'>None explored yet</p>"

    # ── Analyzer inventory ────────────────────────────────────────────────────
    ana_rows = ""
    for ana in state.analyzer_snapshots:
        status_badge = (
            '<span class="badge badge-error">ERROR</span>'
            if ana.error
            else '<span class="badge badge-good">OK</span>'
        )
        obs_preview = "<br>".join(
            _esc(o[:120]) for o in ana.observations[:3]
        ) or "<span class='empty'>none</span>"
        ana_rows += f"""
        <tr>
          <td class="mono small">{_esc(ana.id)}</td>
          <td>{status_badge}</td>
          <td class="small">{_esc(ana.purpose[:100])}</td>
          <td class="small obs-cell">{obs_preview}</td>
          <td class="small">iter {ana.iteration}</td>
        </tr>"""
    ana_table = (
        f"""<table>
          <thead><tr><th>ID</th><th>Status</th><th>Purpose</th><th>Observations</th><th>Iter</th></tr></thead>
          <tbody>{ana_rows}</tbody>
        </table>"""
        if state.analyzer_snapshots
        else "<p class='empty'>No analyzers yet</p>"
    )

    # ── Iteration history ─────────────────────────────────────────────────────
    history_html = ""
    mode_now = _score_mode()
    primary_short = {"point_f1": "point_f1", "f1": "F1@0.5", "composite": "Composite", "dual_rank": "RankSum"}.get(mode_now, "Score")
    for rec in reversed(state.history):
        algo_html = ""
        for snap_d in rec.new_algorithm_snapshots:
            err = snap_d.get("error")
            perf_d = snap_d.get("performance", {})
            sm_d   = snap_d.get("scenario_metrics", {})
            # Pick the right primary metric per mode
            if _is_dual_rank_mode(mode_now):
                primary_v = sm_d.get('dual_rank_sum')
                if isinstance(primary_v, (int, float)):
                    # Seg: (Dice r, mIoU r); counting: (MAE r, mIoU r).
                    if 'dice_rank' in sm_d:
                        extra = (f" (Dice r={sm_d.get('dice_rank','?')}, "
                                 f"mIoU r={sm_d.get('miou_rank','?')})")
                    else:
                        extra = (f" (MAE r={sm_d.get('mae_rank','?')}, "
                                 f"mIoU r={sm_d.get('miou_rank','?')})")
                    score_str = f"{int(primary_v)}{extra}"
                else:
                    score_str = "—"
            else:
                primary_v = perf_d.get("score", "?")
                score_str = f"{primary_v:.4f}" if isinstance(primary_v, (int, float)) else str(primary_v)
            badge = (
                '<span class="badge badge-error">ERROR</span>'
                if err
                else f'<span class="badge badge-neutral">{primary_short} {score_str}</span>'
            )
            algo_html += f"""
            <div class="hist-sol">
              <span class="mono">{_esc(snap_d.get('id', ''))}</span> {badge}
              <div class="small hist-diff">{_esc(snap_d.get('key_differences', '')[:200])}</div>
            </div>"""
        ana_hist_html = ""
        for snap_d in rec.new_analyzer_snapshots:
            err = snap_d.get("error")
            badge = (
                '<span class="badge badge-error">ERROR</span>'
                if err
                else '<span class="badge badge-good">OK</span>'
            )
            ana_hist_html += f"""
            <div class="hist-sol">
              <span class="mono">{_esc(snap_d.get('id', ''))}</span> {badge}
              <div class="small hist-diff">{_esc(snap_d.get('purpose', '')[:200])}</div>
            </div>"""
        after = rec.thoughts_after
        history_html += f"""
        <details class="iter-record">
          <summary>Iteration {rec.iteration} &nbsp;<span class="small ts">{_esc(rec.timestamp[:19])}</span></summary>
          <div class="iter-body">
            <div class="iter-section">
              <h4>New Solutions ({len(rec.new_algorithm_snapshots)})</h4>
              {algo_html or "<p class='empty'>none</p>"}
            </div>
            <div class="iter-section">
              <h4>New Analyzers ({len(rec.new_analyzer_snapshots)})</h4>
              {ana_hist_html or "<p class='empty'>none</p>"}
            </div>
            <div class="iter-section">
              <h4>Updated Insights</h4>
              <div class="thought-block">
                <pre>{_esc(after.get('insights', '(none)'))}</pre>
              </div>
            </div>
            <div class="iter-section">
              <h4>Exploration Status</h4>
              <p class="thought-block">{_esc(after.get('exploration_status', ''))}</p>
            </div>
          </div>
        </details>"""
    history_html = history_html or "<p class='empty'>No iterations completed yet</p>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_valid = len(ranked)
    n_total = len(state.algorithm_snapshots)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VTOS Explorer</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --surface2:#252836;
    --border:#2e3347; --text:#e2e8f0; --muted:#8892a4;
    --accent:#6c8ef5; --good:#4ade80; --bad:#f87171; --warn:#fbbf24;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;font-size:14px;line-height:1.6}}
  h1{{font-size:1.4rem;color:var(--accent)}}
  h2{{font-size:1.1rem;color:var(--text);margin-bottom:.5rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}}
  h3{{font-size:.95rem;color:var(--muted);margin-bottom:.3rem}}
  h4{{font-size:.85rem;color:var(--accent);margin-bottom:.3rem}}
  pre{{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:.5rem;overflow-x:auto;font-size:.8rem;white-space:pre-wrap;word-break:break-word}}
  .layout{{display:grid;grid-template-columns:320px 1fr;gap:1rem;padding:1rem;max-width:1800px;margin:0 auto}}
  .sidebar{{display:flex;flex-direction:column;gap:.8rem}}
  .main{{display:flex;flex-direction:column;gap:.8rem}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem}}
  .header{{background:var(--surface);border-bottom:1px solid var(--border);padding:.8rem 1rem;display:flex;align-items:center;justify-content:space-between}}
  .ts{{color:var(--muted);font-size:.75rem}}
  .mono{{font-family:'Menlo','Courier New',monospace;font-size:.8rem}}
  .small{{font-size:.78rem;color:var(--muted)}}
  .empty{{color:var(--muted);font-style:italic;font-size:.85rem}}
  .badge{{display:inline-block;padding:.1rem .45rem;border-radius:3px;font-size:.72rem;font-weight:600}}
  .badge-good{{background:#14532d;color:var(--good)}}
  .badge-neutral{{background:#1e3a5f;color:#93c5fd}}
  .badge-error{{background:#450a0a;color:var(--bad)}}
  .badge-warn{{background:#451a03;color:var(--warn)}}
  .badge-planned{{background:#451a03;color:var(--warn)}}
  .badge-muted{{background:#1f2937;color:var(--muted)}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;padding:.4rem .6rem;font-size:.78rem;color:var(--muted);border-bottom:1px solid var(--border)}}
  td{{padding:.4rem .6rem;border-bottom:1px solid var(--border);vertical-align:top}}
  tr:hover td{{background:var(--surface2)}}
  .row-error td{{opacity:.6}}
  .rank{{font-weight:700;color:var(--accent);width:3rem}}
  .obs-cell{{max-width:200px;overflow:hidden;text-overflow:ellipsis}}
  .thought-block{{font-size:.85rem;margin-top:.3rem}}
  .thought-block pre{{margin-top:.2rem;margin-bottom:.6rem}}
  .hyp-item{{padding:.4rem 0;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:.2rem}}
  .hyp-text{{font-size:.85rem}}
  .hyp-evidence{{color:var(--muted);font-size:.75rem;font-style:italic}}
  .direction-item{{display:flex;align-items:baseline;gap:.5rem;padding:.3rem 0;border-bottom:1px solid var(--border)}}
  .dir-text{{flex:1;font-size:.85rem}}
  .dir-perf{{color:var(--muted);font-size:.78rem;max-width:160px}}
  details.iter-record{{background:var(--surface);border:1px solid var(--border);border-radius:6px;margin-bottom:.4rem}}
  details.iter-record summary{{padding:.5rem .8rem;cursor:pointer;user-select:none;list-style:none;display:flex;align-items:center;gap:.5rem}}
  details.iter-record summary::-webkit-details-marker{{display:none}}
  details.iter-record summary::before{{content:'▶';font-size:.7rem;color:var(--accent)}}
  details[open].iter-record summary::before{{content:'▼'}}
  .iter-body{{padding:.6rem .8rem;border-top:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr;gap:.8rem}}
  .iter-section{{font-size:.83rem}}
  .hist-sol{{background:var(--surface2);border-radius:4px;padding:.4rem .6rem;margin-bottom:.3rem}}
  .hist-diff{{color:var(--muted);margin-top:.15rem}}
  .problem-text{{font-size:.88rem;color:var(--muted);line-height:1.7;max-height:120px;overflow-y:auto}}
  @media(max-width:900px){{
    .layout{{grid-template-columns:1fr}}
    .iter-body{{grid-template-columns:1fr}}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>VTOS Explorer</h1>
  <span class="ts">Last updated: {_esc(now)} &nbsp;|&nbsp; Iteration: {state.iteration}</span>
</div>
<div class="layout">
  <div class="sidebar">
    <div class="card">
      <h2>Problem</h2>
      <p class="problem-text">{_esc(problem_description)}</p>
    </div>
    <div class="card">
      <h2>Insights</h2>
      <div class="thought-block">
        <pre>{_esc(state.thoughts.insights or '(none yet)')}</pre>
      </div>
    </div>
    <div class="card">
      <h2>Hypotheses</h2>
      {hyp_html}
    </div>
    <div class="card">
      <h2>Insight for Building Analyzer</h2>
      <div class="thought-block">
        <pre>{_esc(state.thoughts.analyzer_insights or '(none yet)')}</pre>
      </div>
    </div>
    <div class="card">
      <h2>Key Problems &amp; Bottlenecks</h2>
      <div class="thought-block">
        <pre>{_esc(state.thoughts.key_problems or '(none yet)')}</pre>
      </div>
    </div>
    <div class="card">
      <h2>Exploration Status</h2>
      <p class="thought-block">{_esc(state.thoughts.exploration_status)}</p>
    </div>
    <div class="card">
      <h2>Exploration Directions</h2>
      {dirs_html}
    </div>
  </div>

  <div class="main">
    <div class="card">
      <h2>Search Progress Summary <span class="small">(score mode: <b>{_score_mode()}</b>)</span></h2>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              {_progress_headers_html(state)}
            </tr>
          </thead>
          <tbody>
            {"".join(_progress_row(rec, state) for rec in state.history)}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Solution Rankings ({n_total} total, {n_valid} valid) <span class="small">— sorted by <b>{_primary_label()}</b></span></h2>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>ID</th><th>{_primary_label()}</th>
              {"<th title='Mean Dice over masks'>Dice^mask</th><th title='Mean IoU over masks'>mIoU^mask</th>" if _is_seg_state(state) else "<th>MAE</th><th>mIoU</th>"}
              <th>File</th>
              <th>Key Differences</th>
              {header_extras}
              <th>Observations</th><th>Iter</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Analyzer Inventory ({len(state.analyzer_snapshots)} total)</h2>
      {ana_table}
    </div>
    <div class="card">
      <h2>Iteration History</h2>
      {history_html}
    </div>
  </div>
</div>
</body>
</html>"""


class HTMLReporter:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.html_path = os.path.join(workspace_dir, "exploration.html")

    def update(self, state: "VisionSearchState", problem_description: str) -> str:
        content = generate_html(state, problem_description)
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(content)
        return self.html_path
