"""Plot diagnostics from an Optuna weight-training history file (the JSON
written incrementally by ``weight_trainer.train_heuristic_weights``).

Produces a single figure with three sections:

1. **Convergence curve** — per-trial objective as a scatter plus a
   running best-so-far line. Quickly answers "is training improving the
   objective?" Pruned trials (a partial estimate) are drawn hollow and
   failing ones as red crosses; only fully evaluated trials count as best.
2. **Per-weight sensitivity** — one scatter per weight (weight value vs
   objective), with marker color scaling from old (light) to new (dark)
   trials. Reveals which weights actually move the objective and roughly
   where the optimum sits.
3. **Per-assembly trajectory** — one faint line per training assembly
   across trials. Shows whether gains are broad-spectrum or carried by a
   few assemblies (and flags any assembly that consistently fails).

The objective is the geometric-mean time ratio against the baseline
(``geomean_ratio``, per assembly ``per_assembly_ratio``). History files
written before the trainer had a baseline carry ``mean_total_s`` /
``per_assembly_total_s`` instead, and are plotted in seconds.

Use either programmatically::

    from plan_sequence.optimizer.plot_weight_history import plot_weight_history
    plot_weight_history("assets/heuristic_weights_optuna_history.json",
                        save_path="assets/optuna_training/history.png")

or from the CLI (run from the repo root)::

    python ASAPx/plan_sequence/optimizer/plot_weight_history.py \
        --history assets/heuristic_weights_optuna_history.json \
        --out     assets/optuna_training/history.png

(``python -m plan_sequence.optimizer.plot_weight_history`` needs ASAPx on
``sys.path`` — either ``cd ASAPx`` first or set ``PYTHONPATH=ASAPx``.)
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEIGHT_KEYS = ('contact_distance', 'free_dof', 'z_alignment', 'pose_change',
               'hold_count')


def _load_history(history):
    """Accept either a path-like or an already-loaded list of trial dicts."""
    if isinstance(history, (str, Path)):
        with open(history) as f:
            data = json.load(f)
    else:
        data = list(history)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of trial dicts, got {type(data).__name__}")
    return sorted(data, key=lambda e: e.get('trial', 0))


def _running_min(xs):
    out = []
    cur = math.inf
    for x in xs:
        if x is not None and x < cur:
            cur = x
        out.append(cur if cur != math.inf else None)
    return out


def _objective(entries):
    """Per-entry objective values plus how to label them: the time ratio
    against the baseline when the history has one, seconds otherwise."""
    if any('geomean_ratio' in e for e in entries):
        return ([e.get('geomean_ratio') for e in entries], 'per_assembly_ratio',
                'time vs baseline (geometric mean)', 'time vs baseline', '{:.3f}x')
    return ([e.get('mean_total_s') for e in entries], 'per_assembly_total_s',
            'mean total assembly time (s)', 'per-assembly total_s', '{:.2f}s')


def _is_complete(e):
    # Legacy entries have no outcome; every one of them was fully evaluated.
    return e.get('outcome', 'complete') == 'complete'


def plot_weight_history(history, save_path=None, show=False, title_suffix=''):
    """Render the training history as a 3-section diagnostic figure.

    Args:
        history: either a path to the history JSON or an already-loaded
            list of trial-entry dicts.
        save_path: optional path to write a PNG. Parent dirs are created.
        show: when True, calls ``plt.show()`` (blocks). Default False.
        title_suffix: appended to the figure suptitle.

    Returns:
        The matplotlib Figure object.
    """
    entries = _load_history(history)
    if not entries:
        raise ValueError("history is empty")

    trials = np.array([e.get('trial', i) for i, e in enumerate(entries)])
    means, per_key, y_label, per_label, fmt = _objective(entries)
    finite_mask = np.array([m is not None and math.isfinite(m) for m in means])
    means_arr = np.array([m if (m is not None and math.isfinite(m)) else np.nan
                          for m in means])
    complete_mask = finite_mask & np.array([_is_complete(e) for e in entries])
    pruned_mask = finite_mask & ~complete_mask
    complete_arr = np.where(complete_mask, means_arr, np.nan)

    best_so_far = _running_min([m if c else None for m, c in zip(means, complete_mask)])
    best_so_far_arr = np.array([b if b is not None else np.nan for b in best_so_far])

    # Identify best trial overall (fully evaluated trials only).
    if complete_mask.any():
        best_idx = int(np.nanargmin(complete_arr))
        best_trial_num = int(trials[best_idx])
        best_value = float(means_arr[best_idx])
    else:
        best_idx = None
        best_trial_num = None
        best_value = None

    # Collect per-assembly time series.
    assembly_ids = set()
    for e in entries:
        per = e.get(per_key) or {}
        assembly_ids.update(per.keys())
    assembly_ids = sorted(assembly_ids)

    # ---- Layout ----
    n_weights = len(WEIGHT_KEYS)
    fig = plt.figure(figsize=(4 + 2.4 * n_weights, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=n_weights,
        height_ratios=[1.2, 1.0, 1.0],
        hspace=0.45, wspace=0.35,
    )

    # ---- Section 1: convergence ----
    ax_conv = fig.add_subplot(gs[0, :])
    ax_conv.scatter(trials[complete_mask], means_arr[complete_mask],
                    s=25, alpha=0.55, color='steelblue', label='per-trial objective')
    if pruned_mask.any():
        ax_conv.scatter(trials[pruned_mask], means_arr[pruned_mask],
                        s=25, alpha=0.55, facecolors='none', edgecolors='steelblue',
                        label='pruned (partial estimate)')
    bad = ~finite_mask
    if bad.any():
        ymin = float(np.nanmin(means_arr)) if finite_mask.any() else 0.0
        ax_conv.scatter(trials[bad], np.full(bad.sum(), ymin),
                        marker='x', color='red', s=40, label='failed trial')
    ax_conv.plot(trials, best_so_far_arr, color='darkorange', lw=2,
                 label='best so far')
    if best_idx is not None:
        ax_conv.scatter([best_trial_num], [best_value], marker='*', s=220,
                        color='darkorange', edgecolor='black', zorder=5,
                        label=f'best (trial {best_trial_num}, {fmt.format(best_value)})')
    ax_conv.set_xlabel('trial number')
    ax_conv.set_ylabel(y_label)
    ax_conv.set_title('Convergence')
    ax_conv.legend(loc='upper right', fontsize=8)
    ax_conv.grid(alpha=0.3)

    # ---- Section 2: per-weight sensitivity ----
    # Color trials by recency (light → dark with trial number).
    if finite_mask.any():
        norm = plt.Normalize(vmin=trials.min(), vmax=trials.max())
        cmap = plt.get_cmap('viridis')
    for col, key in enumerate(WEIGHT_KEYS):
        ax = fig.add_subplot(gs[1, col])
        values = np.array([(e.get('weights') or {}).get(key, np.nan) for e in entries],
                          dtype=float)
        v_mask = finite_mask & ~np.isnan(values)
        # Log-uniform search: spread the axis the way it was sampled. A weight
        # held fixed is a single value; say so instead of plotting a column.
        v_seen = values[~np.isnan(values)]
        fixed = v_seen.size > 0 and np.all(v_seen == v_seen[0])
        if fixed:
            ax.set_title(f'fixed at {v_seen[0]:.3g}', fontsize=8)
        elif v_seen.size and v_seen.min() > 0 and v_seen.max() / v_seen.min() > 20:
            ax.set_xscale('log')
        if v_mask.any():
            ax.scatter(values[v_mask], means_arr[v_mask],
                       c=trials[v_mask], cmap=cmap, norm=norm,
                       s=30, alpha=0.75, edgecolor='none')
            if best_idx is not None and not np.isnan(values[best_idx]):
                ax.scatter([values[best_idx]], [best_value], marker='*',
                           s=160, color='darkorange', edgecolor='black', zorder=5)
        ax.set_xlabel(key, fontsize=9)
        if col == 0:
            ax.set_ylabel(y_label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    # Colorbar for the row (anchored on the last weight axis).
    if finite_mask.any():
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.axes[1:1 + n_weights],
                            orientation='vertical', shrink=0.85, pad=0.02)
        cbar.set_label('trial #', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    # ---- Section 3: per-assembly trajectory ----
    ax_per = fig.add_subplot(gs[2, :])
    if assembly_ids:
        # One faint line per assembly. Failed entries plot as gaps (np.nan).
        for aid in assembly_ids:
            ys = []
            for e in entries:
                per = e.get(per_key) or {}
                v = per.get(aid)
                ys.append(v if (v is not None and math.isfinite(v)) else np.nan)
            ax_per.plot(trials, ys, marker='.', ms=4, alpha=0.55, lw=0.8,
                        label=str(aid))
        # Mean line (over assemblies present per trial), bold on top.
        mean_per_trial = []
        for e in entries:
            per = e.get(per_key) or {}
            vals = [v for v in per.values()
                    if v is not None and math.isfinite(v)]
            mean_per_trial.append(sum(vals) / len(vals) if vals else np.nan)
        ax_per.plot(trials, mean_per_trial, color='black', lw=2,
                    label='mean across assemblies')
        # Legend gets unwieldy with many assemblies; cap at ~10 entries.
        if len(assembly_ids) <= 10:
            ax_per.legend(loc='upper right', fontsize=7, ncol=2)
        else:
            ax_per.legend(['mean across assemblies'], loc='upper right',
                          fontsize=8)
    if per_key == 'per_assembly_ratio':
        ax_per.axhline(1.0, color='grey', lw=1, ls='--')
    ax_per.set_xlabel('trial number')
    ax_per.set_ylabel(per_label)
    ax_per.set_title(f'Per-assembly trajectory ({len(assembly_ids)} assemblies)')
    ax_per.grid(alpha=0.3)

    n_total = len(entries)
    n_ok = int(complete_mask.sum())
    suptitle = (f'Optuna weight training — {n_ok}/{n_total} fully evaluated trials'
                + (f'  ·  best: {fmt.format(best_value)} (trial {best_trial_num})'
                   if best_value is not None else ''))
    if title_suffix:
        suptitle += f'  ·  {title_suffix}'
    fig.suptitle(suptitle, fontsize=12, y=0.995)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'[plot_weight_history] wrote {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Plot Optuna weight-training history.')
    parser.add_argument('--history', type=str,
                        default='assets/heuristic_weights_optuna_history.json',
                        help='Path to the history JSON.')
    parser.add_argument('--out', type=str,
                        default='assets/optuna_training/history.png',
                        help='Output PNG path.')
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively.')
    parser.add_argument('--title-suffix', type=str, default='',
                        help='Extra text appended to the figure suptitle.')
    args = parser.parse_args()
    plot_weight_history(args.history, save_path=args.out,
                        show=args.show, title_suffix=args.title_suffix)
