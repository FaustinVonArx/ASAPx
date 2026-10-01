"""Where the predicted assembly time goes, relative to a baseline planner
(default gen:heur-out), and how many parts have to be held, from an
evaluation summary (weight_trainer.evaluate_heuristic_weights, e.g.
<run>/eval_campaign/summary.json).

    python ASAPx/plan_sequence/optimizer/plot_components_vs_baseline.py \
        --summary <run>/eval_campaign/summary.json --out <file stem> [--baseline heur-out]

Left: per assembly, every timing component of a series divided by the
baseline's total time on that assembly, averaged over the assemblies every
series planned completely, so each assembly counts equally (as in
plot_planner_comparison) and the components still add up: the baseline's bar
is 1, split into its own shares. The whiskers are a 95% bootstrap interval of
the total. Subassembly plans with two workers are a makespan, not a sum of
steps, so they are drawn as a total only. Right: the mean number of extra
parts held per step. Writes <out>.png / .pdf / .json.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_time_components import COMPONENTS  # noqa: E402

SERIES = (  # (summary run, label, colour)
    ('random', 'random decisions', '#bdbdbd'),
    ('heur-out', 'heur-out (baseline)', '#969696'),
    ('reference', 'reference', '#3182bd'),
    ('trained', 'trained', '#31a354'),
    ('trained+split', 'trained + subassemblies, 1 worker', '#9e9ac8'),
    ('trained+split-2w', 'trained + subassemblies, 2 workers', '#54278f'),
    ('trained+split-replan', 'trained + subassemblies (re-planned), 1 worker', '#c994c7'),
    ('trained+split-replan-2w', 'trained + subassemblies (re-planned), 2 workers', '#980043'),
)


def _bootstrap_mean(values, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out', required=True, help='output path without extension')
    ap.add_argument('--baseline', default='heur-out', help='summary run every series is divided by')
    a = ap.parse_args()
    with open(a.summary) as f:
        rows = json.load(f)['per_assembly']
    series = [s for s in SERIES if any(s[0] in r['runs'] for r in rows)]
    labels = {name: label for name, label, _c in series}
    if a.baseline not in labels:
        raise SystemExit(f'baseline {a.baseline!r} not in the summary')
    base_label = labels[a.baseline].replace(' (baseline)', '')

    def ok(r, name):
        run = r['runs'].get(name) or {}
        return run.get('status') == 'ok' and run.get('total_s')

    paired = [r for r in rows if all(ok(r, name) for name, _l, _c in series)]
    n = len(paired)
    if not n:
        raise SystemExit('no assembly planned completely by every series')

    stats = {}
    for name, label, _c in series:
        runs = [r['runs'][name] for r in paired]
        base = [r['runs'][a.baseline]['total_s'] for r in paired]
        ratios = [x['total_s'] / b for x, b in zip(runs, base)]
        # Components only when every assembly has them and they add up to the
        # total (a two-worker time is a makespan).
        comps = None
        if all(x.get('components') for x in runs) and all(
                abs(sum(float(x['components'].get(c) or 0.0) for c, _l, _col in COMPONENTS)
                    - x['total_s']) < 1e-6 * max(1.0, x['total_s']) for x in runs):
            comps = {c: sum(float(x['components'].get(c) or 0.0) / b for x, b in zip(runs, base)) / n
                     for c, _l, _col in COMPONENTS}
        held = [x['metrics']['held_parts'] for x in runs
                if (x.get('metrics') or {}).get('held_parts') is not None]
        stats[name] = {'label': label, 'mean_ratio': sum(ratios) / n, 'ci': _bootstrap_mean(ratios),
                       'components_ratio': comps,
                       'held_parts_per_step': sum(held) / len(held) if len(held) == n else None}
    shown = [c for c in COMPONENTS
             if any((s['components_ratio'] or {}).get(c[0], 0.0) > 1e-6 for s in stats.values())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 1.8 + 0.62 * len(series)),
                                   gridspec_kw={'width_ratios': [1.8, 1]})
    y = list(range(len(series)))[::-1]
    for yi, (name, label, colour) in zip(y, series):
        s = stats[name]
        lo, hi = s['ci']
        if s['components_ratio']:
            left = 0.0
            for c, _cl, col in shown:
                v = s['components_ratio'][c]
                ax1.barh(yi, v, left=left, color=col, height=0.65, edgecolor='white', lw=0.5)
                # The value inside its segment, clear of the CI whisker.
                right = min(left + v, lo) if lo > left + 0.005 else left + v
                if right - left > 0.07 and not (left < lo < left + v and lo - left < 0.07):
                    ax1.text((left + right) / 2, yi, f'{v:.2f}', ha='center', va='center',
                             fontsize=7.5, color='white')
                left += v
        else:
            ax1.barh(yi, s['mean_ratio'], color='white', edgecolor=colour, hatch='//',
                     height=0.65, lw=1.2)
        ax1.errorbar(s['mean_ratio'], yi, xerr=[[s['mean_ratio'] - lo], [hi - s['mean_ratio']]],
                     fmt='none', ecolor='black', capsize=3, lw=1)
        ax1.text(hi + 0.02, yi, f"x{s['mean_ratio']:.2f}", va='center', fontsize=9)
    ax1.axvline(1.0, color='gray', ls='--', lw=1)
    handles = [Patch(facecolor=col, label=cl) for _c, cl, col in shown]
    if any(not stats[name]['components_ratio'] for name, _l, _c in series):
        handles.append(Patch(facecolor='white', edgecolor='#555555', hatch='//',
                             label='total only (two-worker makespan)'))
    ax1.set_yticks(y)
    ax1.set_yticklabels([label for _n, label, _c in series])
    ax1.set_xlabel(f'predicted assembly time / {base_label} on the same assembly (mean, 95% bootstrap CI)')
    ax1.set_title(f'Time relative to baseline, by component (n={n})')
    ax1.set_xlim(0, max(s['ci'][1] for s in stats.values()) * 1.12)
    ax1.legend(handles=handles, fontsize=8, frameon=False, loc='lower right')

    for yi, (name, label, colour) in zip(y, series):
        v = stats[name]['held_parts_per_step']
        if v is None:
            ax2.text(0.02, yi, 'not recorded', va='center', fontsize=8, color='#777',
                     transform=ax2.get_yaxis_transform())
            continue
        ax2.barh(yi, v, color=colour, height=0.65)
        ax2.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_ylim(ax1.get_ylim())
    held_max = max((s['held_parts_per_step'] or 0) for s in stats.values())
    ax2.set_xlim(0, held_max * 1.25 if held_max else 1)
    ax2.set_xlabel('extra parts held per step (mean)')
    ax2.set_title('Required holds')
    for ax in (ax1, ax2):
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + '.png', dpi=200)
    fig.savefig(a.out + '.pdf')
    with open(a.out + '.json', 'w') as f:
        json.dump({'n': n, 'baseline': a.baseline, 'series': stats}, f, indent=2)
    for name, label, _c in series:
        s = stats[name]
        comps = s['components_ratio']
        print(f"{label:<52} x{s['mean_ratio']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
              + (''.join(f"  {c[:-2]} {comps[c]:.3f}" for c, _l, _col in shown) if comps else '  (total only)')
              + (f"   held/step {s['held_parts_per_step']:.3f}" if s['held_parts_per_step'] is not None else ''))
    print(f'wrote {a.out}.png / .pdf / .json')


if __name__ == '__main__':
    main()
