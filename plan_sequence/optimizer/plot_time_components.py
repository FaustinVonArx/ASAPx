"""Where the predicted assembly time goes, per planner, and how many parts have
to be held, from an evaluation summary (weight_trainer.evaluate_heuristic_weights,
e.g. <run>/eval_campaign/summary.json).

    python ASAPx/plan_sequence/optimizer/plot_time_components.py \
        --summary <run>/eval_campaign/summary.json --out <file stem>

Left: the mean time per assembly split into the timing model's components
(part removal motion, base travel around the assembly, reorientation of the
assembly; transitions and hold time only when non-zero), over the assemblies
every series planned completely, so the bars compare the same assemblies.
The two-worker subassembly time is a makespan, not a sum of steps, so it is
drawn as a total only. Right: the mean number of extra parts held per
assembly (sum over its steps of the parts that must be held for the rest to
stay stable), where the summary has it: plan metrics are recorded for flat
sequences, not for subassembly plans. Writes <out>.png / .pdf / .json.
"""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SERIES = (  # (summary run, label, colour)
    ('random', 'random decisions', '#bdbdbd'),
    ('heur-out', 'heur-out (baseline)', '#969696'),
    ('reference', 'reference, selected', '#3182bd'),
    ('trained', 'trained, selected', '#31a354'),
    ('trained+best-1w', 'trained + subassemblies, 1 worker', '#9e9ac8'),
    ('trained+best-2w', 'trained + subassemblies, 2 workers', '#54278f'),
)
COMPONENTS = (  # (component, label, colour)
    ('step_disassembly_s', 'part removal motion', '#4c72b0'),
    ('base_travel_s', 'base travel around the assembly', '#dd8452'),
    ('reorientation_s', 'reorienting the assembly', '#55a868'),
    ('transitions_s', 'arm transitions', '#8172b3'),
    ('hold_s', 'holding parts', '#c44e52'),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out', required=True, help='output path without extension')
    a = ap.parse_args()
    with open(a.summary) as f:
        summary = json.load(f)
    rows = summary['per_assembly']
    series = [s for s in SERIES if any(s[0] in r['runs'] for r in rows)]

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
        total = sum(x['total_s'] for x in runs) / n
        with_comp = [x for x in runs if x.get('components')]
        comps = None
        if len(with_comp) == n:
            comps = {c: sum(float(x['components'].get(c) or 0.0) for x in runs) / n
                     for c, _l, _col in COMPONENTS}
        held = [x['metrics']['held_parts'] * (r['n_parts'] - 1)
                for r, x in zip(paired, runs) if x.get('metrics')]
        held_step = [x['metrics']['held_parts'] for x in runs if x.get('metrics')]
        stats[name] = {'label': label, 'mean_total_s': total, 'components_s': comps,
                       'held_parts_per_assembly': sum(held) / len(held) if len(held) == n else None,
                       'held_parts_per_step': sum(held_step) / len(held_step) if len(held_step) == n else None}
    shown = [c for c in COMPONENTS
             if any((s['components_s'] or {}).get(c[0], 0.0) > 1e-6 for s in stats.values())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 1.8 + 0.62 * len(series)),
                                   gridspec_kw={'width_ratios': [1.6, 1]})
    y = list(range(len(series)))[::-1]
    for yi, (name, label, colour) in zip(y, series):
        s = stats[name]
        if s['components_s']:
            left = 0.0
            for c, _cl, col in shown:
                v = s['components_s'][c]
                ax1.barh(yi, v, left=left, color=col, height=0.65, edgecolor='white', lw=0.5)
                left += v
        else:
            ax1.barh(yi, s['mean_total_s'], color='white', edgecolor=colour, hatch='//',
                     height=0.65, lw=1.2)
        ax1.text(s['mean_total_s'] + 0.5, yi, f"{s['mean_total_s']:.1f} s", va='center', fontsize=9)
    handles = [Patch(facecolor=col, label=cl) for _c, cl, col in shown]
    if any(not stats[name]['components_s'] for name, _l, _c in series):
        handles.append(Patch(facecolor='white', edgecolor='#54278f', hatch='//',
                             label='total only (parallel makespan)'))
    ax1.set_yticks(y)
    ax1.set_yticklabels([label for _n, label, _c in series])
    ax1.set_xlabel('mean predicted time per assembly (s)')
    ax1.set_title(f'Time components (n={n} assemblies every series planned)')
    ax1.legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    ax1.set_xlim(0, max(s['mean_total_s'] for s in stats.values()) * 1.18)

    for yi, (name, label, colour) in zip(y, series):
        s = stats[name]
        if s['held_parts_per_assembly'] is None:
            ax2.text(0.02, yi, 'not recorded for subassembly plans', va='center', fontsize=8,
                     color='#777', transform=ax2.get_yaxis_transform())
            continue
        ax2.barh(yi, s['held_parts_per_assembly'], color=colour, height=0.65)
        ax2.text(s['held_parts_per_assembly'] + 0.05, yi,
                 f"{s['held_parts_per_assembly']:.2f}  ({s['held_parts_per_step']:.2f} per step)",
                 va='center', fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_ylim(ax1.get_ylim())
    held_max = max((s['held_parts_per_assembly'] or 0) for s in stats.values())
    ax2.set_xlim(0, held_max * 1.6 if held_max else 1)
    ax2.set_xlabel('extra parts held, per assembly (mean)')
    ax2.set_title('Required holds')
    for ax in (ax1, ax2):
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    note = ('Means over the same assemblies (larger assemblies weigh more in seconds). "selected" = cheapest '
            'explored sequence under the planner\'s cost; subassemblies = the plan where the timing model '
            'predicts it faster than the flat sequence, else the flat sequence. Holds: parts that must be held '
            'for the rest to stay stable during a step, summed over the steps.')
    fig.text(0.01, 0.005, '\n'.join(textwrap.wrap(note, 190)), fontsize=7.5, color='#444')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + '.png', dpi=200)
    fig.savefig(a.out + '.pdf')
    with open(a.out + '.json', 'w') as f:
        json.dump({'n': n, 'series': stats}, f, indent=2)
    for name, label, _c in series:
        s = stats[name]
        comps = s['components_s']
        print(f"{label:<36} {s['mean_total_s']:6.1f} s"
              + (''.join(f"  {c[:-2]} {comps[c]:5.1f}" for c, _l, _col in shown) if comps else '  (total only)')
              + (f"   holds {s['held_parts_per_assembly']:.2f}/assembly" if s['held_parts_per_assembly'] is not None else ''))
    print(f'wrote {a.out}.png / .pdf / .json')


if __name__ == '__main__':
    main()
