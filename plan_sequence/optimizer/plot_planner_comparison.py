"""Predicted assembly time of the planners against random decisions, on the
held-out assemblies of an evaluation with random seeds (weight_trainer
evaluate_heuristic_weights(random_seeds=N), e.g. cluster/random_baseline_submit.sh).

    python ASAPx/plan_sequence/optimizer/plot_planner_comparison.py \
        --summary <run>/eval_random/summary.json --store assets/optuna_store \
        --weights <run>/heuristic_weights.json --ids <test ids> --out <file stem>

Series, every one as the per-assembly time ratio against random (geometric
mean over its seeds), so 1 is chance and lower is faster:
  reference, first / selected    heuristic, reference weights, without / with
                                 sequence selection (the cheapest explored sequence)
  trained, first / selected      the same with the trained weights
  trained + subassemblies        the trained weights with the recursive
                                 subassembly plan, one worker (sequential)
                                 and S and R taken apart at once (parallel);
                                 where no plan was found or it could not be
                                 carried out, the flat time
The trained-weight runs come from the result store (records with those
weights); the others from the summary. Left: geometric mean with a 95%
bootstrap interval and how many assemblies beat random; right: every
assembly. Writes <out>.png and <out>.pdf.
"""
import argparse
import glob
import json
import math
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _store_series(store, weights, ids):
    """{series: {id: total_s}} of the trained-weight runs in the store."""
    out = {'trained, first': {}, 'trained, selected': {},
           'trained + subassemblies, 1 worker': {}, 'trained + subassemblies, parallel': {}}
    for path in glob.glob(os.path.join(store, 'runs', '*', '*.json')):
        if path.endswith(('assembly.json', '.weights.json')):
            continue
        with open(path) as f:
            r = json.load(f)
        fp = r.get('fingerprint') or {}
        aid = str(r.get('id'))
        if aid not in ids or fp.get('planner') != 'heuristic' or fp.get('weights') != weights:
            continue
        if r.get('status') != 'ok' or not r.get('total_s'):
            continue
        if fp.get('seq_optimizer') == 'divide':
            out['trained + subassemblies, 1 worker'][aid] = r['total_s']
            par = r.get('parallel_total_s') if r.get('split') == 'used' else None
            out['trained + subassemblies, parallel'][aid] = par or r['total_s']
        elif fp.get('sequence_selection', 'first') == 'first':
            out['trained, first'][aid] = r['total_s']
        else:
            out['trained, selected'][aid] = r['total_s']
    return out


def _bootstrap(logs, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(logs) for _ in logs) / len(logs) for _ in range(n))
    return math.exp(means[int(0.025 * n)]), math.exp(means[int(0.975 * n) - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--weights', required=True, help='the trained weights file')
    ap.add_argument('--ids', required=True, help='comma-separated held-out ids')
    ap.add_argument('--out', required=True, help='output path without extension')
    a = ap.parse_args()

    ids = [x for x in a.ids.split(',') if x]
    with open(a.summary) as f:
        summary = json.load(f)
    with open(a.weights) as f:
        weights = json.load(f)
    rows = {r['id']: r['runs'] for r in summary['per_assembly']}

    def from_summary(name):
        return {i: rows[i][name]['total_s'] for i in ids
                if i in rows and rows[i].get(name, {}).get('status') == 'ok'}

    random_t = from_summary('random')
    series = {'reference, first': from_summary('reference-first'),
              'reference, selected': from_summary('reference')}
    series.update(_store_series(a.store, weights, set(ids)))
    order = list(series)
    colors = {'reference, first': '#9ecae1', 'reference, selected': '#3182bd',
              'trained, first': '#a1d99b', 'trained, selected': '#31a354',
              'trained + subassemblies, 1 worker': '#fdae6b',
              'trained + subassemblies, parallel': '#e6550d'}

    stats = {}
    for name in order:
        common = [i for i in ids if i in series[name] and i in random_t]
        logs = [math.log(series[name][i] / random_t[i]) for i in common]
        if not logs:
            continue
        lo, hi = _bootstrap(logs)
        stats[name] = {'n': len(logs), 'geomean': math.exp(sum(logs) / len(logs)),
                       'ci': (lo, hi), 'wins': sum(x < 0 for x in logs),
                       'per_assembly': dict(zip(common, logs))}
    names = [n for n in order if n in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.1, 1]})
    y = list(range(len(names)))[::-1]
    for yi, name in zip(y, names):
        s = stats[name]
        ax1.barh(yi, s['geomean'], color=colors[name], height=0.65)
        ax1.errorbar(s['geomean'], yi, xerr=[[s['geomean'] - s['ci'][0]], [s['ci'][1] - s['geomean']]],
                     fmt='none', ecolor='black', capsize=3, lw=1)
        ax1.text(s['ci'][1] + 0.02, yi, f"x{s['geomean']:.2f}   {s['wins']}/{s['n']} faster",
                 va='center', fontsize=9)
    ax1.axvline(1.0, color='gray', ls='--', lw=1)
    ax1.text(1.01, -0.55, 'random = 1', color='gray', fontsize=9, va='center')
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlim(0, 1.45)
    ax1.set_xlabel('predicted assembly time / random  (geometric mean, 95% bootstrap CI)')
    ax1.set_title('Time relative to random decisions')

    for yi, name in zip(y, names):
        vals = [math.exp(v) for v in stats[name]['per_assembly'].values()]
        jitter = [yi + random.Random(k).uniform(-0.18, 0.18) for k in range(len(vals))]
        ax2.scatter(vals, jitter, s=18, color=colors[name], edgecolor='black', lw=0.3, zorder=3)
    ax2.axvline(1.0, color='gray', ls='--', lw=1)
    ax2.set_xscale('log')
    ticks = [0.25, 0.5, 1, 2]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f'x{t:g}' for t in ticks])
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('per assembly, time / random (log scale)')
    ax2.set_title(f'Every held-out assembly (n={len(ids)})')
    for ax in (ax1, ax2):
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    n_seeds = max((len(rows[i]['random'].get('seeds') or []) for i in ids if i in rows), default=0)
    fig.text(0.01, 0.005,
             f'Random: DFA search with each next frontier drawn at random, geometric mean over {n_seeds} seeds. '
             '"selected" = the cheapest sequence of the explored tree under the planner\'s own cost; '
             '"first" = the first one found. Subassembly runs predate that setting and pick their sequence '
             'the same way (lowest cost).', fontsize=7.5, color='#444', wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + '.png', dpi=200)
    fig.savefig(a.out + '.pdf')
    with open(a.out + '.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in stats.items()}, f, indent=2)
    for name in names:
        s = stats[name]
        print(f"{name:<36} x{s['geomean']:.3f}  CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  "
              f"faster than random {s['wins']}/{s['n']}")
    print(f'wrote {a.out}.png / .pdf / .json')


if __name__ == '__main__':
    main()
