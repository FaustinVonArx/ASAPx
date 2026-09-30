"""Predicted assembly time of the planners against random decisions, on the
held-out assemblies of an evaluation with random seeds (weight_trainer
evaluate_heuristic_weights(random_seeds=N), e.g. cluster/random_baseline_submit.sh).

    python ASAPx/plan_sequence/optimizer/plot_planner_comparison.py \
        --summary <run>/eval_campaign/summary.json --out <file stem> [--ids <ids>]
        [--store assets/optuna_store --weights <run>/heuristic_weights.json]

Series, every one as the per-assembly time ratio against random (geometric
mean over its seeds), so 1 is chance and lower is faster; each drawn when the
summary has it:
  heur-out (baseline)            the gen:heur-out baseline
  reference, first / selected    heuristic, reference weights, without / with
                                 sequence selection (the cheapest explored sequence)
  trained, selected              the same with the trained weights
  trained + subassemblies        the trained weights with the recursive
                                 subassembly plan, one and two workers (S and R
                                 of the outermost split at once), carried out
                                 where the timing model predicts it faster than
                                 'trained, selected' (the planner's default),
                                 else the flat sequence
  ... (always split)             with --show-forced (or summaries without the
                                 choice): the plan carried out on every
                                 assembly that has one; every split at once
                                 ('parallel') for older summaries
With --store and --weights, also 'trained, first' from the result store
(records with those weights and no sequence selection). --ids defaults to
every assembly of the summary. Left: geometric mean with a 95%
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
import matplotlib.ticker
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
    ap.add_argument('--store', default=None, help="with --weights: add 'trained, first'")
    ap.add_argument('--weights', default=None, help='the trained weights file')
    ap.add_argument('--ids', default=None, help='comma-separated ids (default: all in the summary)')
    ap.add_argument('--out', required=True, help='output path without extension')
    ap.add_argument('--show-forced', action='store_true',
                    help='also draw the subassembly plan carried out on every assembly '
                         '(by default only where it is predicted faster)')
    a = ap.parse_args()

    with open(a.summary) as f:
        summary = json.load(f)
    rows = {r['id']: r['runs'] for r in summary['per_assembly']}
    ids = [x for x in a.ids.split(',') if x] if a.ids else list(rows)

    def from_summary(name):
        return {i: rows[i][name]['total_s'] for i in ids
                if i in rows and rows[i].get(name, {}).get('status') == 'ok'}

    random_t = from_summary('random')
    stored = {}
    if a.store and a.weights:
        with open(a.weights) as f:
            stored = _store_series(a.store, json.load(f), set(ids))
    two = from_summary('trained+split-2w')
    # Summaries from before the subassembly runs joined the random evaluation
    # lack them; the store has them then.
    one = from_summary('trained+split') or stored.get('trained + subassemblies, 1 worker', {})
    par = {} if two else (from_summary('trained+split-par')
                          or stored.get('trained + subassemblies, parallel', {}))
    # The subassembly plan is carried out where the timing model predicts it
    # faster than the flat sequence (the planner's default); summaries from
    # before that choice only have it carried out everywhere.
    best1, best2 = from_summary('trained+best-1w'), from_summary('trained+best-2w')
    forced = a.show_forced or not (best1 or best2)
    series = {'heur-out (baseline)': from_summary('heur-out'),
              'reference, first': from_summary('reference-first'),
              'reference, selected': from_summary('reference'),
              'trained, first': stored.get('trained, first', {}),
              'trained, selected': from_summary('trained'),
              'trained + subassemblies, 1 worker': best1,
              'trained + subassemblies, 2 workers': best2,
              'trained + subassemblies (always split), 1 worker': one if forced else {},
              'trained + subassemblies (always split), 2 workers': two if forced else {},
              'trained + subassemblies (always split), parallel': par if forced else {}}
    order = [n for n, v in series.items() if v]
    colors = {'heur-out (baseline)': '#969696', 'reference, first': '#9ecae1', 'reference, selected': '#3182bd',
              'trained, first': '#a1d99b', 'trained, selected': '#31a354',
              'trained + subassemblies, 1 worker': '#9e9ac8',
              'trained + subassemblies, 2 workers': '#54278f',
              'trained + subassemblies (always split), 1 worker': '#fdae6b',
              'trained + subassemblies (always split), 2 workers': '#e6550d',
              'trained + subassemblies (always split), parallel': '#e6550d'}

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 1.6 + 0.62 * len(names)), gridspec_kw={'width_ratios': [1.1, 1]})
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
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ticks = [0.25, 0.5, 1, 2]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f'x{t:g}' for t in ticks])
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('per assembly, time / random (log scale)')
    ax2.set_title(f'Every assembly (n={len(random_t)})')
    for ax in (ax1, ax2):
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    n_seeds = max((len(rows[i]['random'].get('seeds') or []) for i in ids if i in rows), default=0)
    note = (f'Random: DFA search with each next frontier drawn at random, '
            f'geometric mean over {n_seeds} seed{"s" if n_seeds != 1 else ""}. '
            '"selected" = the cheapest sequence of the explored tree under the planner\'s own cost'
            + ('; "first" = the first one found.' if any(n.endswith('first') for n in names) else '.'))
    if best1 or best2:
        note += (' Subassemblies: the plan is carried out where the timing model predicts it '
                 'faster than the flat sequence with as many workers, else the flat sequence.')
    elif 'trained + subassemblies (always split), 1 worker' in names and not two:
        note += (' Subassembly runs predate that setting and pick their sequence the same way '
                 '(lowest cost); "parallel" = every split at once.')
    fig.text(0.01, 0.005, note, fontsize=7.5, color='#444', wrap=True)
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

    # By size: the same ratio per band of part counts, one line per series.
    n_parts = {r['id']: r['n_parts'] for r in summary['per_assembly']}
    bands = [(lo, hi) for lo, hi in ((5, 9), (10, 14), (15, 20), (21, 25), (26, 30))
             if any(lo <= n_parts[i] <= hi for i in random_t)]
    if len(bands) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        xs, ys, los, his, ns = [], [], [], [], []
        for b, (lo, hi) in enumerate(bands):
            logs = [v for i, v in stats[name]['per_assembly'].items() if lo <= n_parts[i] <= hi]
            if len(logs) < 2:
                continue
            g = math.exp(sum(logs) / len(logs))
            ci = _bootstrap(logs)
            xs.append(b - 0.4 + width * (k + 0.5))
            ys.append(g)
            los.append(g - ci[0])
            his.append(ci[1] - g)
            ns.append(len(logs))
        ax.errorbar(xs, ys, yerr=[los, his], fmt='o', color=colors[name], ecolor=colors[name],
                    capsize=2, ms=5, label=name, mec='black', mew=0.4)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    counts = []
    for lo, hi in bands:
        counts.append(sum(lo <= n_parts[i] <= hi for i in random_t))
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{lo}-{hi} parts\n(n={c})' for (lo, hi), c in zip(bands, counts)])
    ax.set_ylabel('time / random (geometric mean, 95% CI)')
    ax.set_title('By assembly size')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(a.out + '_by_size.png', dpi=200)
    fig.savefig(a.out + '_by_size.pdf')
    print(f'wrote {a.out}_by_size.png / .pdf')


if __name__ == '__main__':
    main()
