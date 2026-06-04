"""Multi-selector comparison planner.

A DFA variant that, at every frontier-truncation step, asks N configurable
"selectors" (heuristic / first / random / vision-LLM) which candidate to take
next, logs ALL their picks, then advances the search by a UNIFORM RANDOM pick
from the same sample so no selector's bias compounds across iterations.

Optionally, a VLM meta-selector is shown only the candidates picked by the
constituent selectors and asked to choose among them — useful for measuring
which constituent the VLM most agrees with.

Per-step records are appended to `<log_dir>/comparison_decisions.jsonl`.
At the end of plan() the planner writes a `comparison_summary.{json,txt}`
digest covering pairwise agreement rates, match-with-random baselines, and
(when enabled) VLM-vs-constituent agreement.

Configure via `settings.comparison_planner`:

    comparison_planner = {
        'planners':       ['heuristic', 'first'],   # constituent selectors
        'l':              None,                      # subsample size; None = all
        'use_vlm_meta':   False,                     # enable VLM meta-selector
        'vlm_model':      None,                      # overrides LLM_model
        'render_size':    (512, 512),
        'cache_dir':      'comparison_cache',
    }

Supported selector names (`planners` list):
  - 'heuristic'   — HeuristicDFASequencePlanner._cost_child argmin over sample
  - 'first'       — index 0 (default DFA arrival-order behavior)
  - 'random'      — uniform random over sample (independent of the forward-pick)
  - 'llm'         — vision LLM pick over the full sample
  - 'gen:<name>'  — wrap a candidate-PART generator from
                    `plan_sequence.generator.generators` as a selector. The
                    generator's `generate_candidate_part(parent_G)` yields parts
                    in its preferred order; we pick the sample candidate whose
                    removed-part appears earliest. Available: 'gen:heur-out',
                    'gen:heur-vol', 'gen:learn', 'gen:rand', 'gen:dfa'.
                    Note: generators with a base-part filter (`heur-*`) will
                    silently exclude the base part; planners with the same
                    filter already skip base-part removal as a candidate.
"""

import json
import random as _random
from collections import Counter
from itertools import combinations
from pathlib import Path

from .heuristic import HeuristicDFASequencePlanner
from ._renders import render_part_in_context


class ComparisonDFASequencePlanner(HeuristicDFASequencePlanner):

    DEFAULT_CFG = {
        'planners': ['heuristic'],
        'l': None,
        'use_vlm_meta': False,
        'vlm_model': None,
        'render_size': (512, 512),
        'cache_dir': 'comparison_cache',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison_cfg = self._load_comparison_cfg()
        self._cmp_decisions = []
        self._cmp_decisions_path = None
        self._selectors = self._build_selectors()
        if not self._selectors:
            print('[Comparison] WARNING: no valid selectors; defaulting to ["first"].')
            self._selectors = [('first', self._pick_first)]

    # ------------------------------------------------------------------
    # configuration

    def _load_comparison_cfg(self):
        cfg = dict(self.DEFAULT_CFG)
        try:
            import settings as _us
            user = getattr(_us, 'comparison_planner', None) or {}
            if isinstance(user, dict):
                cfg.update(user)
            cfg['vlm_model'] = cfg.get('vlm_model') or getattr(_us, 'LLM_model', None)
        except ImportError:
            pass
        return cfg

    def _build_selectors(self):
        out = []
        for name in self.comparison_cfg.get('planners', []):
            if name == 'heuristic':
                out.append((name, self._pick_heuristic))
            elif name == 'llm':
                out.append((name, self._pick_llm))
            elif name in ('dfa', 'first'):
                out.append((name, self._pick_first))
            elif name == 'random':
                out.append((name, self._pick_random))
            elif name.startswith('gen:'):
                # Generator-as-selector: 'gen:heur-out', 'gen:heur-vol',
                # 'gen:learn', 'gen:rand', 'gen:dfa'. The generator's per-part
                # ranking is mapped onto the sample by picking the candidate
                # whose part_move appears earliest in the generator's output.
                out.append((name, self._make_gen_selector(name[4:])))
            else:
                print(f'[Comparison] unknown selector {name!r}; skipping')
        return out

    # ------------------------------------------------------------------
    # plan() override — emit summary at end (incl. early termination)

    def plan(self, *args, **kwargs):
        try:
            return super().plan(*args, **kwargs)
        finally:
            try:
                self._write_summary()
            except Exception as e:
                print(f'[Comparison] could not write summary: {e}')

    # ------------------------------------------------------------------
    # selectors  (signature: (tree, sample, log_dir) -> (idx|None, extras|None))

    def _pick_heuristic(self, tree, sample, log_dir):
        weights = self._load_weights()
        # All samples in a Comparison batch share one parent (max_frontier=1),
        # so look up parent_pose once and reuse it across the cost calls.
        parent_G = sample[0][2] if sample else None
        parent_pose = self._parent_pose_for(tree, parent_G) if parent_G is not None else None
        costs = [self._cost_child(weights, *t, parent_pose=parent_pose) for t in sample]
        idx = min(range(len(sample)), key=lambda i: costs[i])
        return idx, {'costs': [float(c) for c in costs]}

    def _pick_first(self, tree, sample, log_dir):
        return 0, None

    def _pick_random(self, tree, sample, log_dir):
        return _random.randrange(len(sample)), None

    # --- generator-backed selectors --------------------------------------

    def _get_generator(self, gen_name):
        """Lazily instantiate a named generator (heur-out, heur-vol, learn,
        rand, dfa) and cache it. Generators are layer-below objects that
        produce a per-part preference for a given subassembly; we collapse
        that into a single top-1 sample pick in `_make_gen_selector`."""
        cache = getattr(self, '_gen_cache', None)
        if cache is None:
            cache = {}
            self._gen_cache = cache
        if gen_name in cache:
            return cache[gen_name]
        try:
            from plan_sequence.generator import generators as _generators
            cls = _generators.get(gen_name)
            if cls is None:
                print(f'[Comparison] unknown generator {gen_name!r}; skipping')
                cache[gen_name] = None
                return None
            gen = cls(
                self.asset_folder, self.assembly_dir,
                base_part=self.base_part, save_sdf=self.save_sdf,
            )
        except Exception as e:
            print(f'[Comparison] failed to build generator {gen_name!r}: {e}')
            cache[gen_name] = None
            return None
        cache[gen_name] = gen
        return gen

    def _make_gen_selector(self, gen_name):
        """Build a selector callable that uses `gen_name`'s part ranking to
        pick a sample index. The selector returns (sample_idx, extras) where
        extras records the chosen candidate's rank in the generator's output."""
        def _fn(tree, sample, log_dir):
            gen = self._get_generator(gen_name)
            if gen is None:
                return None, {'error': f'unknown generator {gen_name!r}'}
            parent_G = sample[0][2]
            try:
                ordered_parts = list(gen.generate_candidate_part(parent_G))
            except Exception as e:
                return None, {'error': f'generate_candidate_part failed: {e}'}
            sample_parts = []
            for G_prime, _si, parent in sample:
                removed = next(iter(set(parent) - set(G_prime)), None)
                sample_parts.append(removed)
            best_idx = None
            best_rank = None
            for i, part in enumerate(sample_parts):
                try:
                    rank = ordered_parts.index(part)
                except ValueError:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                return None, {'gen_order_size': len(ordered_parts),
                              'reason': 'no sample part appeared in generator output'}
            return best_idx, {'rank': int(best_rank),
                              'gen_order_size': len(ordered_parts)}
        return _fn

    def _pick_llm(self, tree, sample, log_dir):
        cfg = self.comparison_cfg
        cache_root = Path(log_dir) / cfg.get('cache_dir', 'comparison_cache')
        size = tuple(cfg.get('render_size', (512, 512)))
        parent_G = sample[0][2]
        history_items = self._render_history(tree, parent_G, cache_root, size)
        # Randomize presentation order so the VLM's positional bias can't
        # systematically favor any sample slot. `presentation` maps the LLM-side
        # candidate index back to the original sample index.
        presentation = list(range(len(sample)))
        _random.shuffle(presentation)
        candidate_items = []
        for v_idx, s_idx in enumerate(presentation):
            G_prime, _si, parent = sample[s_idx]
            removed = next(iter(set(parent) - set(G_prime)), None)
            path = render_part_in_context(
                self.assembly_dir, list(parent), removed,
                cache_dir=str(cache_root / 'renders'), size=size,
            )
            candidate_items.append((path, f"Candidate {v_idx}: remove part {removed}"))
        try:
            import sequence_planner as _seq
            chosen, raw = _seq.choose_nodes_via_llm(
                history_items=history_items,
                candidate_items=candidate_items,
                k=1,
                model=cfg.get('vlm_model'),
                cache_dir=str(cache_root / 'llm_responses'),
            )
            if chosen:
                v_pick = int(chosen[0])
                if 0 <= v_pick < len(presentation):
                    return presentation[v_pick], {
                        'raw_response': raw,
                        'presentation_order': presentation,
                    }
            return None, {'raw_response': raw, 'presentation_order': presentation}
        except Exception as e:
            return None, {'error': str(e)}

    # ------------------------------------------------------------------
    # history rendering shared with VLM meta-selector

    def _history_for(self, tree, parent_G, depth=2):
        out = []
        cur = tuple(parent_G)
        for _ in range(depth):
            preds = list(tree.predecessors(cur))
            if not preds:
                break
            pred = preds[0]
            edata = tree.edges[pred, cur]
            removed = edata.get('sim_info', {}).get('part_move')
            if removed is None:
                break
            out.append((pred, removed))
            cur = pred
        out.reverse()
        return out

    def _render_history(self, tree, parent_G, cache_root, size):
        items = []
        for sub, removed in self._history_for(tree, parent_G, depth=2):
            path = render_part_in_context(
                self.assembly_dir, list(sub), removed,
                cache_dir=str(cache_root / 'renders'), size=size,
            )
            items.append((path, f"Removed part {removed}"))
        return items

    # ------------------------------------------------------------------
    # VLM meta-selector

    def _vlm_meta_pick(self, tree, sample, parent_G, picks, log_dir):
        """Render only the candidates picked by ≥1 constituent selector and
        ask the VLM to choose among them. Returns (sample_idx, raw_response,
        mapping_dict) where mapping maps the VLM-side candidate index back to
        the original sample index for transparency / logging.

        Presentation order is RANDOMLY SHUFFLED every call so positional bias
        in the VLM can't systematically favor any constituent. The prompt does
        NOT name which constituent picked which candidate — that would leak
        identity and bias the selection.
        """
        meta_sample_idxs = sorted({i for i in picks.values() if i is not None})
        if len(meta_sample_idxs) < 2:
            return None, 'fewer than 2 distinct constituent picks; skipped', {}
        # Shuffle the presentation. mapping[v_idx] = original sample idx.
        _random.shuffle(meta_sample_idxs)
        cfg = self.comparison_cfg
        cache_root = Path(log_dir) / cfg.get('cache_dir', 'comparison_cache')
        size = tuple(cfg.get('render_size', (512, 512)))
        history_items = self._render_history(tree, parent_G, cache_root, size)
        candidate_items = []
        mapping = {}
        for v_idx, s_idx in enumerate(meta_sample_idxs):
            G_prime, _si, parent = sample[s_idx]
            removed = next(iter(set(parent) - set(G_prime)), None)
            path = render_part_in_context(
                self.assembly_dir, list(parent), removed,
                cache_dir=str(cache_root / 'renders'), size=size,
            )
            # Label by VLM-side index only — picker names are NOT included
            # to avoid identity leakage (we still log them server-side via
            # the per-decision JSON record).
            candidate_items.append((path, f"Candidate {v_idx}: remove part {removed}"))
            mapping[v_idx] = s_idx
        try:
            import sequence_planner as _seq
            chosen, raw = _seq.choose_nodes_via_llm(
                history_items=history_items,
                candidate_items=candidate_items,
                k=1,
                model=cfg.get('vlm_model'),
                cache_dir=str(cache_root / 'vlm_meta_responses'),
            )
            if chosen:
                v_pick = int(chosen[0])
                if v_pick in mapping:
                    return mapping[v_pick], raw, mapping
            return None, raw, mapping
        except Exception as e:
            return None, f'ERROR: {e}', mapping

    # ------------------------------------------------------------------
    # main hook: frontier selection

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        if not feasible_children:
            return []
        if len(feasible_children) == 1:
            return [feasible_children[0][0]]

        l = self.comparison_cfg.get('l')
        if l is not None and len(feasible_children) > l:
            sample = _random.sample(feasible_children, l)
        else:
            sample = list(feasible_children)

        parent_G = sample[0][2]
        candidate_parts = []
        for G_prime, _si, parent in sample:
            removed = next(iter(set(parent) - set(G_prime)), None)
            candidate_parts.append(removed)

        log_dir = getattr(self, 'log_dir', None) or '/tmp/comparison_planner'

        # Each constituent picks.
        picks = {}
        extras = {}
        for name, fn in self._selectors:
            try:
                idx, extra = fn(tree, sample, log_dir)
            except Exception as e:
                print(f'[Comparison] selector {name!r} failed: {e}')
                idx, extra = None, {'error': str(e)}
            picks[name] = idx
            if extra is not None:
                extras[name] = extra

        # Optional VLM meta-selector over the constituents' choices.
        vlm_idx = None
        vlm_raw = None
        vlm_mapping = {}
        if self.comparison_cfg.get('use_vlm_meta'):
            vlm_idx, vlm_raw, vlm_mapping = self._vlm_meta_pick(
                tree, sample, parent_G, picks, log_dir,
            )

        # Uniform random is always computed (for logging + as a fallback when
        # use_vlm_for_progress is set but the VLM didn't return a valid pick).
        random_idx = _random.randrange(len(sample))

        # Choose what actually drives forward progress.
        if self.comparison_cfg.get('use_vlm_for_progress') and vlm_idx is not None:
            chosen_idx = int(vlm_idx)
            forward_source = 'vlm_meta'
        else:
            chosen_idx = int(random_idx)
            forward_source = 'random'

        decision = {
            'parent': sorted(map(str, parent_G)),
            'sample_size': len(sample),
            'sample_parts': [str(p) for p in candidate_parts],
            'planner_picks_idx': {n: i for n, i in picks.items()},
            'planner_picks_part': {
                n: (str(candidate_parts[i]) if i is not None else None)
                for n, i in picks.items()
            },
            'vlm_meta_idx': vlm_idx,
            'vlm_meta_part': (str(candidate_parts[vlm_idx])
                              if vlm_idx is not None else None),
            'vlm_meta_mapping': vlm_mapping,
            'vlm_meta_raw_response': vlm_raw,
            'random_idx': int(random_idx),
            'random_part': str(candidate_parts[random_idx]),
            'chosen_idx': chosen_idx,
            'chosen_part': str(candidate_parts[chosen_idx]),
            'forward_source': forward_source,
            'extras': extras,
        }
        self._append_decision(log_dir, decision)

        return [sample[chosen_idx][0]]

    def _append_decision(self, log_dir, decision):
        self._cmp_decisions.append(decision)
        try:
            if self._cmp_decisions_path is None:
                p = Path(log_dir) / 'comparison_decisions.jsonl'
                p.parent.mkdir(parents=True, exist_ok=True)
                self._cmp_decisions_path = p
            with open(self._cmp_decisions_path, 'a') as f:
                f.write(json.dumps(decision, default=str) + '\n')
        except OSError as e:
            print(f'[Comparison] could not write decisions: {e}')

    # ------------------------------------------------------------------
    # summary

    def _write_summary(self):
        log_dir = getattr(self, 'log_dir', None)
        if not log_dir:
            return
        decisions = self._cmp_decisions
        total = len(decisions)
        names = [n for n, _ in self._selectors]

        responded = Counter()           # name -> # decisions where pick != None
        match_random = Counter()        # name -> # times matched the always-random baseline
        match_forward = Counter()       # name -> # times matched the actually-traversed (forward) pick
        pair_agree = Counter()          # (a,b) -> # times both responded AND agreed
        pair_both_resp = Counter()      # (a,b) -> # times both responded
        forward_sources = Counter()     # 'random' / 'vlm_meta' counts
        for d in decisions:
            chosen = d['chosen_idx']
            # `random_idx` is the bias-free baseline (always uniform random).
            # Fall back to chosen_idx for legacy records that don't have it.
            baseline = d.get('random_idx', chosen)
            forward_sources[d.get('forward_source', 'random')] += 1
            picks = d['planner_picks_idx']
            for n in names:
                p = picks.get(n)
                if p is None:
                    continue
                responded[n] += 1
                if p == baseline:
                    match_random[n] += 1
                if p == chosen:
                    match_forward[n] += 1
            for a, b in combinations(sorted(names), 2):
                pa, pb = picks.get(a), picks.get(b)
                if pa is None or pb is None:
                    continue
                pair_both_resp[(a, b)] += 1
                if pa == pb:
                    pair_agree[(a, b)] += 1

        vlm_responded = 0
        vlm_match_random = 0           # vs always-random baseline
        vlm_match_forward = 0          # vs actually-traversed pick (trivially 100% when use_vlm_for_progress)
        vlm_match_constituent = Counter()
        for d in decisions:
            if d.get('vlm_meta_idx') is None:
                continue
            vlm_responded += 1
            baseline = d.get('random_idx', d['chosen_idx'])
            if d['vlm_meta_idx'] == baseline:
                vlm_match_random += 1
            if d['vlm_meta_idx'] == d['chosen_idx']:
                vlm_match_forward += 1
            for n, p in d['planner_picks_idx'].items():
                if p is not None and p == d['vlm_meta_idx']:
                    vlm_match_constituent[n] += 1

        avg_sample = (sum(d['sample_size'] for d in decisions) / total) if total else 0.0
        baseline_pct = (100.0 / avg_sample) if avg_sample > 0 else 0.0

        # ---- JSON ----
        summary = {
            'total_decisions': total,
            'selectors': names,
            'avg_sample_size': avg_sample,
            'random_baseline_pct': baseline_pct,
            'per_selector_responded': dict(responded),
            'per_selector_match_random_count': dict(match_random),
            'per_selector_match_random_rate': {
                n: (match_random[n] / responded[n]) if responded[n] else None
                for n in names
            },
            'per_selector_match_forward_count': dict(match_forward),
            'per_selector_match_forward_rate': {
                n: (match_forward[n] / responded[n]) if responded[n] else None
                for n in names
            },
            'forward_source_counts': dict(forward_sources),
            'use_vlm_for_progress': bool(self.comparison_cfg.get('use_vlm_for_progress')),
            'pairwise_agreement_count': {
                f"{a}|{b}": pair_agree[(a, b)]
                for a, b in combinations(sorted(names), 2)
            },
            'pairwise_both_responded': {
                f"{a}|{b}": pair_both_resp[(a, b)]
                for a, b in combinations(sorted(names), 2)
            },
            'pairwise_agreement_rate': {
                f"{a}|{b}": (pair_agree[(a, b)] / pair_both_resp[(a, b)])
                if pair_both_resp[(a, b)] else None
                for a, b in combinations(sorted(names), 2)
            },
            'vlm_meta_used': bool(self.comparison_cfg.get('use_vlm_meta')),
            'vlm_meta_responded': vlm_responded,
            'vlm_meta_match_random_count': vlm_match_random,
            'vlm_meta_match_random_rate': (vlm_match_random / vlm_responded) if vlm_responded else None,
            'vlm_meta_match_forward_count': vlm_match_forward,
            'vlm_meta_match_forward_rate': (vlm_match_forward / vlm_responded) if vlm_responded else None,
            'vlm_meta_match_constituent_count': dict(vlm_match_constituent),
            'vlm_meta_match_constituent_rate': {
                n: (vlm_match_constituent[n] / vlm_responded) if vlm_responded else None
                for n in names
            },
            'decisions': decisions,
        }
        try:
            with open(Path(log_dir) / 'comparison_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except OSError as e:
            print(f'[Comparison] could not write comparison_summary.json: {e}')

        # ---- TXT ----
        use_vlm_progress = bool(self.comparison_cfg.get('use_vlm_for_progress'))
        forward_desc = ('VLM meta-pick (random fallback if VLM didn\'t respond)'
                        if use_vlm_progress else 'uniform random over the sample')

        lines = []
        lines.append('Comparison Planner — multi-selector decision summary')
        lines.append(f'(forward exploration: {forward_desc}; all selectors logged regardless)')
        lines.append('=' * 80)
        lines.append(f"Selectors:                  {names}")
        lines.append(f"Total decisions:            {total}")
        lines.append(f"Avg sample size (l):        {avg_sample:.2f}")
        lines.append(f"Random baseline (1/l):      {baseline_pct:.1f}%  (chance rate of "
                     f"matching the uniform-random baseline pick)")
        lines.append(f"Forward source counts:      {dict(forward_sources)}")
        lines.append('')
        lines.append('Per-selector match with the UNIFORM-RANDOM baseline pick (bias-free):')
        lines.append(f"  {'selector':<14}  {'matched':>8}  {'responded':>9}  rate")
        for n in names:
            r = match_random[n]
            resp = responded[n]
            pct = (100.0 * r / resp) if resp else 0.0
            lines.append(f"  {n:<14}  {r:>8d}  {resp:>9d}  {pct:>5.1f}%")
        lines.append('')
        # The "forward pick" view is only meaningfully different from the
        # baseline when the VLM meta-pick was used to drive exploration.
        if use_vlm_progress:
            lines.append('Per-selector match with the FORWARD (actually-traversed) pick:')
            lines.append(f"  {'selector':<14}  {'matched':>8}  {'responded':>9}  rate")
            for n in names:
                r = match_forward[n]
                resp = responded[n]
                pct = (100.0 * r / resp) if resp else 0.0
                lines.append(f"  {n:<14}  {r:>8d}  {resp:>9d}  {pct:>5.1f}%")
            lines.append('')

        if len(names) > 1:
            lines.append('Pairwise agreement (when BOTH selectors responded):')
            lines.append(f"  {'selector A':<14}  {'selector B':<14}  {'agree':>6}  {'both':>5}  rate")
            for a, b in combinations(sorted(names), 2):
                cnt = pair_agree[(a, b)]
                base = pair_both_resp[(a, b)]
                pct = (100.0 * cnt / base) if base else 0.0
                lines.append(f"  {a:<14}  {b:<14}  {cnt:>6d}  {base:>5d}  {pct:>5.1f}%")
            lines.append('')

        if vlm_responded > 0 or self.comparison_cfg.get('use_vlm_meta'):
            lines.append('VLM meta-selector (picked among the constituents\' choices):')
            lines.append(f"  responded:                {vlm_responded}/{total}")
            if vlm_responded > 0:
                lines.append(f"  matched random baseline:  {vlm_match_random}/{vlm_responded} "
                             f"= {100.0*vlm_match_random/vlm_responded:.1f}%")
                if use_vlm_progress:
                    lines.append(f"  matched forward (chosen): {vlm_match_forward}/{vlm_responded} "
                                 f"= {100.0*vlm_match_forward/vlm_responded:.1f}%  "
                                 f"(should be 100% when use_vlm_for_progress=True and VLM responded)")
                lines.append('  matched constituent:')
                for n in names:
                    c = vlm_match_constituent[n]
                    pct = (100.0 * c / vlm_responded) if vlm_responded else 0.0
                    lines.append(f"    {n:<14}  {c:>4d}/{vlm_responded}  = {pct:>5.1f}%")
            lines.append('')

        # Per-decision table.
        col_names = names + (['vlm_meta'] if vlm_responded > 0 else [])
        hdr = (f"{'#':>3}  {'sample':>6}  "
               + "  ".join(f"{n:>12}" for n in col_names)
               + f"  {'chosen':>12}")
        lines.append(hdr)
        for i, d in enumerate(decisions):
            picks = d['planner_picks_part']
            row = [f"{i:>3}", f"{d['sample_size']:>6}"]
            for n in names:
                row.append(f"{str(picks.get(n)):>12}")
            if 'vlm_meta' in col_names:
                row.append(f"{str(d.get('vlm_meta_part')):>12}")
            row.append(f"{str(d['chosen_part']):>12}")
            lines.append('  '.join(row))

        try:
            with open(Path(log_dir) / 'comparison_summary.txt', 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError as e:
            print(f'[Comparison] could not write comparison_summary.txt: {e}')
