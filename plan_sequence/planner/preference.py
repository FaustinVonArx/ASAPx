"""Preference-learning frontier selection planner.

Online structured-perceptron weight learning on top of the heuristic cost
function. At every beam step (requires max_frontier=1):

  1. Build the feature vector phi(node) for each feasible candidate.
  2. The model picks the cheapest candidate by the current learned weights.
  3. If the model is UNCERTAIN (gap between the best two candidates < epsilon)
     AND stdin is interactive, render the candidates and ask a human which one
     to remove. The human's pick `n_star` drives a Passive-Aggressive weight
     update (see WeightOptimizer) and becomes the forward step.
  4. Otherwise the model's own pick drives forward progress uninterrupted.

The learned weights steer the live search (model-driven), and persist to disk
across runs so learning accumulates. Non-interactive / batch runs degrade
gracefully: the query is skipped (logged as `skipped_no_tty`) and the planner
behaves as the heuristic planner with the current learned weights — it never
blocks on input().

Decoupling: all the learning math lives in WeightOptimizer, which has no
knowledge of trees or beams. This planner is the thin glue that extracts
features, surfaces candidates to the human, and feeds the optimizer.
"""

import json
import sys
from pathlib import Path

import numpy as np

from .heuristic import HeuristicDFASequencePlanner, FEATURE_ORDER
from .weight_optimizer import WeightOptimizer
from ._renders import render_part_in_context


class PreferenceLearningDFASequencePlanner(HeuristicDFASequencePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pref_cfg = self._load_pref_cfg()
        self.optimizer = WeightOptimizer(
            n_features=len(FEATURE_ORDER),
            feature_names=list(FEATURE_ORDER),
            epsilon=float(self.pref_cfg.get('epsilon', 0.1)),
            save_path=self.pref_cfg.get('weights_path'),
            log_path=self.pref_cfg.get('log_path'),
            lower_is_better=True,
        )
        self._pref_decisions = []
        self._pref_decisions_path = None

    def _load_pref_cfg(self):
        cfg = {}
        try:
            import settings as _user_settings
            base = getattr(_user_settings, 'preference_planner', None) or {}
            if isinstance(base, dict):
                cfg.update(base)
        except ImportError:
            pass
        cfg.setdefault('epsilon', 0.1)
        cfg.setdefault('l', 6)
        cfg.setdefault('render_size', (512, 512))
        cfg.setdefault('cache_dir', 'preference_cache')
        cfg.setdefault('weights_path', 'assets/preference_weights.json')
        cfg.setdefault('log_path', 'assets/preference_decisions.jsonl')
        return cfg

    # Learned weights drive every _cost_child call (and the inherited cumulative
    # beam), so the live search follows what's been learned so far.
    def _load_weights(self):
        return self.optimizer.as_dict()

    def plan(self, *args, **kwargs):
        # PreferenceLearningDFASequencePlanner requires beam width 1 so the
        # candidate set the human sees at each step is unambiguous. Force the
        # invariant here (instead of asserting downstream) and warn loudly if
        # the caller asked for something else — typically settings.max_frontier
        # > 1 left over from a different planner.
        requested = kwargs.get('max_frontier', 1)
        if requested != 1:
            print(f'[PrefDFA] max_frontier={requested} requested but the preference '
                  f'planner requires 1; coercing to 1 for this run (set '
                  f'settings.max_frontier = 1 to silence this).')
        kwargs['max_frontier'] = 1
        try:
            return super().plan(*args, **kwargs)
        finally:
            try:
                self._write_pref_summary()
            except Exception as e:
                print(f'[PrefDFA] could not write preference summary: {e}')

    def _history_for(self, tree, parent_G, depth=2):
        out = []
        cur = tuple(parent_G)
        for _ in range(depth):
            preds = list(tree.predecessors(cur))
            if not preds:
                break
            pred = preds[0]
            removed = tree.edges[pred, cur].get('sim_info', {}).get('part_move')
            if removed is None:
                break
            out.append((pred, removed))
            cur = pred
        out.reverse()
        return out

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        # plan() coerces max_frontier to 1 for this planner, so the
        # downstream "all candidates share one parent" assumption is valid
        # under the normal entry path. If a caller invokes
        # _select_next_frontier directly with a wider beam (e.g. tests),
        # restrict the sample to candidates that share the first parent so
        # the human-facing logic still sees a single coherent step.
        if not feasible_children:
            return []
        if max_frontier != 1:
            print(f'[PrefDFA] _select_next_frontier invoked with max_frontier={max_frontier}; '
                  f'restricting to candidates sharing the first parent.')
            first_parent = feasible_children[0][2]
            feasible_children = [t for t in feasible_children if t[2] == first_parent]
        if len(feasible_children) == 1:
            return [feasible_children[0][0]]

        # All candidates share one parent (enforced above).
        parent_G = feasible_children[0][2]

        # Optionally cap the candidate set shown/scored at l (random subset).
        l = int(self.pref_cfg.get('l', 6))
        if len(feasible_children) <= l:
            sample = list(feasible_children)
        else:
            import random as _random
            sample = _random.sample(feasible_children, l)

        # Features + model pick. parent_pose is shared across the sample
        # (max_frontier=1 → all candidates share one parent), so look it up once.
        parent_pose = self._parent_pose_for(tree, parent_G)
        feats = [self._features_child(gp, si, pg, parent_pose=parent_pose)
                 for (gp, si, pg) in sample]
        Phi = np.vstack(feats)
        candidate_parts = [next(iter(set(pg) - set(gp)), None)
                           for (gp, si, pg) in sample]
        n_hat = self.optimizer.predict(Phi)

        wants_query = self.optimizer.should_query(Phi)
        interactive = bool(getattr(sys.stdin, 'isatty', lambda: False)())

        pick = n_hat
        status = 'no_query'
        n_star = None

        if wants_query and not interactive:
            status = 'skipped_no_tty'
        elif wants_query and interactive:
            n_star = self._ask_human(tree, parent_G, sample, candidate_parts, n_hat)
            if n_star is None:
                status = 'human_skipped'
            else:
                self.optimizer.update(Phi[n_hat], Phi[n_star])
                status = 'queried'
                pick = n_star

        self._record_decision(parent_G, candidate_parts, Phi, n_hat, n_star,
                              pick, status)
        return [sample[pick][0]]

    def _ask_human(self, tree, parent_G, sample, candidate_parts, n_hat):
        """Render the candidates, print an indexed menu, and read the human's
        pick from stdin. Returns the chosen sample index, or None to skip
        (empty / invalid input)."""
        log_dir = getattr(self, 'log_dir', None) or '/tmp/preference_planner'
        renders_dir = Path(log_dir) / self.pref_cfg.get('cache_dir', 'preference_cache') / 'renders'
        size = tuple(self.pref_cfg.get('render_size', (512, 512)))

        history = self._history_for(tree, parent_G, depth=2)
        hist_txt = ' → '.join(f'removed {r}' for _sub, r in history) or '(none)'

        print('\n' + '=' * 70)
        print(f'[preference] uncertain step — parent has {len(parent_G)} parts')
        print(f'[preference] recent history: {hist_txt}')
        print('[preference] candidates (model would pick *):')
        for i, (gp, si, pg) in enumerate(sample):
            removed = candidate_parts[i]
            # Render the candidate in its OWN step-specific stable pose so the
            # human sees how the assembly would actually be oriented when this
            # step is performed (e.g. lying on its side after a flip). Falls
            # back to canonical orientation when the planner used pose=None.
            step_pose = si.get('pose') if isinstance(si, dict) else None
            try:
                png = render_part_in_context(self.assembly_dir, list(pg), removed,
                                             cache_dir=str(renders_dir), size=size,
                                             pose=step_pose)
            except Exception as e:
                png = f'(render failed: {e})'
            star = '*' if i == n_hat else ' '
            print(f'  [{i}]{star} remove part {removed}   {png}')

        try:
            raw = input('[preference] pick index (Enter to skip): ').strip()
        except EOFError:
            return None
        if raw == '':
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f'[preference] invalid input {raw!r}; skipping.')
            return None
        if not (0 <= idx < len(sample)):
            print(f'[preference] index {idx} out of range; skipping.')
            return None
        return idx

    def _record_decision(self, parent_G, candidate_parts, Phi, n_hat, n_star,
                         pick, status):
        decision = {
            'parent': sorted(map(str, parent_G)),
            'sample_parts': [str(p) for p in candidate_parts],
            'features': [[float(v) for v in row] for row in Phi],
            'feature_order': list(FEATURE_ORDER),
            'n_hat_idx': int(n_hat),
            'n_hat_part': str(candidate_parts[n_hat]),
            'n_star_idx': (int(n_star) if n_star is not None else None),
            'n_star_part': (str(candidate_parts[n_star]) if n_star is not None else None),
            'pick_idx': int(pick),
            'pick_part': str(candidate_parts[pick]),
            'status': status,
            'weights_after': self.optimizer.as_dict(),
            'n_updates': self.optimizer.n_updates,
        }
        self._pref_decisions.append(decision)
        log_dir = getattr(self, 'log_dir', None)
        if not log_dir:
            return
        try:
            if self._pref_decisions_path is None:
                p = Path(log_dir) / 'preference_decisions.jsonl'
                p.parent.mkdir(parents=True, exist_ok=True)
                self._pref_decisions_path = p
            with open(self._pref_decisions_path, 'a') as f:
                f.write(json.dumps(decision, default=str) + '\n')
        except OSError as e:
            print(f'[PrefDFA] could not write preference_decisions.jsonl: {e}')

    def _write_pref_summary(self):
        log_dir = getattr(self, 'log_dir', None)
        if not log_dir:
            return
        decisions = self._pref_decisions
        from collections import Counter
        statuses = Counter(d['status'] for d in decisions)
        summary = {
            'total_decisions': len(decisions),
            'status_counts': dict(statuses),
            'n_updates': self.optimizer.n_updates,
            'final_weights': self.optimizer.as_dict(),
            'feature_order': list(FEATURE_ORDER),
            'epsilon': self.optimizer.epsilon,
            'decisions': decisions,
        }
        try:
            with open(Path(log_dir) / 'preference_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except OSError as e:
            print(f'[PrefDFA] could not write preference_summary.json: {e}')
