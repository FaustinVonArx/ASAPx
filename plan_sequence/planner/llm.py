"""LLM-guided frontier selection planner.

Drop-in replacement for HeuristicDFASequencePlanner that, at every frontier
truncation step:

  1. Randomly samples up to `l` candidates from the available feasible children
     (configured via settings.llm_planner['l']).
  2. Renders each candidate (and the previous two removal steps) as a pyvista
     PNG where the part being removed is highlighted in red.
  3. Asks a vision LLM (via sequence_planner.choose_nodes_via_llm) to pick the
     best 1 candidate.
  4. Also computes the heuristic score's pick from the same sample.
  5. Forward exploration is driven by a UNIFORM RANDOM pick from the sample
     — neither the LLM's nor the heuristic's choice steers the search. This
     keeps the trajectory unbiased so the two methods get scored on the same
     random walk and any systematic preference can't compound across iterations.
  6. All three picks (random/chosen, LLM, heuristic) are logged per step to
     `<log_dir>/llm_decisions.jsonl`; at the end of plan() a digest is written
     to `llm_summary.json` and `llm_summary.txt` in the same directory.

Token tracking: the LLM call charges tokens against the same Eval budget the
rest of the pipeline uses (set by sequence_planner.get_assembly_plans_ASAP via
module-level `_ACTIVE_EVAL`). Once the budget is exhausted the LLM call is
skipped and we fall back to the heuristic pick FOR LOGGING (the forward step
is still random), marking the decision as `fallback_token_limit` so it
doesn't pollute agreement-rate stats.

Requires max_frontier=1 (beam width 1) so the history shown to the LLM is
unambiguous; raises AssertionError otherwise.
"""

import json
import random as _random
from collections import Counter
from pathlib import Path

from .heuristic import HeuristicDFASequencePlanner
from ._renders import render_part_in_context


class LLMDFASequencePlanner(HeuristicDFASequencePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_cfg = self._load_llm_cfg()
        self._llm_decisions = []          # in-memory mirror of llm_decisions.jsonl
        self._llm_decisions_path = None   # set lazily on first append once log_dir exists

    def _load_llm_cfg(self):
        cfg = {}
        try:
            import settings as _user_settings
            base = getattr(_user_settings, 'llm_planner', None) or {}
            if isinstance(base, dict):
                cfg.update(base)
            cfg['model'] = cfg.get('model') or getattr(_user_settings, 'LLM_model', None)
        except ImportError:
            pass
        cfg.setdefault('l', 6)
        cfg.setdefault('render_size', (512, 512))
        cfg.setdefault('cache_dir', 'llm_cache')
        return cfg

    def plan(self, *args, **kwargs):
        # Write a side-by-side LLM vs heuristic summary at the end of every
        # planning run, even when an exception aborts plan() partway through.
        try:
            return super().plan(*args, **kwargs)
        finally:
            try:
                self._write_llm_summary()
            except Exception as e:
                print(f'[LLMDFA] could not write llm_summary: {e}')

    def _history_for(self, tree, parent_G, depth=2):
        """Walk up `depth` edges from `parent_G`, returning
        [(subassembly_at_step, removed_part), ...] earliest first.
        """
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

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        assert max_frontier == 1, (
            'LLMDFASequencePlanner requires max_frontier=1 (set settings.max_frontier = 1) '
            'so the disassembly history shown to the LLM is unambiguous.'
        )

        if not feasible_children:
            return []
        if len(feasible_children) == 1:
            # Trivial case — no choice to make, skip the LLM call entirely.
            return [feasible_children[0][0]]

        # With max_frontier=1, all candidates share the same parent.
        parent_G = feasible_children[0][2]

        # Random sample of l (or all if fewer available).
        l = int(self.llm_cfg.get('l', 6))
        if len(feasible_children) <= l:
            sample = list(feasible_children)
        else:
            sample = _random.sample(feasible_children, l)

        # Heuristic-best within the SAME sample, logged for offline comparison.
        weights = self._load_weights()
        heuristic_scores = [self._score_child(weights, *t) for t in sample]
        heuristic_best = max(range(len(sample)), key=lambda i: heuristic_scores[i])

        # Resolve cache root under log_dir. Fall back to /tmp when log_dir unset.
        log_dir = getattr(self, 'log_dir', None) or '/tmp/llm_planner'
        cache_root = Path(log_dir) / self.llm_cfg.get('cache_dir', 'llm_cache')
        renders_dir = cache_root / 'renders'
        responses_dir = cache_root / 'responses'

        # Render history (parent's last two removals).
        history_items = []
        for sub, removed in self._history_for(tree, parent_G, depth=2):
            path = render_part_in_context(
                self.assembly_dir, list(sub), removed,
                cache_dir=str(renders_dir),
                size=tuple(self.llm_cfg.get('render_size', (512, 512))),
            )
            history_items.append((path, f"Removed part {removed}"))

        # Render the l candidates.
        candidate_items = []
        candidate_parts = []
        for i, (G_prime, _sim_info, parent) in enumerate(sample):
            removed_set = set(parent) - set(G_prime)
            removed_part = next(iter(removed_set), None)
            candidate_parts.append(removed_part)
            path = render_part_in_context(
                self.assembly_dir, list(parent), removed_part,
                cache_dir=str(renders_dir),
                size=tuple(self.llm_cfg.get('render_size', (512, 512))),
            )
            candidate_items.append((path, f"Candidate {i}: remove part {removed_part}"))

        # Call the LLM. The function lives in the project-root sequence_planner
        # module; it's on sys.path since that script is the pipeline entry point.
        llm_choice = None
        raw_response = None
        status = 'llm_chose'
        try:
            import sequence_planner as _seq
            chosen_idxs, raw_response = _seq.choose_nodes_via_llm(
                history_items=history_items,
                candidate_items=candidate_items,
                k=1,
                model=self.llm_cfg.get('model'),
                cache_dir=str(responses_dir),
            )
            if chosen_idxs:
                llm_choice = int(chosen_idxs[0])
            elif raw_response and raw_response.startswith('SKIPPED'):
                status = 'fallback_token_limit'
            else:
                status = 'fallback_error'
        except Exception as e:
            raw_response = f"ERROR: {e}"
            status = 'fallback_error'
            print(f'[LLMDFA] LLM call failed ({e}); falling back to heuristic best.')

        if llm_choice is None or not (0 <= llm_choice < len(sample)):
            llm_choice = heuristic_best
            if status == 'llm_chose':  # defensive; shouldn't happen
                status = 'fallback_error'

        # Forward progress is driven by a UNIFORM RANDOM pick from the sample,
        # not by the LLM's choice. This keeps the search trajectory unbiased
        # so any systematic preference of either method doesn't compound across
        # iterations — both methods are scored only against the global tree the
        # random walk uncovers. LLM and heuristic picks are still recorded for
        # offline agreement analysis.
        chosen_idx = _random.randrange(len(sample))

        decision = {
            'parent': sorted(map(str, parent_G)),
            'sample_parts': [str(p) for p in candidate_parts],
            'heuristic_scores': [float(s) for s in heuristic_scores],
            'heuristic_idx': int(heuristic_best),
            'heuristic_part': str(candidate_parts[heuristic_best]) if candidate_parts else None,
            'llm_idx': int(llm_choice),
            'llm_part': str(candidate_parts[llm_choice]) if candidate_parts else None,
            'chosen_idx': int(chosen_idx),
            'chosen_part': str(candidate_parts[chosen_idx]) if candidate_parts else None,
            'agreement': llm_choice == heuristic_best,
            'llm_matches_chosen': llm_choice == chosen_idx,
            'heuristic_matches_chosen': heuristic_best == chosen_idx,
            'status': status,
            'llm_raw_response': raw_response,
        }
        self._record_decision(log_dir, decision)

        return [sample[chosen_idx][0]]

    def _record_decision(self, log_dir, decision):
        self._llm_decisions.append(decision)
        try:
            if self._llm_decisions_path is None:
                p = Path(log_dir) / 'llm_decisions.jsonl'
                p.parent.mkdir(parents=True, exist_ok=True)
                self._llm_decisions_path = p
            with open(self._llm_decisions_path, 'a') as f:
                f.write(json.dumps(decision, default=str) + '\n')
        except OSError as e:
            print(f'[LLMDFA] could not write llm_decisions.jsonl: {e}')

    def _write_llm_summary(self):
        # Always emit a summary file when plan() ran, even with zero decisions —
        # the batch aggregator uses presence/absence to distinguish "ran but
        # trivial path" from "didn't run at all".
        log_dir = getattr(self, 'log_dir', None)
        if not log_dir:
            return

        decisions = self._llm_decisions
        total = len(decisions)
        statuses = Counter(d['status'] for d in decisions)
        llm_calls = statuses.get('llm_chose', 0)
        # LLM vs heuristic agreement (only meaningful when the LLM responded).
        agreements = sum(1 for d in decisions if d['status'] == 'llm_chose' and d['agreement'])
        disagreements = [d for d in decisions if d['status'] == 'llm_chose' and not d['agreement']]
        # Agreement with the random pick that actually drove forward progress.
        # Useful as a sanity baseline: a method that "agreed" with random more
        # often than 1/l isn't necessarily better — but a method whose pick
        # was rarely on the explored path is being scored on out-of-sample data.
        llm_random_match = sum(1 for d in decisions
                                if d['status'] == 'llm_chose' and d.get('llm_matches_chosen'))
        heur_random_match = sum(1 for d in decisions
                                 if d.get('heuristic_matches_chosen'))
        # Expected baseline rate: 1 / sample size (averaged).
        avg_sample_len = (sum(len(d['sample_parts']) for d in decisions) / total) if total else 0.0
        baseline_pct = (100.0 / avg_sample_len) if avg_sample_len > 0 else 0.0

        summary_json = {
            'total_decisions': total,
            'status_counts': dict(statuses),
            'llm_responded': llm_calls,
            'avg_sample_size': avg_sample_len,
            'random_baseline_pct': baseline_pct,
            'llm_vs_heuristic_agreements': agreements,
            'llm_vs_heuristic_disagreements': len(disagreements),
            'llm_vs_heuristic_rate': (agreements / llm_calls) if llm_calls > 0 else None,
            'llm_matches_random_count': llm_random_match,
            'llm_matches_random_rate': (llm_random_match / llm_calls) if llm_calls > 0 else None,
            'heuristic_matches_random_count': heur_random_match,
            'heuristic_matches_random_rate': (heur_random_match / total) if total > 0 else None,
            'decisions': decisions,
        }

        try:
            with open(Path(log_dir) / 'llm_summary.json', 'w') as f:
                json.dump(summary_json, f, indent=2, default=str)
        except OSError as e:
            print(f'[LLMDFA] could not write llm_summary.json: {e}')

        # Human-readable text digest.
        lines = []
        lines.append('LLM vs Heuristic Frontier Selection — comparison summary')
        lines.append('(forward exploration is driven by uniform-random picks; both methods are logged)')
        lines.append('=' * 80)
        lines.append(f"Total decisions:               {total}")
        for s, c in statuses.items():
            lines.append(f"  status={s:<25} {c}")
        lines.append('')
        lines.append(f"Avg sample size (l):           {avg_sample_len:.2f}")
        lines.append(f"Random baseline (1/l):         {baseline_pct:.1f}%")
        if llm_calls > 0:
            lines.append('')
            lines.append(f"LLM vs heuristic agreement:    "
                         f"{agreements}/{llm_calls} = {100.0*agreements/llm_calls:.1f}%")
            lines.append(f"LLM picked random's choice:    "
                         f"{llm_random_match}/{llm_calls} = "
                         f"{100.0*llm_random_match/llm_calls:.1f}%")
        if total > 0:
            lines.append(f"Heuristic picked random's choice: "
                         f"{heur_random_match}/{total} = "
                         f"{100.0*heur_random_match/total:.1f}%")
        lines.append('')
        lines.append(
            f"{'#':>3}  {'parent|':>9}  {'sample':>6}  "
            f"{'heur':>10}  {'llm':>10}  {'chosen':>10}  {'status':<22}  agree"
        )
        for i, d in enumerate(decisions):
            lines.append(
                f"{i:>3}  {len(d['parent']):>9}  {len(d['sample_parts']):>6}  "
                f"{str(d['heuristic_part']):>10}  {str(d['llm_part']):>10}  "
                f"{str(d.get('chosen_part')):>10}  "
                f"{d['status']:<22}  {'Y' if d['agreement'] else 'N'}"
            )
        if disagreements:
            lines.append('')
            lines.append('Disagreements (LLM responded, picked different part than heuristic):')
            for d in disagreements:
                lines.append(
                    f"  parent={d['parent']!r}  "
                    f"heuristic→{d['heuristic_part']}  llm→{d['llm_part']}  "
                    f"chosen(random)→{d.get('chosen_part')}  "
                    f"sample={d['sample_parts']}"
                )

        try:
            with open(Path(log_dir) / 'llm_summary.txt', 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError as e:
            print(f'[LLMDFA] could not write llm_summary.txt: {e}')
