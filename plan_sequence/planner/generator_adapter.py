"""Generator-as-cost adapter for the parallel DFA framework.

Wraps any `plan_sequence.generator.generators` entry (rand, heur-vol,
heur-out, learn, dfa) as the frontier-cost function for the parallel DFA
planner, so non-parallel single-best-pick generators can be benchmarked on
equal footing against HeuristicDFASequencePlanner without rewriting them.

How it works:
  - Each iteration the planner asks `self.seq_generator.generate_candidate_part(parent_G)`
    for the generator's preferred ordering of removable parts.
  - For each feasible child (G_prime, sim_info, parent_G) coming back from
    the parallel batch, the cost is the index of its removed part in that
    ordering (lower = preferred). Parts the generator doesn't list get the
    worst rank (len(ordered_parts)), keeping cumulative cost finite.
  - HeuristicDFASequencePlanner's cumulative-cost beam-search and
    `_cum_cost` bookkeeping carry over unchanged — only `_cost_child` is
    overridden.

Wiring:
  --planner gen-adapter         selects this class
  --generator <name>            chooses which generator drives the ranking
                                (already wired via run_seq_plan into
                                self.seq_generator)

The HeuristicDFASequencePlanner-style feature weights are ignored — the
generator emits a scalar rank directly. `_load_weights` returns a stub so
inherited callers (`_select_next_frontier`, `_print_candidate_summary`'s
score column) don't trip.
"""

from .heuristic import HeuristicDFASequencePlanner, FEATURE_ORDER


class GeneratorAdapterDFASequencePlanner(HeuristicDFASequencePlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # parent_G tuple -> [parts in generator's preferred order].
        # Generators are deterministic per (parent_G, seed) so one call per
        # parent suffices for an entire frontier evaluation.
        self._gen_order_cache: dict[tuple, list] = {}

    def _gen_order(self, parent_G):
        key = tuple(parent_G)
        cached = self._gen_order_cache.get(key)
        if cached is not None:
            return cached
        try:
            ordered = list(self.seq_generator.generate_candidate_part(parent_G))
        except Exception as e:
            print(f'[gen-adapter] generate_candidate_part({key}) failed: {e}')
            ordered = []
        self._gen_order_cache[key] = ordered
        return ordered

    def _cost_child(self, weights, G_prime, sim_info, parent_G, parent_pose=None):
        # Generator-rank cost. `weights` and `parent_pose` are accepted to
        # match the parent signature but ignored here.
        removed = next(iter(set(parent_G) - set(G_prime)), None)
        if removed is None:
            return 0.0
        order = self._gen_order(parent_G)
        try:
            return float(order.index(removed))
        except ValueError:
            return float(len(order))

    def _load_weights(self):
        # The cost function ignores weights, but the inherited frontier
        # selector + the DFA per-iteration debug table both call this and
        # then forward the dict to _cost_child. Return a unit-weight dict
        # in FEATURE_ORDER so nothing breaks.
        return {k: 1.0 for k in FEATURE_ORDER}
