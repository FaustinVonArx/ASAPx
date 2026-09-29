import random

from .dfa import DFASequencePlanner


class RandomFrontierDFASequencePlanner(DFASequencePlanner):
    """The DFA search with random decisions: every iteration's next frontier
    is a uniform random sample of the distinct feasible next subassemblies,
    instead of a ranked (heuristic) or submission-order (plain dfa) choice.
    Everything else -- candidate checks, budget, frontier width, backtracking,
    leaf expansion -- is the DFA planner's, so against the heuristic planner
    it measures what the ranking is worth. Plain dfa is no random baseline:
    with n_success_term=None it submits parts in id order and keeps the first
    feasible ones.

    Seeded by `seed()` (the run's --seed) through its own generator, so a
    plan is reproducible and does not depend on anything else drawing from
    the global `random`. No cost, so no sequence selection either: the first
    complete sequence found is the plan."""

    def seed(self, seed):
        super().seed(seed)
        self._rng = random.Random(seed)

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        rng = getattr(self, '_rng', None)
        if rng is None:
            rng = self._rng = random.Random(0)
        distinct = list(dict.fromkeys(tuple(G_prime) for G_prime, _, _ in feasible_children))
        picked = rng.sample(distinct, min(max_frontier, len(distinct)))
        return [list(G) for G in picked]
