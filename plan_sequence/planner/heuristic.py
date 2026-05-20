import networkx as nx
import numpy as np

from .dfa import DFASequencePlanner
from plan_sequence.physics_planner import get_contact_graph, CONTACT_EPS


class HeuristicDFASequencePlanner(DFASequencePlanner):
    """DFA variant that ranks frontier candidates by a weighted score across:

      1. `contact_distance` — shortest contact-graph distance from the candidate
         part to any already-removed part (shorter is better → negated so larger
         contributes more).
      2. `edge_count` — number of contact edges in the induced subgraph on the
         post-removal subassembly (fewer is better → negated).
      3. `z_alignment` — dot product of the disassembly action with world +z
         (higher is better).

    Weights are read from ``settings.heuristic_weights`` (project-root
    settings.py) at plan time, so they can be tuned without touching planner
    code. Missing keys fall back to ``DEFAULT_WEIGHTS``.
    """

    DEFAULT_WEIGHTS = {
        'contact_distance': 1.0,
        'edge_count': 1.0,
        'z_alignment': 1.0,
    }

    def _load_weights(self):
        cfg = None
        try:
            import settings as user_settings  # project-root settings.py
            cfg = getattr(user_settings, 'heuristic_weights', None)
        except ImportError:
            pass
        weights = dict(self.DEFAULT_WEIGHTS)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if k in weights:
                    weights[k] = float(v)
        return weights

    def _contact_graph_full(self):
        g = getattr(self, '_cg_full', None)
        if g is None:
            g = get_contact_graph(
                self.asset_folder, self.assembly_dir, self.parts,
                contact_eps=CONTACT_EPS, save_sdf=self.save_sdf,
            )
            self._cg_full = g
        return g

    def _score_child(self, weights, G_prime, sim_info, parent_G):
        cg = self._contact_graph_full()
        moved = list(set(parent_G) - set(G_prime))
        if not moved:
            return 0.0
        candidate = moved[0]
        already_removed = set(self.parts) - set(parent_G)

        # Metric 1: shortest contact-graph distance from the candidate to any
        # already-removed part. Smaller is better → negate to make larger=better.
        if not already_removed:
            m1 = 0.0
        elif not cg.has_node(candidate):
            m1 = -float(len(self.parts))
        else:
            best = None
            for p in already_removed:
                if not cg.has_node(p):
                    continue
                try:
                    d = nx.shortest_path_length(cg, candidate, p)
                except nx.NetworkXNoPath:
                    continue
                if best is None or d < best:
                    best = d
            m1 = -float(best if best is not None else len(self.parts))

        # Metric 2: induced edges on the post-removal subassembly. Fewer = better.
        present = [p for p in G_prime if cg.has_node(p)]
        m2 = -float(cg.subgraph(present).number_of_edges())

        # Metric 3: alignment of action vector with world +z (action is already
        # in world frame, rotated by the stable pose inside check_assemblable).
        action = sim_info.get('action')
        if action is None:
            m3 = 0.0
        else:
            a = np.asarray(action, dtype=float)
            n = float(np.linalg.norm(a))
            m3 = float(a[2] / n) if n > 1e-9 else 0.0

        return (
            weights['contact_distance'] * m1
            + weights['edge_count'] * m2
            + weights['z_alignment'] * m3
        )

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        if not feasible_children:
            return []
        weights = self._load_weights()
        scored = sorted(
            feasible_children,
            key=lambda triple: -self._score_child(weights, *triple),
        )
        return [G_prime for G_prime, _, _ in scored[:max_frontier]]
