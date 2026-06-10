import json
import os

import networkx as nx
import numpy as np

from .dfa import DFASequencePlanner
from plan_sequence.physics_planner import get_contact_graph, CONTACT_EPS


# Where Optuna-trained weights live on disk. Read at plan time when
# settings.heuristic_weights_source == "optuna" (a separate file from
# settings.heuristic_weights so the two can be compared head-to-head).
# Override per-call via settings.heuristic_weights_optuna_path if you need a
# different location.
_OPTUNA_WEIGHTS_DEFAULT_PATH = os.path.join('assets', 'heuristic_weights_optuna.json')


# Canonical feature ordering for the linear cost `cost = w · phi`. The weight
# dict (settings.heuristic_weights) and any learned weight vector are indexed
# in this order.
FEATURE_ORDER = ('contact_distance', 'free_dof', 'z_alignment', 'pose_change',
                 'hold_count')


class HeuristicDFASequencePlanner(DFASequencePlanner):
    """DFA variant that ranks frontier candidates by a weighted COST it
    MINIMISES. Each term is a non-negative cost where 0 is ideal:

      1. `contact_distance` — natural log of (shortest contact-graph distance
         from the candidate part to any already-removed part) + 1, i.e.
         ``ln(d+1)``. 0 when ``d=0`` (adjacent to the removed cluster) and
         grows sub-linearly, so distant candidates are penalised but not as
         harshly as a raw-hop metric. ``d`` itself is 0 when nothing is removed
         yet, and is set to the full part count when the candidate is
         disconnected from every already-removed part.
      2. `free_dof` — number of BLOCKED axes (out of 6) the candidate part has
         in the current subassembly, i.e. `len(dof) - sum(dof)`, read from the
         per-edge DoF probe. 0 = fully free part (cheapest), so we prefer
         removing parts that are already loose. Requires the planner to run
         with ``get_dof=True``; otherwise this metric contributes 0 (inert).
      3. `z_alignment` — non-upwardness of the disassembly action: `1 - z`
         where z is the unit action's world +z component. 0 = pulls straight
         up (cheapest), 2 = straight down.
      4. `pose_change` — angular dissimilarity between the parent step's stable
         pose rotation and this edge's pose rotation, normalised to [0, 1]
         (0 = identical orientation, 1 = 180° re-orientation). Penalises plans
         that ask the operator to flip / re-orient the assembly between
         consecutive steps. 0 at the root (no predecessor) and whenever either
         pose is None. Requires `parent_pose` to be threaded in by the caller;
         pass None to keep this feature inert for a given call.
      5. `hold_count` — number of EXTRA parts that must be held (beyond the
         moving gripper) for the assembly to stay stable during the step,
         taken from `len(sim_info['parts_fix'])`. 0 means the rest of the
         assembly is self-supporting and a single gripper is enough; higher
         values mean the operator/robot needs additional fixtures or hands.
         0 (inert) when `parts_fix` is None (stability check did not run or
         failed) — the feature only carries signal for feasible edges.

    All metrics are oriented so smaller = better; selection MINIMISES the
    weighted sum.

    Weights are read from ``settings.heuristic_weights`` (project-root
    settings.py) at plan time, so they can be tuned without touching planner
    code. Missing keys fall back to ``DEFAULT_WEIGHTS``.
    """

    DEFAULT_WEIGHTS = {
        'contact_distance': 1.0,
        'free_dof': 0.6,
        'z_alignment': 2.0,
        'pose_change': 2.0,
        'hold_count': 3.0,
    }

    def _load_weights(self):
        # Source selection: settings.heuristic_weights_source controls whether
        # we read settings.heuristic_weights (default) or the on-disk Optuna-
        # trained weights file. The two are kept separate so you can compare
        # default vs trained weights head-to-head without losing either set.
        source = 'default'
        optuna_path = _OPTUNA_WEIGHTS_DEFAULT_PATH
        try:
            import settings as user_settings  # project-root settings.py
            source = getattr(user_settings, 'heuristic_weights_source', 'default')
            optuna_path = getattr(user_settings, 'heuristic_weights_optuna_path',
                                  _OPTUNA_WEIGHTS_DEFAULT_PATH)
        except ImportError:
            user_settings = None

        weights = dict(self.DEFAULT_WEIGHTS)

        if source == 'optuna':
            if os.path.exists(optuna_path):
                try:
                    with open(optuna_path) as _f:
                        cfg = json.load(_f)
                    if isinstance(cfg, dict):
                        for k, v in cfg.items():
                            if k in weights:
                                weights[k] = float(v)
                    return weights
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as _e:
                    print(f"[heuristic] WARN failed to load optuna weights from "
                          f"{optuna_path} ({_e}); falling back to settings.heuristic_weights")
            else:
                print(f"[heuristic] WARN heuristic_weights_source='optuna' but "
                      f"{optuna_path} does not exist; falling back to settings.heuristic_weights")

        # Default path: settings.heuristic_weights (manually-set or hard-coded).
        cfg = None
        if user_settings is not None:
            cfg = getattr(user_settings, 'heuristic_weights', None)
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

    # `_parent_pose_for` is inherited from SequencePlanner (base.py) — moved
    # there so non-heuristic planners can use the same lookup without
    # depending on this subclass.

    def _features_child(self, G_prime, sim_info, parent_G, parent_pose=None):
        """Feature vector phi(node) for the linear cost, ordered by
        FEATURE_ORDER. Returns a length-len(FEATURE_ORDER) numpy array of
        non-negative costs (all-zero when no part moved).

        parent_pose: the 4×4 pose matrix used at the predecessor step (i.e.
        the pose stored in the in-edge of `parent_G`). Used only by the
        `pose_change` feature; pass None to keep that feature inert."""
        cg = self._contact_graph_full()
        moved = list(set(parent_G) - set(G_prime))
        if not moved:
            return np.zeros(len(FEATURE_ORDER), dtype=float)
        candidate = moved[0]
        already_removed = set(self.parts) - set(parent_G)

        # Feature 1 (contact_distance): ln(d+1) where d is the shortest
        # contact-graph distance (hops) from the candidate to any already-
        # removed part. d=0 when nothing is removed yet (→ ln(1)=0); d is set
        # to the full part count when the candidate is disconnected. The log
        # softens the penalty for distant candidates compared to a raw-hop cost.
        if not already_removed:
            d = 0.0
        elif not cg.has_node(candidate):
            d = float(len(self.parts))
        else:
            best = None
            for p in already_removed:
                if not cg.has_node(p):
                    continue
                try:
                    dist = nx.shortest_path_length(cg, candidate, p)
                except nx.NetworkXNoPath:
                    continue
                if best is None or dist < best:
                    best = dist
            d = float(best if best is not None else len(self.parts))
        c1 = float(np.log(d + 1.0))

        # Feature 2 (free_dof): blocked DoF of the candidate part in the current
        # subassembly. sim_info['dof'] is a length-6 {0,1} array (1 = part can
        # translate freely along that probe axis); blocked-axis count = len - sum,
        # so a fully free (loose) part costs 0 and we prefer removing it. Absent
        # when the planner runs without get_dof=True → contributes 0 (inert).
        dof = sim_info.get('dof')
        if dof is None:
            c2 = 0.0
        else:
            arr = np.asarray(dof, dtype=float)
            c2 = float(arr.size - arr.sum())

        # Feature 3 (z_alignment): non-upwardness of the action (already in world
        # frame, rotated by the stable pose inside check_assemblable). 1 - z:
        # 0 = straight up (cheapest), 2 = straight down. None action → 1.0.
        action = sim_info.get('action')
        if action is None:
            c3 = 1.0
        else:
            a = np.asarray(action, dtype=float)
            n = float(np.linalg.norm(a))
            c3 = (1.0 - float(a[2] / n)) if n > 1e-9 else 1.0

        # Feature 4 (pose_change): rotation angle between the predecessor's
        # stable pose and this edge's pose, normalised to [0, 1]. cos(theta)
        # = (trace(R_parent.T @ R_current) - 1) / 2; cost = (1 - cos)/2 maps
        # 0° → 0 and 180° → 1. 0 (inert) at the root or when either pose is
        # missing — same convention as the other action-dependent features.
        cur_pose = sim_info.get('pose')
        if parent_pose is None or cur_pose is None:
            c4 = 0.0
        else:
            try:
                Rp = np.asarray(parent_pose, dtype=float)[:3, :3]
                Rc = np.asarray(cur_pose, dtype=float)[:3, :3]
                R_diff = Rp.T @ Rc
                trace = float(np.trace(R_diff))
                cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
                c4 = (1.0 - cos_theta) / 2.0
            except (ValueError, TypeError):
                c4 = 0.0

        # Feature 5 (hold_count): number of extra parts that must be held
        # to keep the assembly stable during the step. sim_info['parts_fix']
        # lists those parts; +1 (the moving gripper) is constant across all
        # candidates so we leave it out. Inert (0) when parts_fix is None
        # — stability never resolved a fix list (failed or skipped).
        parts_fix = sim_info.get('parts_fix')
        if parts_fix is None:
            c5 = 0.0
        else:
            c5 = float(len(parts_fix))

        return np.array([c1, c2, c3, c4, c5], dtype=float)

    def _cost_child(self, weights, G_prime, sim_info, parent_G, parent_pose=None):
        phi = self._features_child(G_prime, sim_info, parent_G, parent_pose=parent_pose)
        w = np.array([weights[k] for k in FEATURE_ORDER], dtype=float)
        return float(w @ phi)

    def plan(self, *args, **kwargs):
        # Fresh path-cost table per plan() call. Root (full assembly) has
        # cumulative cost 0; each child's cumulative = parent's + edge cost.
        self._cum_cost = {tuple(self.parts): 0.0}
        return super().plan(*args, **kwargs)

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        """Beam-search selection by *cumulative path cost* (minimised).

        Each candidate's cost is the parent path's cumulative cost plus the
        single-edge `_cost_child`. Ties between two parents leading to the
        same G_prime are broken by min (the cheaper path "owns" that node).
        The `max_frontier` lowest cumulative costs survive.
        """
        if not feasible_children:
            return []
        weights = self._load_weights()

        # Lazy init guard if plan() wasn't entered through our override
        # (e.g. tests calling _select_next_frontier directly).
        if not hasattr(self, '_cum_cost'):
            self._cum_cost = {tuple(self.parts): 0.0}

        # Cost each candidate cumulatively; dedupe by G_prime, keeping the
        # cheapest (min) cumulative cost and remembering which (sim_info, parent)
        # produced it so the returned ordering is reproducible.
        best_for_child: dict[tuple, tuple[float, tuple]] = {}
        for triple in feasible_children:
            G_prime, sim_info, parent_G = triple
            parent_cum = self._cum_cost.get(tuple(parent_G), 0.0)
            parent_pose = self._parent_pose_for(tree, parent_G)
            edge = self._cost_child(weights, G_prime, sim_info, parent_G,
                                    parent_pose=parent_pose)
            cand_cum = parent_cum + edge
            key = tuple(G_prime)
            prev = best_for_child.get(key)
            if prev is None or cand_cum < prev[0]:
                best_for_child[key] = (cand_cum, triple)

        # Persist the chosen cumulative cost for downstream layers.
        for key, (cum, _) in best_for_child.items():
            existing = self._cum_cost.get(key)
            if existing is None or cum < existing:
                self._cum_cost[key] = cum

        ranked = sorted(best_for_child.values(), key=lambda x: x[0])
        return [triple[0] for _, triple in ranked[:max_frontier]]
