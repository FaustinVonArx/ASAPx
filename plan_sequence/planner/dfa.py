import traceback
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import networkx as nx

from .base import SequencePlanner, _simulate_standalone_tagged
from plan_sequence.physics_planner import get_contact_graph, CONTACT_EPS
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses
from utils.parallel import parallel_execute
import settings


class DFASequencePlanner(SequencePlanner):

    G_path = None

    # Edge / node colors keyed by simulation outcome. Failure reasons are split
    # so the tree shows *why* an edge is infeasible (assembly vs. stability vs.
    # tool) rather than a single red.
    _OUTCOME_COLORS = {
        'feasible':   '#2ca02c',  # green
        'unfinished': '#ffd700',  # gold
        'assembly':   '#d62728',  # red
        'stability':  '#1f77b4',  # blue
        'tool':       '#ff7f0e',  # orange
    }

    @staticmethod
    def _classify_sim_info(sim_info):
        if sim_info.get('unfinished', False):
            return 'unfinished'
        if sim_info.get('feasible', False):
            return 'feasible'
        reason = sim_info.get('fail_reason')
        if reason in DFASequencePlanner._OUTCOME_COLORS:
            return reason
        # Fallback for sim_info dicts written before fail_reason existed.
        if sim_info.get('action') is None:
            return 'assembly'
        if sim_info.get('parts_fix') is None:
            return 'stability'
        return 'assembly'

    @staticmethod
    def plot_tree(tree, save_path=None):
        import os
        import tempfile
        from collections import Counter
        from matplotlib.patches import Patch

        colors = DFASequencePlanner._OUTCOME_COLORS
        classify = DFASequencePlanner._classify_sim_info

        def _edge_kind(edge):
            return classify(tree.edges[edge]['sim_info'])

        def _node_kind(node):
            if tree.nodes[node]['n_gripper'] is not None:
                return 'feasible'
            incoming = list(tree.in_edges(node))
            if not incoming:
                return 'feasible'  # root
            kinds = [_edge_kind(e) for e in incoming]
            if all(k == 'unfinished' for k in kinds):
                return 'unfinished'
            fail_kinds = [k for k in kinds if k not in ('feasible', 'unfinished')]
            if not fail_kinds:
                return 'assembly'
            return Counter(fail_kinds).most_common(1)[0][0]

        node_colors = [colors[_node_kind(n)] for n in tree.nodes]
        edge_colors = [colors[_edge_kind(e)] for e in tree.edges]

        # graphviz_layout is the preferred hierarchical layout but it depends
        # on pygraphviz (system `graphviz` + python `pygraphviz`). The
        # backend-loading dance occasionally fails in fresh subprocess
        # interpreters even when it works fine in the parent shell — fall
        # through to pydot, then to a spring layout, so the plot is at least
        # produced rather than killing the planning loop.
        pos = None
        for _attempt in ('agraph', 'pydot', 'spring'):
            try:
                if _attempt == 'agraph':
                    from networkx.drawing.nx_agraph import graphviz_layout
                    pos = graphviz_layout(tree, prog='dot')
                elif _attempt == 'pydot':
                    from networkx.drawing.nx_pydot import graphviz_layout as _pydot_layout
                    pos = _pydot_layout(tree, prog='dot')
                else:
                    pos = nx.spring_layout(tree, seed=0)
                break
            except Exception as _layout_err:
                print(f'[DFA.plot_tree] layout backend {_attempt!r} failed: {_layout_err}')
        if pos is None:
            # Even spring failed (extremely unlikely) — bail gracefully.
            print('[DFA.plot_tree] no layout backend available; skipping plot.')
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        nx.draw(tree, pos, ax=ax, node_color=node_colors, edge_color=edge_colors,
                with_labels=False, node_size=40, arrowsize=6)
        legend = [
            Patch(facecolor=colors['feasible'],   label='feasible'),
            Patch(facecolor=colors['unfinished'], label='unfinished'),
            Patch(facecolor=colors['assembly'],   label='assembly fail (A)'),
            Patch(facecolor=colors['stability'],  label='stability fail (S)'),
            Patch(facecolor=colors['tool'],       label='tool fail (T)'),
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=9, framealpha=0.9)
        out = save_path or tempfile.mktemp(suffix='.png', prefix='dfa_tree_')
        try:
            # Make sure the parent directory exists (in case log_dir resolution
            # took an unexpected turn in a subprocess).
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            fig.savefig(out, dpi=120, bbox_inches='tight')
            print(f'[DFA.plot_tree] tree → {out}')
        except Exception as _save_err:
            print(f'[DFA.plot_tree] savefig({out!r}) failed: {_save_err}')
        finally:
            plt.close(fig)

    def _reset(self):
        G0 = self.parts.copy()
        self.G_path = [G0]

    def _select_node(self, tree):
        # Pop exhausted nodes off the DFS path. A node is "live" if it still has
        # unexpanded candidates (out_degree < len) or any 'unfinished' child edge
        # left over from an early-terminated batch.
        while self.G_path:
            G = self.G_path[-1]
            if tree.out_degree(tuple(G)) < len(G) or self._has_unfinished_children(tree, G):
                return G
            self.G_path.pop()

        # Stack drained but the tree isn't fully explored — happens when an
        # n_success_term-bounded batch left 'unfinished' children somewhere off
        # the DFS spine, or when the dive backtracked past nodes that still have
        # missing edges. Seed the path from any such node so we can resume.
        fallback = self._find_resumable_node(tree)
        if fallback is None:
            return None
        self.G_path.append(fallback)
        return fallback

    def _has_unfinished_children(self, tree, G):
        for _, child in tree.out_edges(tuple(G)):
            if tree.edges[tuple(G), child]['sim_info'].get('unfinished', False):
                return True
        return False

    def _find_resumable_node(self, tree):
        nodes = self._find_resumable_nodes(tree, max_n=1)
        return nodes[0] if nodes else None

    def _find_resumable_nodes(self, tree, max_n):
        # Walk the tree for nodes that still have unexpanded or 'unfinished'
        # candidates.
        #
        # Sort order — cumulative path cost first, depth (smallest first)
        # second. Heuristic / preference / llm / comparison planners maintain
        # `self._cum_cost` (root=0, child=parent_cum + edge cost); cheaper
        # nodes are preferred backtrack targets so the frontier collapse at
        # 3-part subassemblies (2-part children are terminal and don't
        # propagate the frontier — see `feasible_children` skip in plan())
        # lands on the next-most-promising sibling instead of "whichever
        # bigger node happened to come first". Planners that don't populate
        # `_cum_cost` see infinity here and the secondary key (depth) does
        # all the work — same behaviour as before.
        #
        # Critical: only feasibly-reached nodes (n_gripper is not None) qualify.
        # Infeasible nodes are present in the tree because base.py:_update_tree
        # adds a node with n_gripper=None whenever a child edge comes back
        # infeasible. Expanding from such a node and getting a feasible child
        # back fires the `parent not feasible` assert in base._update_tree —
        # the DFA frontier collapse + this helper were the path resurrecting
        # those infeasible nodes as expansion targets.
        cum = getattr(self, '_cum_cost', None) or {}
        inf = float('inf')

        def _key(node):
            return (cum.get(tuple(node), inf), len(node))

        out = []
        for node in sorted(tree.nodes, key=_key):
            if len(node) <= 2:
                continue
            if tree.nodes[node].get('n_gripper') is None:
                continue
            if tree.out_degree(node) < len(node) or self._has_unfinished_children(tree, node):
                out.append(list(node))
                if len(out) >= max_n:
                    break
        return out

    def _select_next_frontier(self, tree, feasible_children, max_frontier):
        """Choose up to `max_frontier` child subassemblies for the next iteration.

        feasible_children: list of (G_prime, sim_info, parent_G) tuples in the
        order they came back from the parallel batch. Default takes the first
        `max_frontier` entries; subclasses can override to apply a quality score.
        """
        return [G_prime for G_prime, _, _ in feasible_children[:max_frontier]]

    def _print_candidate_summary(self, iter_idx, parents, received_by_parent,
                                 cand_dbg, next_frontier, tree=None):
        """Compact per-frontier debug overview. One small table per parent;
        rows are the parent's evaluated children (one per pose, since the
        same part with different poses runs as separate sim tasks).
        Columns: part, asm time, stab time, stability outcome, score.

        Score = the planner's per-edge cost computed live via _cost_child when
        the planner exposes that hook (HeuristicDFASequencePlanner and its
        subclasses — preference, comparison, llm). Falling back to the planner's
        `_cum_cost` table (which holds cumulative path cost when present), and
        finally to "—" for planners with no cost function. Per-edge cost is
        what's shown so each candidate row carries a meaningful number even
        when a subclass overrode _select_next_frontier without touching
        `_cum_cost` (the original failure mode this debug aid was missing).

        '★' marks children that survived into the next frontier.
        """
        def _fmt_t(dt):
            return f'{dt:.2f}s' if isinstance(dt, (int, float)) else '  —  '

        def _stab_outcome(sim_info):
            # 'parts_fix' is None when the stability sim never ran (assembly
            # failed first) OR when it ran and rejected the action. The two
            # are disambiguated by whether _dt_stab was captured.
            ran_stab = cand_dbg.get(id(sim_info), {}).get('dt_stab') is not None
            if not ran_stab:
                return 'skipped'
            if sim_info.get('parts_fix') is not None:
                return 'PASS'
            # Stability ran and rejected → fallen-parts list lives on fail_evidence.
            ev = sim_info.get('fail_evidence') or {}
            extra = ev.get('extra_ids')
            if isinstance(extra, (list, tuple)) and extra:
                tag = ','.join(map(str, extra[:3]))
                if len(extra) > 3:
                    tag += f',+{len(extra) - 3}'
                return f'FAIL[{tag}]'
            return 'FAIL'

        cum = getattr(self, '_cum_cost', None) or {}
        next_set = {tuple(g) for g in (next_frontier or [])}

        # Load weights once per iteration when the planner has a cost function.
        # `_cost_child` is defined on HeuristicDFASequencePlanner and inherited
        # by preference/comparison/llm; plain DFA / random don't have it.
        weights = None
        if hasattr(self, '_cost_child') and hasattr(self, '_load_weights'):
            try:
                weights = self._load_weights()
            except Exception:
                weights = None

        print(f'\n[DFA.frontier iter={iter_idx}] candidates ({len(parents)} parent(s)):')
        for i, G in enumerate(parents):
            rows = received_by_parent.get(i, [])
            label = f'{sorted(G)[:4]}{"..." if len(G) > 4 else ""}' if len(G) > 4 else f'{sorted(G)}'
            print(f'  parent[{i}] {label}  ({len(rows)} child eval(s))')
            print(f'    {"part":<20}  {"asm":>7}  {"stab":>7}  {"stability":<22}  {"score":>9}  next')
            print(f'    {"-"*20}  {"-"*7}  {"-"*7}  {"-"*22}  {"-"*9}  ----')

            # parent_pose drives the pose_change feature inside _cost_child.
            # Looked up via _parent_pose_for when the live tree was threaded
            # in by the caller; falls back to None for planners / tests that
            # don't pass tree. With parent_pose=None, the pose_change feature
            # is 0 (inert) — score reflects the other features only.
            parent_pose = self._parent_pose_for(tree, G) if tree is not None else None

            for sim_info, real_arg in rows:
                dbg = cand_dbg.get(id(sim_info), {})
                part = dbg.get('part', sim_info.get('part_move', '?'))
                dt_p = _fmt_t(dbg.get('dt_path'))
                dt_s = _fmt_t(dbg.get('dt_stab'))
                stab = _stab_outcome(sim_info)
                g_prime_key = tuple(real_arg[5])

                score = None
                # Primary: compute the per-edge cost live. Robust to subclasses
                # that override _select_next_frontier without writing _cum_cost.
                if weights is not None:
                    try:
                        score = float(self._cost_child(
                            weights, list(real_arg[5]), sim_info, list(G),
                            parent_pose=parent_pose,
                        ))
                    except Exception:
                        score = None
                # Secondary: cumulative cost from _cum_cost (heuristic populates this).
                if score is None:
                    score = cum.get(g_prime_key)

                score_str = f'{score:9.3f}' if isinstance(score, (int, float)) else '    —    '
                mark = ' ★' if g_prime_key in next_set else '  '
                print(f'    {str(part):<20}  {dt_p:>7}  {dt_s:>7}  {stab:<22}  {score_str}  {mark}')

    def _compute_poses(self, tree, G, max_poses, pose_reuse):
        if self.base_part is not None:
            return [None]
        poses = list(tree.nodes[tuple(G)]['poses'][:pose_reuse])
        G_mesh = get_combined_mesh(self.assembly_dir, G)
        _t0 = time()
        fresh = get_stable_poses(G_mesh, max_num=max_poses - pose_reuse)
        self._timing['stable_pose'] += time() - _t0
        self._timing_counts['stable_pose'] += 1
        # When multiple fresh stable poses are returned for this subassembly,
        # prefer the orientation closest to the parent step's pose so the
        # operator doesn't have to re-orient between consecutive steps unless
        # the geometry actually demands it. No effect at the root (no parent).
        parent_pose = self._parent_pose_for(tree, G)
        if parent_pose is not None and len(fresh) > 1:
            fresh = self._sort_poses_by_proximity(fresh, parent_pose)
        poses.extend(fresh)
        # Add the parent step's pose as an extra candidate (one more than
        # max_poses by design). The robot otherwise has to re-orient between
        # consecutive steps whenever the trimesh stable-pose list for this
        # subassembly doesn't happen to land on the parent's orientation;
        # explicitly probing the parent pose lets the planner keep the
        # assembly stationary when assemblability + stability still hold for
        # the child. Inserted at the front so per-edge sort orders (which
        # break ties by proximity to parent_pose) pick it first when feasible.
        # Deduped against existing candidates by a small angular tolerance.
        if parent_pose is not None:
            eps = 1e-3
            already_present = any(
                self._rotation_angle_between(parent_pose, p) < eps
                for p in poses if p is not None
            )
            if not already_present:
                poses.insert(0, parent_pose)
        if not poses:
            poses = [None]
        return poses

    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        # Used by the base serial plan() when num_proc == 1.
        super()._update_tree(tree, parts_parent, parts_child, n_eval, sim_info)
        if sim_info['feasible'] and len(parts_child) > 2:
            self.G_path.append(parts_child)
        else:
            self.G_path[-1] = parts_parent

    def plan(self, budget, max_grippers, max_poses=3, pose_reuse=0, early_term=False,
             timeout=None, plan_grasp=False, plan_arm=False, gripper_type=None,
             gripper_scale=None, optimizer='L-BFGS-B', debug=0, render=False, log_dir=None,
             n_success_term=1, connect_path=False, max_frontier=4):
        print(f'[DFA.plan] start planning with budget={budget}, num_proc={self.num_proc}, max_grippers={max_grippers}, max_poses={max_poses}, pose_reuse={pose_reuse}, early_term={early_term}, timeout={timeout}, plan_grasp={plan_grasp}, plan_arm={plan_arm}, gripper_type={gripper_type}, gripper_scale={gripper_scale}, optimizer={optimizer}, debug={debug}, render={render}, log_dir={log_dir}, n_success_term={n_success_term}, connect_path={connect_path}, max_frontier={max_frontier}, get_dof={self.get_dof}')
        if self.num_proc == 1:
            return super().plan(
                budget, max_grippers, max_poses, pose_reuse, early_term,
                timeout, plan_grasp, plan_arm, gripper_type, gripper_scale,
                optimizer, debug, render, log_dir, connect_path=connect_path,
            )

        if plan_grasp or plan_arm:
            # The parallel DFA path's _simulate_standalone worker has no
            # grasp_planner hook, so search-time grasp/arm IK filtering is
            # silently skipped. This is intentional: per-candidate IK +
            # collision checks would dominate the parallel batch cost. The
            # render pipeline (play_logged_plan._render_step_worker) instead
            # computes grasps lazily when show_arm=True, and the full
            # reach/disassembly/retreat trajectory is then planned in
            # render_grasp_arm via ArmMotionPlanner. Print once so users see
            # the flag wasn't a no-op overall.
            print('[DFA.plan] plan_grasp/plan_arm requested but not enforced '
                  'during parallel DFA search; grasps + arm trajectories are '
                  'computed at render time instead.')

        self.t_start = time()
        solution_found = False
        self.stop_msg = None
        self._timing = defaultdict(float)
        self._timing_counts = defaultdict(int)
        # Stash log_dir on self so overridable hooks (e.g. LLM frontier selection)
        # can write caches / decision logs alongside the planner output.
        self.log_dir = log_dir
        assert budget is not None or timeout is not None

        self._reset()
        self.n_eval = 0
        self._n_assembly_checks = 0
        self._n_assembly_success = 0
        self._n_stability_checks = 0
        self._n_stability_success = 0
        G0 = self.parts.copy()
        tree = nx.DiGraph()
        action = getattr(settings, 'no_stable_pose_action', 'exit')
        # Mirror SequencePlanner.plan()'s root-pose setup. The parallel DFA
        # branch (num_proc > 1) does NOT call super().plan(), so anything we
        # only put in the base class is invisible here — every change to
        # initial-pose handling must be replicated below or it silently
        # no-ops under preference / heuristic / llm / comparison runs.
        from plan_sequence.stable_pose import (
            get_combined_mesh as _get_combined_mesh,
            get_stable_poses as _get_stable_poses,
            translation_pose_to_ground as _translation_pose_to_ground,
        )
        from plan_sequence.planner._renders import render_unstable_parts as _render_unstable_parts
        from pathlib import Path as _Path

        G0_mesh = _get_combined_mesh(self.assembly_dir, G0)
        trimesh_candidates = _get_stable_poses(G0_mesh, max_num=max_poses)
        _per_pose = None
        if action == 'skip' and self.base_part is None:
            initial_poses = list(trimesh_candidates)
            if not initial_poses:
                initial_poses = [_translation_pose_to_ground(G0_mesh)]
                print("[DFA.plan] no trimesh stable pose for full assembly; "
                      "using translation-only ground-lift fallback "
                      "(settings.no_stable_pose_action='skip')")
            else:
                print(f"[DFA.plan] skipping initial stable-pose gravity check "
                      f"(settings.no_stable_pose_action='skip'); seeded "
                      f"{len(initial_poses)} trimesh stable pose(s) on root node")
            observed_fallen = frozenset()
        else:
            initial_poses, observed_fallen, _per_pose = self._initial_stable_poses(G0, max_poses, log_dir=log_dir)
            # Render one PNG per attempted pose (precheck_unstable_<i>.png)
            # in that pose's orientation with that pose's fallen parts
            # highlighted, instead of a single union image that hid the
            # per-pose breakdown. Gated by render_sequence; the textual ID
            # list still prints regardless.
            if not initial_poses and observed_fallen:
                print(f"[DFA.plan] precheck observed {len(observed_fallen)} "
                      f"part(s) falling: {sorted(observed_fallen)}")
                if getattr(settings, 'render_sequence', True):
                    _save_dir = _Path(log_dir) if log_dir else _Path('/tmp')
                    for _entry in _per_pose:
                        if _entry['success'] or not _entry['fallen']:
                            continue
                        _idx = _entry['pose_idx']
                        _png = _render_unstable_parts(
                            self.assembly_dir, G0, _entry['fallen'],
                            save_path=_save_dir / f'precheck_unstable_{_idx:02d}.png',
                            pose=_entry['pose'],
                        )
                        if _png is not None:
                            print(f"[DFA.plan] unstable-parts visualisation pose "
                                  f"{_idx}: {_png}  (fallen={_entry['fallen']})")
        # Interactive picker. Same fallback semantics as the serial branch:
        # if the verified set has <2 entries, augment with the unverified
        # trimesh candidates so the picker has something to show.
        initial_poses = self._pick_initial_pose_interactive(
            G0, initial_poses, log_dir=log_dir,
            fallback_candidates=trimesh_candidates,
            per_pose=_per_pose,
        )
        tree.add_node(tuple(G0), n_eval=0, n_gripper=1, poses=initial_poses)

        if self.base_part is None and not initial_poses and action != 'skip':
            if action == 'exit':
                self.stop_msg = 'no self-stable initial pose'
                print('[DFA.plan] aborting: no self-stable initial pose found '
                      "(settings.no_stable_pose_action='exit')")
                return tree
            elif action == 'ignore_unstable':
                self._ignored_unstable_parts = frozenset(observed_fallen)
                print(f"[DFA.plan] no self-stable initial pose; "
                      f"ignoring {sorted(self._ignored_unstable_parts)} in all future stability checks "
                      f"(settings.no_stable_pose_action='ignore_unstable')")
            else:  # 'continue' or anything unrecognised
                print(f"[DFA.plan] no self-stable initial pose; continuing with no seed "
                      f"(settings.no_stable_pose_action={action!r})")

        if debug > 0:
            contact_graph = get_contact_graph(self.asset_folder, self.assembly_dir, G0, contact_eps=CONTACT_EPS, save_sdf=self.save_sdf)
            isolated = [p for p in G0 if contact_graph.degree(p) == 0]
            print(f'[DFA.plan] contact graph: {contact_graph.number_of_nodes()} nodes, {contact_graph.number_of_edges()} edges')
            print(f'[DFA.plan] contact edges: {list(contact_graph.edges())}')
            if isolated:
                print(f'[DFA.plan] WARNING: {len(isolated)} isolated parts (no contact detected, stability check will fail immediately): {isolated}')
            else:
                print('[DFA.plan] all parts have at least one contact neighbour')

        # Multi-parent frontier: each iteration expands up to `max_frontier` parents
        # in one pooled batch, then derives the next frontier from feasible children.
        # max_frontier=1 collapses to single-parent DFS.
        self.frontier = [G0]
        self._iter_timings = []  # (iter_idx, n_parents_expanded, n_received, dt_seconds)
        iter_idx = 0

        try:
            while True:
                if early_term and solution_found:
                    self.stop_msg = 'solution found'
                    break
                if budget is not None and self.n_eval >= budget:
                    self.stop_msg = 'budget reached'
                    break
                if self._check_fully_explored(tree, G0):
                    self.stop_msg = 'tree fully explored'
                    break
                if timeout is not None and (time() - self.t_start) > timeout:
                    self.stop_msg = 'timeout'
                    break

                iter_idx += 1
                iter_start = time()

                # Keep only parents in the current frontier that still have work
                # (unexpanded candidates or 'unfinished' children to retry).
                live = []
                for G in self.frontier:
                    if tree.out_degree(tuple(G)) < len(G) or self._has_unfinished_children(tree, G):
                        live.append(G)
                        if len(live) >= max_frontier:
                            break

                # Frontier collapsed — backtrack via the tree to any resumable nodes.
                if not live:
                    live = self._find_resumable_nodes(tree, max_n=max_frontier)
                    if not live:
                        self.stop_msg = 'tree exhausted (no resumable node)'
                        break

                self.frontier = list(live)

                # Build the pooled task list across all live parents. Each task carries
                # parent_idx as its final positional arg; the wrapper strips it before
                # calling _simulate_standalone and re-attaches it to the sim_info dict
                # so the per-parent terminate callback can see it.
                per_parent_sim_tasks = {i: [] for i in range(len(live))}
                worker_args = []
                for i, G in enumerate(live):
                    parts_removed_G = [part for part in G0 if part not in G]
                    poses = self._compute_poses(tree, G, max_poses, pose_reuse)
                    if n_success_term is None:
                        candidate_parts = [p for p in G if p != self.base_part] if self.base_part is not None else list(G)
                    else:
                        candidate_parts = self.seq_generator.generate_candidate_part(G)
                    for p in candidate_parts:
                        G_prime = G.copy()
                        G_prime.remove(p)
                        if tree.has_edge(tuple(G), tuple(G_prime)):
                            prev = tree.edges[tuple(G), tuple(G_prime)]['sim_info']
                            if not prev.get('unfinished', False):
                                continue
                        for pose in poses:
                            per_parent_sim_tasks[i].append((p, G_prime.copy(), pose))

                    remaining_timeout = (None if timeout is None else timeout - (time() - self.t_start))
                    for p, G_prime, pose in per_parent_sim_tasks[i]:
                        worker_args.append((
                            self.asset_folder, self.assembly_dir, self.save_sdf, self.base_part,
                            p, G_prime, parts_removed_G, pose, max_grippers,
                            remaining_timeout, optimizer, max(debug - 2, 0), render, self.allow_gap, self.get_dof,
                            self.tools, self.skip_stability, self._ignored_unstable_parts,
                            i,  # parent_idx tag — stripped by _simulate_standalone_tagged
                        ))

                if not worker_args:
                    # Every live parent's candidates were already resolved; drop them
                    # so the next iteration triggers a fresh backtrack.
                    self.frontier = []
                    self._iter_timings.append((iter_idx, len(live), 0, time() - iter_start))
                    continue

                # Per-parent success quotas. Stop the whole pool only when every live
                # parent has hit its quota — that way we never starve a parent of its
                # validity info.
                per_parent_success = defaultdict(int)
                def _terminate(sim_info):
                    pi = sim_info.get('_parent_idx')
                    if pi is None:
                        return False
                    if sim_info['feasible']:
                        per_parent_success[pi] += 1
                    if n_success_term is None:
                        return False
                    return all(per_parent_success[j] >= n_success_term for j in range(len(live)))

                received = []  # (sim_info, real_arg_tuple, parent_idx)
                for sim_info, arg in parallel_execute(
                    _simulate_standalone_tagged, worker_args, self.num_proc,
                    show_progress=debug > 0, desc='DFA parallel sim', return_args=True,
                    terminate_func=_terminate,
                ):
                    parent_idx = arg[-1]
                    real_arg = arg[:-1]
                    sim_info.pop('_parent_idx', None)
                    received.append((sim_info, real_arg, parent_idx))

                if debug > 0:
                    early_stopped = len(received) < len(worker_args)
                    print(f'[DFA.plan] batch: parents={len(live)}  submitted={len(worker_args)}  '
                          f'received={len(received)}  successes={dict(per_parent_success)}  '
                          f'early_term={early_stopped}')

                self.n_eval += len(received)

                # Per-candidate debug table data — capture timings here before
                # the accumulation loop pops them from sim_info. id(sim_info) is
                # a stable key because each worker call returns a fresh dict.
                _cand_dbg = {}
                for sim_info, real_arg, parent_idx in received:
                    _cand_dbg[id(sim_info)] = {
                        'parent_idx': parent_idx,
                        'part': real_arg[4],
                        'dt_path': sim_info.get('_dt_path'),
                        'dt_stab': sim_info.get('_dt_stab'),
                    }

                # Accumulate per-check counts/timings (worker-side timings are stripped here).
                for sim_info, _, _ in received:
                    self._n_assembly_checks += 1
                    dt_path = sim_info.pop('_dt_path', None)
                    if dt_path is not None:
                        self._timing['path_finding'] += dt_path
                        self._timing_counts['path_finding'] += 1
                    dt_stab = sim_info.pop('_dt_stab', None)
                    dt_tool = sim_info.pop('_dt_tool', None)
                    if sim_info['action'] is not None:
                        self._n_assembly_success += 1
                        self._n_stability_checks += 1
                        if dt_stab is not None:
                            self._timing['stability_check'] += dt_stab
                            self._timing_counts['stability_check'] += 1
                        if sim_info['parts_fix'] is not None:
                            self._n_stability_success += 1
                    if dt_tool is not None:
                        self._timing['tool_check'] += dt_tool
                        self._timing_counts['tool_check'] += 1

                # Bucket results by parent_idx.
                received_by_parent = defaultdict(list)  # parent_idx -> [(sim_info, real_arg)]
                for sim_info, real_arg, parent_idx in received:
                    received_by_parent[parent_idx].append((sim_info, real_arg))

                # Per-parent post-processing: tree updates, DOF accumulation,
                # unfinished marking, and feasible-child collection for the next layer.
                #
                # Each _update_tree call needs a unique, monotonic n_eval tag because
                # two parents in the same batch can produce the same child
                # subassembly (e.g. [A,B,C] and [B,C,D] both yield [B,C]) — the
                # `_update_tree` assertion requires the tag to be strictly greater
                # than the existing node's tag. We start the counter just below the
                # post-increment self.n_eval so tags stay <= self.n_eval and remain
                # strictly greater than any tag used by prior batches.
                n_eval_tag = self.n_eval - len(received)

                # (G_prime, chosen_sim_info, parent_G) — passed to _select_next_frontier
                # so subclasses can score candidates using sim outcomes + parent context.
                feasible_children = []
                for i, G in enumerate(live):
                    parent_received = received_by_parent[i]
                    # Look up the pose used to reach this parent (None at root).
                    # Used below to break ties between feasible (part, pose)
                    # results for the same child by preferring the pose closest
                    # to the parent's pose — otherwise the pick was whichever
                    # worker happened to return first (race-dependent).
                    parent_pose = self._parent_pose_for(tree, G)

                    # Group this parent's results by G_prime (multiple poses per part).
                    results_by_edge = defaultdict(list)
                    for sim_info, real_arg in parent_received:
                        results_by_edge[tuple(real_arg[5])].append((sim_info, list(real_arg[5]), real_arg[4]))

                    for g_prime_key, edge_results in results_by_edge.items():
                        # Order results: feasible first, then by proximity of
                        # sim_info['pose'] to parent_pose (smaller rotation
                        # angle wins). Infeasible entries are sorted to the
                        # end so we still fall back to one of them when no
                        # pose was feasible. Stable sort; ties (same proximity
                        # or both None) preserve completion order, which is
                        # the prior behaviour.
                        edge_results.sort(key=lambda r: (
                            not r[0].get('feasible', False),
                            self._rotation_angle_between(parent_pose, r[0].get('pose')),
                        ))
                        chosen_sim_info, G_prime, p = edge_results[0]
                        n_eval_tag += 1
                        super()._update_tree(tree, G, G_prime, n_eval_tag, chosen_sim_info)

                        if chosen_sim_info['feasible']:
                            if len(G_prime) == 2:
                                solution_found = True
                            else:
                                feasible_children.append((G_prime, chosen_sim_info, G))

                        if debug > 2:
                            print(f'[DFA.plan] add edge ({G} → {G_prime}), feasible: {chosen_sim_info["feasible"]}')

                    if self.get_dof:
                        import numpy as np
                        node_dof = tree.nodes[tuple(G)].setdefault('dof_info', {})
                        for sim_info, real_arg in parent_received:
                            part = real_arg[4]
                            dof = sim_info.get('dof')
                            if dof is None:
                                continue
                            dof = np.asarray(dof, dtype=int)
                            prev = node_dof.get(part)
                            node_dof[part] = dof if prev is None else (np.asarray(prev, dtype=int) | dof)

                    # Mark this parent's submitted-but-not-received tasks as 'unfinished'
                    # (killed by the pool's terminate quota or by process termination).
                    received_g_primes = set(results_by_edge.keys())
                    marked = set()
                    for p, G_prime, pose in per_parent_sim_tasks[i]:
                        g_prime_key = tuple(G_prime)
                        if g_prime_key in received_g_primes or g_prime_key in marked:
                            continue
                        if tree.has_edge(tuple(G), g_prime_key):
                            if not tree.edges[tuple(G), g_prime_key]['sim_info'].get('unfinished', False):
                                continue
                        marked.add(g_prime_key)
                        if not tree.has_node(g_prime_key):
                            tree.add_node(g_prime_key, n_eval=self.n_eval, n_gripper=None, poses=[])
                        tree.add_edge(tuple(G), g_prime_key, n_eval=self.n_eval, sim_info={
                            'feasible': False, 'unfinished': True,
                            'part_move': p, 'pose': pose, 'action': None,
                            'base_part': self.base_part, 'parts_fix': None, 'grasp': None,
                        })

                # Next frontier: up to max_frontier feasible children from this batch.
                # If none were feasible, leave the frontier empty so the next iteration
                # backtracks via the tree scan. Subclasses may override
                # _select_next_frontier to rank candidates by a quality score.
                self.frontier = self._select_next_frontier(tree, feasible_children, max_frontier)

                # Per-frontier candidate summary: part id, assemblability check
                # time, stability check time, stability outcome, frontier score.
                # Scores come from the planner's _cum_cost map when present
                # (heuristic / preference / llm / comparison); plain DFA / random
                # show "—".
                self._print_candidate_summary(
                    iter_idx, live, received_by_parent, _cand_dbg,
                    next_frontier=self.frontier, tree=tree,
                )

                self._iter_timings.append((iter_idx, len(live), len(received), time() - iter_start))

                if debug > 0:
                    print(f'[DFA.plan] elapsed: {time() - self.t_start:.2f}s  evals: {self.n_eval}  '
                          f'next_frontier={len(self.frontier)}')
                    self.plot_tree(tree, save_path=log_dir + f'/dfa_tree_eval{self.n_eval}.png' if log_dir is not None else None)

                if log_dir is not None:
                    stats = self.get_stats(tree)
                    self.log(tree, stats, log_dir)

            self._expand_leaf(tree, max_poses, pose_reuse, None, optimizer, debug, render)
            self.plot_tree(tree)

        except (Exception, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                self.stop_msg = 'interrupt'
            else:
                self.stop_msg = 'exception'
            print(e, f'from {self.assembly_dir}')
            print(traceback.format_exc())

        assert self.stop_msg is not None, '[DFA.plan] bug: unexpectedly stopped'
        if debug > 0:
            print(f'[DFA.plan] stopped: {self.stop_msg}')
            self._print_timing_summary()
            n_assembly_fail = self._n_assembly_checks - self._n_assembly_success
            n_stability_fail = self._n_stability_checks - self._n_stability_success
            print(f'  assembly checks:   {self._n_assembly_checks:4d}  '
                  f'success: {self._n_assembly_success}  fail: {n_assembly_fail}')
            print(f'  stability checks:  {self._n_stability_checks:4d}  '
                  f'success: {self._n_stability_success}  fail: {n_stability_fail}')

        return tree
