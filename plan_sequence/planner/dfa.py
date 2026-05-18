import traceback
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import networkx as nx

from .base import SequencePlanner, _simulate_standalone_tagged
from plan_sequence.physics_planner import get_contact_graph, CONTACT_EPS
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses
from utils.parallel import parallel_execute


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
        import tempfile
        from collections import Counter
        from matplotlib.patches import Patch
        from networkx.drawing.nx_agraph import graphviz_layout

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
        pos = graphviz_layout(tree, prog='dot')
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
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        if save_path is None:
            print(f'[DFA] tree saved to {out}')

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
        # candidates. Smaller subassemblies come first (deeper in the tree),
        # so backtracking prefers nodes closer to the most recent frontier.
        out = []
        for node in sorted(tree.nodes, key=len):
            if len(node) <= 2:
                continue
            if tree.out_degree(node) < len(node) or self._has_unfinished_children(tree, node):
                out.append(list(node))
                if len(out) >= max_n:
                    break
        return out

    def _compute_poses(self, tree, G, max_poses, pose_reuse):
        if self.base_part is not None:
            return [None]
        poses = list(tree.nodes[tuple(G)]['poses'][:pose_reuse])
        G_mesh = get_combined_mesh(self.assembly_dir, G)
        _t0 = time()
        poses.extend(get_stable_poses(G_mesh, max_num=max_poses - pose_reuse))
        self._timing['stable_pose'] += time() - _t0
        self._timing_counts['stable_pose'] += 1
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
            raise NotImplementedError('Grasp/arm planning is not supported in parallel DFA mode')

        self.t_start = time()
        solution_found = False
        self.stop_msg = None
        self._timing = defaultdict(float)
        self._timing_counts = defaultdict(int)
        assert budget is not None or timeout is not None

        self._reset()
        self.n_eval = 0
        self._n_assembly_checks = 0
        self._n_assembly_success = 0
        self._n_stability_checks = 0
        self._n_stability_success = 0
        G0 = self.parts.copy()
        tree = nx.DiGraph()
        #NOTE call inital stable pose here 
        tree.add_node(tuple(G0), n_eval=0, n_gripper=1, poses=[])

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
                            self.tools, self.skip_stability,
                            i,  # parent_idx tag — stripped by _simulate_standalone_tagged
                        ))

                if not worker_args:
                    # Every live parent's candidates were already resolved; drop them
                    # so the next iteration triggers a fresh backtrack.
                    self.frontier = []
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

                feasible_children = []  # list of (G_prime, parent_idx) in arrival order
                for i, G in enumerate(live):
                    parent_received = received_by_parent[i]

                    # Group this parent's results by G_prime (multiple poses per part).
                    results_by_edge = defaultdict(list)
                    for sim_info, real_arg in parent_received:
                        results_by_edge[tuple(real_arg[5])].append((sim_info, list(real_arg[5]), real_arg[4]))

                    for g_prime_key, edge_results in results_by_edge.items():
                        chosen_sim_info, G_prime, p = next(
                            (r for r in edge_results if r[0]['feasible']), edge_results[0]
                        )
                        n_eval_tag += 1
                        super()._update_tree(tree, G, G_prime, n_eval_tag, chosen_sim_info)

                        if chosen_sim_info['feasible']:
                            if len(G_prime) == 2:
                                solution_found = True
                            else:
                                feasible_children.append(G_prime)

                        if debug > 0:
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
                # backtracks via the tree scan.
                self.frontier = feasible_children[:max_frontier]

                if debug > 1:
                    print(f'[DFA.plan] elapsed: {time() - self.t_start:.1f}s  evals: {self.n_eval}  '
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
