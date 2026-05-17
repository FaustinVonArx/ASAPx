import traceback
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import networkx as nx

from .base import SequencePlanner, _simulate_standalone
from plan_sequence.physics_planner import get_contact_graph, CONTACT_EPS
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses
from utils.parallel import parallel_execute


class DFASequencePlanner(SequencePlanner):

    G_path = None

    @staticmethod
    def plot_tree(tree, save_path=None):
        import tempfile
        from networkx.drawing.nx_agraph import graphviz_layout

        def _node_color(node):
            if tree.nodes[node]['n_gripper'] is not None:
                return 'g'
            incoming = list(tree.in_edges(node))
            if incoming and all(tree.edges[e]['sim_info'].get('unfinished', False) for e in incoming):
                return 'gold'
            return 'r'

        def _edge_color(edge):
            info = tree.edges[edge]['sim_info']
            if info.get('unfinished', False):
                return 'gold'
            return 'g' if info['feasible'] else 'r'

        node_colors = [_node_color(node) for node in tree.nodes]
        edge_colors = [_edge_color(edge) for edge in tree.edges]
        pos = graphviz_layout(tree, prog='dot')
        fig, ax = plt.subplots(figsize=(16, 10))
        nx.draw(tree, pos, ax=ax, node_color=node_colors, edge_color=edge_colors, with_labels=False, node_size=40, arrowsize=6)
        out = save_path or tempfile.mktemp(suffix='.png', prefix='dfa_tree_')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        if save_path is None:
            print(f'[DFA] tree saved to {out}')

    def _reset(self):
        G0 = self.parts.copy()
        self.G_path = [G0]

    def _select_node(self, tree):
        G = self.G_path[-1]
        if tree.out_degree(tuple(G)) < len(G):
            return G
        else:
            self.G_path.pop()
            return self._select_node(tree)

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
             n_success_term=1, connect_path=False):
        print(f'[DFA.plan] start planning with budget={budget}, num_proc={self.num_proc}, max_grippers={max_grippers}, max_poses={max_poses}, pose_reuse={pose_reuse}, early_term={early_term}, timeout={timeout}, plan_grasp={plan_grasp}, plan_arm={plan_arm}, gripper_type={gripper_type}, gripper_scale={gripper_scale}, optimizer={optimizer}, debug={debug}, render={render}, log_dir={log_dir}, n_success_term={n_success_term}, connect_path={connect_path}, get_dof={self.get_dof}')
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

                G = self._select_node(tree)
                parts_removed = [part for part in G0 if part not in G]

                # Compute stable poses for the current subassembly.
                if self.base_part is not None:
                    poses = [None]
                else:
                    poses = tree.nodes[tuple(G)]['poses'][:pose_reuse]
                    G_mesh = get_combined_mesh(self.assembly_dir, G)
                    _t0 = time()
                    poses.extend(get_stable_poses(G_mesh, max_num=max_poses - pose_reuse))
                    self._timing['stable_pose'] += time() - _t0
                    self._timing_counts['stable_pose'] += 1
                    if len(poses) == 0:
                        poses = [None]

                # Collect every unexplored (part, pose) pair for this node in one batch.
                # Edges marked 'unfinished' (cut short by a previous early-termination batch)
                # are eligible for retry.
                sim_tasks = []  # (part, G_prime, pose)
                for p in self.seq_generator.generate_candidate_part(G):
                    G_prime = G.copy()
                    G_prime.remove(p)
                    if tree.has_edge(tuple(G), tuple(G_prime)):
                        prev = tree.edges[tuple(G), tuple(G_prime)]['sim_info']
                        if not prev.get('unfinished', False):
                            continue
                    for pose in poses:
                        sim_tasks.append((p, G_prime.copy(), pose))

                if not sim_tasks:
                    # All candidates already explored; _select_node will backtrack next iteration.
                    continue

                # arg layout: [0]=asset_folder [1]=assembly_dir [2]=save_sdf [3]=base_part
                #              [4]=part_move   [5]=G_prime      [6]=parts_removed [7]=pose
                #              [8]=max_grippers [9]=timeout [10]=optimizer [11]=debug [12]=render
                #              [13]=allow_gap
                remaining_timeout = (None if timeout is None else timeout - (time() - self.t_start))
                worker_args = [
                    (self.asset_folder, self.assembly_dir, self.save_sdf, self.base_part,
                     p, G_prime, parts_removed, pose, max_grippers,
                     remaining_timeout, optimizer, max(debug - 2, 0), render, self.allow_gap, self.get_dof)
                    for p, G_prime, pose in sim_tasks
                ]

                # Run all tasks in parallel; terminate once n_success_term successes arrive.
                # Unevaluated candidates (due to early termination) have no tree edge and
                # will be retried if we backtrack to this node later.
                n_success = [0]
                def _terminate(sim_info):
                    if sim_info['feasible']:
                        n_success[0] += 1
                    return n_success_term is not None and n_success[0] >= n_success_term

                received = []
                for sim_info, arg in parallel_execute(
                    _simulate_standalone, worker_args, self.num_proc,
                    show_progress=debug > 0, desc='DFA parallel sim', return_args=True,
                    terminate_func=_terminate,
                ):
                    received.append((sim_info, arg))

                if debug > 0:
                    early_stopped = len(received) < len(worker_args)
                    print(f'[DFA.plan] batch: submitted={len(worker_args)}  received={len(received)}  '
                          f'successes={n_success[0]}/{n_success_term}  early_term={early_stopped}')

                # Count only tasks that actually ran (early termination may have skipped some).
                self.n_eval += len(received)

                # Accumulate per-check success/failure counts.
                for sim_info, _ in received:
                    self._n_assembly_checks += 1
                    if sim_info['action'] is not None:
                        self._n_assembly_success += 1
                        self._n_stability_checks += 1
                        if sim_info['feasible']:
                            self._n_stability_success += 1

                # Group by G_prime; for each part keep only the first feasible pose result
                # (or any infeasible result if no pose worked).
                results_by_edge = defaultdict(list)
                for sim_info, arg in received:
                    results_by_edge[tuple(arg[5])].append((sim_info, list(arg[5]), arg[4]))

                best_feasible_child = None
                for g_prime_key, edge_results in results_by_edge.items():
                    chosen_sim_info, G_prime, p = next(
                        (r for r in edge_results if r[0]['feasible']), edge_results[0]
                    )
                    super()._update_tree(tree, G, G_prime, self.n_eval, chosen_sim_info)

                    if chosen_sim_info['feasible']:
                        if len(G_prime) == 2:
                            solution_found = True
                        elif best_feasible_child is None:
                            best_feasible_child = G_prime

                    if debug > 0:
                        print(f'[DFA.plan] add edge ({G} → {G_prime}), feasible: {chosen_sim_info["feasible"]}')

                if self.get_dof:
                    node_dof = tree.nodes[tuple(G)].setdefault('dof_info', {})
                    for sim_info, arg in received:
                        part = arg[4]
                        if sim_info.get('dof') is not None and part not in node_dof:
                            node_dof[part] = sim_info['dof']

                # Mark candidates that were submitted but never returned (killed by early
                # termination) so they show up yellow in the tree and can be retried later.
                received_g_primes = set(results_by_edge.keys())
                marked = set()
                for p, G_prime, pose in sim_tasks:
                    g_prime_key = tuple(G_prime)
                    if g_prime_key in received_g_primes or g_prime_key in marked:
                        continue
                    if tree.has_edge(tuple(G), g_prime_key):
                        # already has a real result from an earlier batch — leave it alone
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

                if debug > 1:
                    print(f'[DFA.plan] elapsed: {time() - self.t_start:.1f}s  evals: {self.n_eval}')
                    self.plot_tree(tree, save_path=log_dir + f'/dfa_tree_eval{self.n_eval}.png' if log_dir is not None else None)

                # Go deeper into the first feasible child; otherwise _select_node will backtrack.
                if best_feasible_child is not None:
                    self.G_path.append(best_feasible_child)

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
