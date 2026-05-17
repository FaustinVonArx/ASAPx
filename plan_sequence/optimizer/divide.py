import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .base import BaseSequenceOptimizer


DOF_ORDER = ['+Z', '-Z', '+X', '-X', '+Y', '-Y']
N_DOF = len(DOF_ORDER)


class DivideOptimizer(BaseSequenceOptimizer):
    '''
    Optimizer that builds a per-part obstruction graph from a planning tree.

    Expects the tree to carry per-node DoF info (planner run with get_dof=True),
    where tree.nodes[G]['dof_info'][p] is the length-6 translation DoF vector of
    part p probed in sub-assembly G. Vector order is world-frame [+Z, -Z, +X, -X, +Y, -Y]
    (see MultiPartPathPlanner.compute_dof). 0 = blocked, 1 = free.

    For every feasible edge G_parent --remove r--> G and every part p still in G,
    if p had dof_parent[d] == 0 and dof_child[d] == 1 then r is recorded as
    blocking p in direction d. Any dof_parent[d] == 0 also marks p's "any-blocker"
    row for direction d, so directions known to be blocked are surfaced even
    when the specific blocker is not yet attributable.

    Per-part matrix on graph.nodes[p]['obstruction']:
        - shape (n_parts + 1, 6), dtype int
        - rows 0..n_parts-1: blocker attribution; M[i, d] == 1 means parts[i]
          blocks p in direction d.
        - row n_parts: aggregate "any-blocker"; M[-1, d] == 1 means p is blocked
          in direction d at some point in the tree.

    Part ordering is exposed on graph.graph['parts_idx'] (part_id -> row index)
    and graph.graph['dof_order'].
    '''

    def __init__(self, tree, asset_folder=None, assembly_dir=None):
        super().__init__(tree)
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.obstruction_graph = None
        self.locally_free = None
        self.symmetric_additions = None

    def build_obstruction_graph(self):
        if not any(self.tree.nodes[n].get('dof_info') for n in self.tree.nodes):
            print('[DivideOptimizer] tree has no dof_info; re-run planner with get_dof=True. Aborting.')
            return None

        all_parts = list(self.root)
        n_parts = len(all_parts)
        parts_idx = {p: i for i, p in enumerate(all_parts)}
        any_row = n_parts

        if self.asset_folder is not None and self.assembly_dir is not None:
            from plan_sequence.physics_planner import get_contact_graph
            graph = get_contact_graph(self.asset_folder, self.assembly_dir, parts=all_parts)
        else:
            graph = nx.Graph()
            for p in all_parts:
                graph.add_node(p)

        for p in all_parts:
            graph.nodes[p]['obstruction'] = np.zeros((n_parts + 1, N_DOF), dtype=int)
        graph.graph['parts_idx'] = parts_idx
        graph.graph['dof_order'] = list(DOF_ORDER)

        # Traverse only the feasibly-reachable sub-tree.
        visited = {self.root}
        stack = [self.root]
        while stack:
            u = stack.pop()
            dof_u = self.tree.nodes[u].get('dof_info') or {}

            for p, dof in dof_u.items():
                if p not in parts_idx:
                    continue
                M = graph.nodes[p]['obstruction']
                for d in range(N_DOF):
                    if int(dof[d]) == 0:
                        M[any_row, d] = 1

            for _, v, edata in self.tree.out_edges(u, data=True):
                sim_info = edata.get('sim_info') or {}
                if not sim_info.get('feasible'):
                    continue
                removed = sim_info.get('part_move')
                dof_v = self.tree.nodes[v].get('dof_info') or {}
                if removed in parts_idx:
                    for p in v:
                        dof_parent = dof_u.get(p)
                        dof_child = dof_v.get(p)
                        if dof_parent is None or dof_child is None:
                            continue
                        M = graph.nodes[p]['obstruction']
                        for d in range(N_DOF):
                            if int(dof_parent[d]) == 0 and int(dof_child[d]) == 1:
                                M[parts_idx[removed], d] = 1
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        self.obstruction_graph = graph
        return graph

    def find_locally_free_subassemblies(self, timeout=100):
        '''
        Greedy walk over partitions (S, R), starting at S = all parts, R = empty.
        At each step, move the part r ∈ S to R that minimises the total number
        of blocked directions on the resulting S. The walk stops when S is
        empty or the new state is fully blocked.

        Constraint accumulation: re-evaluated from scratch at every state. For
        each p ∈ R, p contributes "S blocked in direction d^1" iff p was
        observed blocked in d (any-row) AND either a per-part attribution
        points to some part still in S, or no per-part attribution exists
        (conservative case 4a).

        Every recorded (non-fully-blocked) partition is scored:

            score = (min(|S|, |R|) / N) / (1 + contacts(S, R))

        where contacts is the number of edges in the contact graph crossing
        the (S, R) cut. Higher = more balanced cut with fewer contact bridges.

        Returns: list of (S, R, score) tuples sorted by score descending. Also
        stored on self.locally_free. Times out after `timeout` seconds.
        '''
        if self.obstruction_graph is None:
            print('[DivideOptimizer] obstruction graph not built; call build_obstruction_graph() first.')
            return None

        graph = self.obstruction_graph
        parts_idx = graph.graph['parts_idx']
        n_parts = len(parts_idx)
        parts_list = [None] * n_parts
        for p, i in parts_idx.items():
            parts_list[i] = p

        def part_blocks_S(part_id, S):
            M = graph.nodes[part_id]['obstruction']
            any_row = M[-1]
            attributions = M[:-1]
            contrib = np.zeros(N_DOF, dtype=bool)
            for d in range(N_DOF):
                if not any_row[d]:
                    continue
                known = [parts_list[i] for i in range(n_parts) if attributions[i, d]]
                if not known:
                    contrib[d ^ 1] = True
                elif any(b in S for b in known):
                    contrib[d ^ 1] = True
            return contrib

        def state_blocked(S, R):
            blocked = np.zeros(N_DOF, dtype=bool)
            for p in R:
                blocked |= part_blocks_S(p, S)
            return blocked

        def cut_contacts(S, R):
            n = 0
            for u, v in graph.edges():
                if (u in S and v in R) or (u in R and v in S):
                    n += 1
            return n

        def cut_score(S, R):
            return (min(len(S), len(R)) / n_parts) / (1 + cut_contacts(S, R))

        S = frozenset(parts_idx)
        R = frozenset()
        blocked = state_blocked(S, R)
        results = []

        t_start = time.time()
        timed_out = False
        while True:
            if time.time() - t_start > timeout:
                timed_out = True
                break
            if blocked.all():
                break
            if not S and R:
                # symmetric to the (all, ∅) start state already recorded
                break

            results.append((S, R, cut_score(S, R)))

            if not S:
                break

            best_r = None
            best_new_blocked = None
            best_new_count = None
            for r in S:
                cand_S = S - {r}
                cand_R = R | {r}
                cand_blocked = state_blocked(cand_S, cand_R)
                new_count = int(cand_blocked.sum())
                if best_new_count is None or new_count < best_new_count:
                    best_r = r
                    best_new_blocked = cand_blocked
                    best_new_count = new_count

            S = S - {best_r}
            R = R | {best_r}
            blocked = best_new_blocked

        elapsed = time.time() - t_start
        suffix = ' (TIMED OUT)' if timed_out else ''
        results.sort(key=lambda t: t[2], reverse=True)
        print(f'[DivideOptimizer] find_locally_free_subassemblies: '
              f'{len(results)} free partitions via greedy walk, elapsed {elapsed:.2f}s{suffix}')
        if results:
            S_top, R_top, s_top = results[0]
            print(f'  top: |S|={len(S_top)} |R|={len(R_top)} score={s_top:.4f}')

        self.locally_free = results
        return results

    def find_symmetric_additions(self):
        '''
        For every existing per-part entry M_p[i_q, d] == 1 (q blocks p in d),
        check if the physically required mirror entry M_q[i_p, d^1] (p blocks q
        in the opposite direction) is missing. Record the missing entries (and
        their any-blocker counterparts) as proposed additions, but do NOT
        modify the obstruction matrices.

        Returns: dict {part_id: int matrix (n_parts + 1, 6)} where a 1 marks a
        proposed addition. Also stored on self.symmetric_additions.
        '''
        if self.obstruction_graph is None:
            print('[DivideOptimizer] obstruction graph not built; call build_obstruction_graph() first.')
            return None

        graph = self.obstruction_graph
        parts_idx = graph.graph['parts_idx']
        n_parts = len(parts_idx)
        parts_list = [None] * n_parts
        for p, i in parts_idx.items():
            parts_list[i] = p

        additions = {p: np.zeros((n_parts + 1, N_DOF), dtype=int) for p in parts_list}

        n_existing = 0
        for p in parts_list:
            M_p = graph.nodes[p]['obstruction']
            i_p = parts_idx[p]
            for q in parts_list:
                if q == p:
                    continue
                i_q = parts_idx[q]
                M_q = graph.nodes[q]['obstruction']
                for d in range(N_DOF):
                    if M_p[i_q, d] != 1:
                        continue
                    n_existing += 1
                    d_opp = d ^ 1
                    if M_q[i_p, d_opp] == 0:
                        additions[q][i_p, d_opp] = 1
                    if M_q[-1, d_opp] == 0:
                        additions[q][-1, d_opp] = 1

        n_added = sum(int(A.sum()) for A in additions.values())
        print(f'[DivideOptimizer] find_symmetric_additions: '
              f'{n_existing} existing per-part entries, {n_added} proposed entries '
              f'({n_added} would be added if mirrors are applied).')

        self.symmetric_additions = additions
        return additions

    def visualize_symmetric_additions(self, save_path=None, show=True):
        '''
        Same layout as visualize_obstruction_graph, but proposed mirror entries
        (from find_symmetric_additions) are overlaid in blue while existing
        entries stay red.
        '''
        if self.obstruction_graph is None:
            print('[DivideOptimizer] obstruction graph not built.')
            return None
        if self.symmetric_additions is None:
            self.find_symmetric_additions()

        graph = self.obstruction_graph
        additions = self.symmetric_additions
        parts_idx = graph.graph['parts_idx']
        dof_order = graph.graph['dof_order']
        parts = list(parts_idx.keys())
        n_parts = len(parts)
        row_labels = [str(p) for p in parts] + ['any']

        n_panels = n_parts + 1
        n_cols = min(4, n_panels)
        n_rows = math.ceil(n_panels / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        ax0 = axes[0]
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=0)
        nx.draw(
            graph, pos, ax=ax0, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=420, font_size=8,
        )
        ax0.set_title('Contact graph')
        ax0.set_axis_off()

        existing_rgb = np.array([0.85, 0.20, 0.20])
        proposed_rgb = np.array([0.20, 0.45, 0.90])

        for i, p in enumerate(parts):
            ax = axes[i + 1]
            M = graph.nodes[p]['obstruction']
            A = additions[p]

            rgb = np.ones((M.shape[0], M.shape[1], 3), dtype=float)
            rgb[M > 0] = existing_rgb
            rgb[A > 0] = proposed_rgb

            ax.imshow(rgb, aspect='auto')
            ax.set_title(f'Part {p}')
            ax.set_xticks(range(len(dof_order)))
            ax.set_xticklabels(dof_order, fontsize=7)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.axhline(n_parts - 0.5, color='black', linewidth=1)

            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    if M[r, c]:
                        ax.text(c, r, '1', ha='center', va='center', color='white', fontsize=7)
                    elif A[r, c]:
                        ax.text(c, r, '+', ha='center', va='center', color='white', fontsize=7)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=existing_rgb, label='existing'),
            plt.Rectangle((0, 0), 1, 1, color=proposed_rgb, label='proposed (mirror)'),
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=False)

        for j in range(n_panels, len(axes)):
            axes[j].set_axis_off()

        fig.tight_layout(rect=(0, 0.03, 1, 1))
        if save_path is not None:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f'[DivideOptimizer] symmetric-additions graph saved to {save_path}')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def visualize_obstruction_graph(self, save_path=None, show=True):
        '''
        Render the obstruction graph: the contact graph in the first panel, and
        one heatmap per part for its (n_parts + 1, 6) obstruction matrix.
        '''
        if self.obstruction_graph is None:
            print('[DivideOptimizer] obstruction graph not built; call build_obstruction_graph() first.')
            return None

        graph = self.obstruction_graph
        parts_idx = graph.graph['parts_idx']
        dof_order = graph.graph['dof_order']
        parts = list(parts_idx.keys())
        n_parts = len(parts)
        row_labels = [str(p) for p in parts] + ['any']

        n_panels = n_parts + 1
        n_cols = min(4, n_panels)
        n_rows = math.ceil(n_panels / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        ax0 = axes[0]
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=0)
        nx.draw(
            graph, pos, ax=ax0, with_labels=True,
            node_color='lightblue', edge_color='gray',
            node_size=420, font_size=8,
        )
        ax0.set_title('Contact graph')
        ax0.set_axis_off()

        for i, p in enumerate(parts):
            ax = axes[i + 1]
            M = graph.nodes[p]['obstruction']
            ax.imshow(M, cmap='Reds', vmin=0, vmax=1, aspect='auto')
            ax.set_title(f'Part {p}')
            ax.set_xticks(range(len(dof_order)))
            ax.set_xticklabels(dof_order, fontsize=7)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.axhline(n_parts - 0.5, color='black', linewidth=1)
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    if M[r, c]:
                        ax.text(c, r, '1', ha='center', va='center', color='white', fontsize=7)

        for j in range(n_panels, len(axes)):
            axes[j].set_axis_off()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f'[DivideOptimizer] obstruction graph saved to {save_path}')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
