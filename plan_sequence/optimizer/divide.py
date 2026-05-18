import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv

from .base import BaseSequenceOptimizer


DOF_ORDER = ['+Z', '-Z', '+X', '-X', '+Y', '-Y']
N_DOF = len(DOF_ORDER)

# Default weights for the cut score used by find_locally_free_subassemblies.
# score = BALANCE_WEIGHT * (min(|S|, |R|) / n_parts)
#       - CONTACT_WEIGHT * (cut_contacts(S, R) / max(n_edges_total, 1))
#       - FRAGMENTATION_WEIGHT * (max(0, comps(S) + comps(R) - 2) / max(n_parts - 2, 1))
# where comps(X) is the number of connected components in the contact-graph
# subgraph induced by X. All three normalised terms land in [0, ~1], so weight
# magnitudes are directly comparable. Fragmentation penalises cuts that split
# either side into multiple physically-disconnected sub-pieces.
BALANCE_WEIGHT = 1.0
CONTACT_WEIGHT = 0.7
FRAGMENTATION_WEIGHT = 1.0


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

    def build_obstruction_graph(self, apply_mirrors=True):
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

        if apply_mirrors:
            self.find_symmetric_additions()
            if self.symmetric_additions:
                for p, A in self.symmetric_additions.items():
                    graph.nodes[p]['obstruction'] |= A

        return graph

    def find_locally_free_subassemblies(self, timeout=100,
                                        balance_weight=BALANCE_WEIGHT,
                                        contact_weight=CONTACT_WEIGHT,
                                        fragmentation_weight=FRAGMENTATION_WEIGHT):
        '''
        DFS over partitions (S, R). Children of a state are single-part additions
        r ∈ S that are in contact (per the contact graph) with the current R, or
        any r ∈ S when R is empty. All children are pushed onto a stack ordered
        so the best child is popped first; ranking is (fewer-new-blocked-dofs
        ascending, cut-score descending). Visited partitions are memoised under
        a canonical key — frozenset({R, all_parts - R}) — so (R = A, S = B) and
        (R = B, S = A) collapse to the same state and the cut {A, B} is
        recorded only once.

        Constraint accumulation: re-evaluated from scratch at every state. For
        each p ∈ R, p contributes "S blocked in direction d^1" iff some
        per-part attribution M_p[i_q, d] == 1 with q still in S. Unattributed
        any-row entries are ignored (optimistic): if we know p is blocked in d
        but don't know which part is responsible, we don't propagate the
        constraint.

        Every recorded (non-fully-blocked) partition is scored:

            score = balance_weight * (min(|S|, |R|) / n_parts)
                  - contact_weight * (cut_contacts(S, R) / max(n_edges_total, 1))
                  - fragmentation_weight
                      * (max(0, comps(S) + comps(R) - 2) / max(n_parts - 2, 1))

        where cut_contacts is the number of contact-graph edges crossing the
        cut and comps(X) is the number of connected components in the contact
        subgraph induced by X. The fragmentation term is zero when both sides
        are each a single connected piece (the physical ideal) and grows as
        either side breaks into more components.

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
            attributions = M[:-1]
            contrib = np.zeros(N_DOF, dtype=bool)
            for d in range(N_DOF):
                known = [parts_list[i] for i in range(n_parts) if attributions[i, d]]
                if any(b in S for b in known):
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

        n_edges_total = graph.number_of_edges()
        edge_norm = max(n_edges_total, 1)
        frag_norm = max(n_parts - 2, 1)

        def induced_components(parts):
            if not parts:
                return 0
            return nx.number_connected_components(graph.subgraph(parts))

        def cut_score(S, R):
            balance = min(len(S), len(R)) / n_parts
            contacts = cut_contacts(S, R) / edge_norm
            comps = induced_components(S) + induced_components(R)
            fragmentation = max(0, comps - 2) / frag_norm
            return (balance_weight * balance
                    - contact_weight * contacts
                    - fragmentation_weight * fragmentation)

        S0 = frozenset(parts_idx)
        R0 = frozenset()
        blocked0 = state_blocked(S0, R0)

        def canonical(R):
            return frozenset((R, S0 - R))

        stack = [(S0, R0, blocked0)]
        visited = set()
        results = []

        t_start = time.time()
        timed_out = False
        while stack:
            if time.time() - t_start > timeout:
                timed_out = True
                break

            S, R, blocked = stack.pop()
            key = canonical(R)
            if key in visited:
                continue
            visited.add(key)

            if blocked.all():
                continue
            if not S and R:
                # complement of the (all, ∅) start state; same partition.
                continue

            results.append((S, R, cut_score(S, R)))

            if not S:
                continue

            if R:
                candidates = [r for r in S if any(graph.has_edge(r, q) for q in R)]
            else:
                candidates = list(S)
            if not candidates:
                continue

            children = []
            for r in candidates:
                cand_S = S - {r}
                cand_R = R | {r}
                if canonical(cand_R) in visited:
                    continue
                cand_blocked = state_blocked(cand_S, cand_R)
                new_count = int(cand_blocked.sum())
                children.append((cand_S, cand_R, cand_blocked, new_count, cut_score(cand_S, cand_R)))

            # Rank: fewer-new-blocked asc, then cut-score desc. Push worst-first
            # so the best child is on top of the stack and popped first.
            children.sort(key=lambda c: (c[3], -c[4]))
            for cand_S, cand_R, cand_blocked, _, _ in reversed(children):
                stack.append((cand_S, cand_R, cand_blocked))

        elapsed = time.time() - t_start
        suffix = ' (TIMED OUT)' if timed_out else ''
        results.sort(key=lambda t: t[2], reverse=True)
        print(f'[DivideOptimizer] find_locally_free_subassemblies: '
              f'{len(results)} free partitions via DFS over {len(visited)} states, '
              f'elapsed {elapsed:.2f}s{suffix}')

        def _fmt(idx, entry):
            S_, R_, s_ = entry
            return (f'  [{idx:>3}] score={s_:.4f}  '
                    f'|S|={len(S_)}  |R|={len(R_)}  '
                    f'S={sorted(S_)}  R={sorted(R_)}')

        if results:
            top_n = min(10, len(results))
            print(f'top {top_n} by score:')
            for i in range(top_n):
                print(_fmt(i, results[i]))
            if len(results) > 10:
                bot_n = min(10, len(results) - top_n)
                print(f'bottom {bot_n} by score:')
                for i in range(len(results) - bot_n, len(results)):
                    print(_fmt(i, results[i]))

        self.locally_free = results
        return results

    def verify_locally_free(self, top_k=10, num_proc=1, save_sdf=False, max_time=30, force_mag=None):
        '''
        Physically verify the top-k locally-free partitions by combining each
        side's parts into a single rigid mesh and running a path-planning
        simulation. Top-k entries are taken in the order of self.locally_free
        (already score-sorted descending). Verification happens in parallel
        across partitions.

        Stores the verified-only filtered list on self.verified_locally_free.
        self.locally_free is left untouched.

        Returns: list of (S, R, score) tuples that passed verification.
        '''
        if not self.locally_free:
            print('[DivideOptimizer] no partitions to verify; call find_locally_free_subassemblies() first.')
            return None
        if self.asset_folder is None or self.assembly_dir is None:
            print('[DivideOptimizer] asset_folder/assembly_dir not set; cannot run physics verification.')
            return None

        from plan_sequence.physics_planner import FORCE_MAG, _verify_standalone
        from utils.parallel import parallel_execute

        if force_mag is None:
            force_mag = FORCE_MAG

        to_verify = self.locally_free[:top_k] if top_k is not None else list(self.locally_free)
        worker_args = [
            (self.asset_folder, self.assembly_dir,
             sorted(S), sorted(R), save_sdf, max_time, force_mag)
            for (S, R, _) in to_verify
        ]

        results_by_key = {}
        for result, arg in parallel_execute(
            _verify_standalone, worker_args, num_proc,
            show_progress=True, desc='verify locally-free',
            return_args=True, raise_exception=False,
        ):
            key = (tuple(arg[2]), tuple(arg[3]))
            results_by_key[key] = result['verified']

        verified = []
        for S, R, score in to_verify:
            key = (tuple(sorted(S)), tuple(sorted(R)))
            if results_by_key.get(key, False):
                verified.append((S, R, score))

        self.verified_locally_free = verified
        print(f'[DivideOptimizer] verify_locally_free: '
              f'{len(verified)}/{len(to_verify)} partitions verified separable')
        for i, (S, R, s) in enumerate(verified):
            print(f'  [{i:>3}] score={s:.4f}  |S|={len(S)}  |R|={len(R)}  '
                  f'S={sorted(S)}  R={sorted(R)}')
        return verified

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
                    if A[r, c]:
                        ax.text(c, r, '+', ha='center', va='center', color='white', fontsize=7)
                    elif M[r, c]:
                        ax.text(c, r, '1', ha='center', va='center', color='white', fontsize=7)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=existing_rgb, label='directly observed'),
            plt.Rectangle((0, 0), 1, 1, color=proposed_rgb, label='mirror-inferred'),
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

    def visualize_subassemblies(self, meshes, output_dir, top_n=10, bottom_n=10,
                                 S_color='crimson', R_color='steelblue', opacity=0.85):
        '''
        Render the top-`top_n` and bottom-`bottom_n` scored partitions from
        self.locally_free as PNG screenshots. `meshes` must be a dict
        {part_id: pyvista.PolyData}. Files are written to `output_dir`.
        '''
        if not self.locally_free:
            print('[DivideOptimizer] no locally_free partitions to render. '
                  'Call find_locally_free_subassemblies() first.')
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_total = len(self.locally_free)
        top_n = min(top_n, n_total)
        bottom_n = min(bottom_n, max(0, n_total - top_n))

        selected = []
        for rank in range(top_n):
            selected.append(('top', rank, self.locally_free[rank]))
        for rank in range(n_total - bottom_n, n_total):
            selected.append(('bot', rank, self.locally_free[rank]))

        for tag, rank, (S, R, score) in selected:
            plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
            for p in S:
                if p in meshes:
                    plotter.add_mesh(meshes[p], color=S_color, opacity=opacity, show_edges=False)
            for p in R:
                if p in meshes:
                    plotter.add_mesh(meshes[p], color=R_color, opacity=opacity, show_edges=False)
            plotter.add_text(
                f'rank {rank}  score={score:.4f}\n|S|={len(S)} (red)  |R|={len(R)} (blue)',
                font_size=10, position='upper_left',
            )
            plotter.camera_position = 'iso'
            fname = f'{tag}_{rank:03d}_score{score:.4f}.png'
            plotter.screenshot(str(out / fname))
            plotter.close()

        print(f'[DivideOptimizer] visualized {len(selected)} subassemblies to {out}')

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
