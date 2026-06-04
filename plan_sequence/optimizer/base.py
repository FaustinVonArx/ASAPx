import random

import networkx as nx


class BaseSequenceOptimizer:
    '''
    Base optimizer that selects a disassembly sequence from a planning tree
    produced by SequencePlanner.plan (see plan_sequence/run_seq_plan.py).

    Tree structure (networkx.DiGraph):
        - Nodes are tuples of remaining part-ids (root = all parts, leaves = single part).
        - Node attrs: 'n_eval', 'n_gripper' (None if infeasible), 'poses'.
        - Edge attrs: 'n_eval', 'sim_info' with keys 'feasible', 'part_move', 'pose',
          'parts_fix', 'action', 'grasp', 'dof', 'base_part'.

    A "valid sequence" is a root-to-leaf path where every edge has
    sim_info['feasible'] == True and the leaf is at the lowest level
    (len(node) == 1) with n_gripper is not None. The sequence itself is
    the ordered list of 'part_move' values along that path.
    '''

    def __init__(self, tree):
        self.tree = tree
        self.root = self._find_root(tree)

    @staticmethod
    def _find_root(tree):
        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                return node
        raise ValueError('[optimizer] tree has no root')

    def find_valid_sequences(self):
        '''
        Enumerate every root-to-leaf path through feasible edges whose leaf is a
        valid terminal node (single remaining part with n_gripper is not None).

        Returns: list[list[part_id]] - each inner list is one sequence in removal order.
        '''
        sequences = []
        path = []

        def dfs(node):
            if len(node) == 1:
                if self.tree.nodes[node].get('n_gripper') is not None:
                    sequences.append(list(path))
                return
            for _, child, edata in self.tree.out_edges(node, data=True):
                sim_info = edata.get('sim_info') or {}
                if not sim_info.get('feasible'):
                    continue
                path.append(sim_info['part_move'])
                dfs(child)
                path.pop()

        dfs(self.root)
        return sequences

    def optimize(self):
        '''
        Pick an "optimal" sequence. Base implementation returns a random valid one;
        subclasses override with a real objective (e.g. min grippers, min path length,
        min reorientations).

        Returns: list[part_id] or None if no valid sequence exists.
        '''
        sequences = self.find_valid_sequences()
        if not sequences:
            return None
        return random.choice(sequences)

    def cost_sequence(self, sequence, cost_fn):
        '''
        Walk the root→leaf path defined by `sequence` (an ordered list of
        part_move ids) and sum a per-edge cost.

        cost_fn(child_parts, sim_info, parent_parts) → float — typically a bound
        closure over HeuristicDFASequencePlanner._cost_child.

        Returns (total_cost, per_edge_costs). If any step has no matching
        feasible edge in the tree, returns (None, None).
        '''
        node = self.root
        total = 0.0
        per_edge = []
        for part in sequence:
            match = None
            for _, child, edata in self.tree.out_edges(node, data=True):
                sim_info = edata.get('sim_info') or {}
                if not sim_info.get('feasible'):
                    continue
                if sim_info.get('part_move') == part:
                    match = (child, sim_info)
                    break
            if match is None:
                return None, None
            child, sim_info = match
            c = float(cost_fn(list(child), sim_info, list(node)))
            per_edge.append(c)
            total += c
            node = child
        return total, per_edge

    def optimize_scored(self, cost_fn=None, divide_optimizer=None,
                        threshold=0.1, debug=0):
        '''
        Sibling to ``optimize()``: pick the valid sequence with the LOWEST
        per-edge-cost sum, and (optionally) probe a DivideOptimizer for a
        (S, R) split whose R is decomposed into connected components of the
        contact graph. The split is purely diagnostic — the full chosen
        sequence is always returned. When ``debug > 0`` the chosen sequence,
        its cost, the best split found, the threshold verdict, and the
        per-component sub-sequences (filtered from the full sequence) are
        printed.

        Args:
            cost_fn: callable (child_parts, sim_info, parent_parts) → float
                (lower = better). If None, falls back to ``optimize()``
                (random valid pick).
            divide_optimizer: a DivideOptimizer whose
                ``build_obstruction_graph`` + ``find_locally_free_subassemblies``
                + ``verify_locally_free`` have already run. When None, the
                split section of the diagnostic is suppressed.
            threshold: minimum DivideOptimizer score above which the split is
                flagged "ACCEPTED" in the debug output.
            debug: when > 0, emits the diagnostic block.

        Returns: list[part_id] — the full chosen optimal sequence, or None.
        '''
        sequences = self.find_valid_sequences()
        if not sequences:
            return None
        if cost_fn is None:
            return random.choice(sequences)

        scored = []
        for seq in sequences:
            total, per_edge = self.cost_sequence(seq, cost_fn)
            if total is None:
                continue
            scored.append((seq, total, per_edge))
        if not scored:
            return random.choice(sequences)

        # Lowest total cost wins.
        scored.sort(key=lambda x: x[1])
        best_seq, best_cost, best_per_edge = scored[0]

        split_info = self._extract_split_info(
            best_seq, best_per_edge, divide_optimizer, threshold,
        )

        if debug > 0:
            self._print_optimize_scored_diagnostic(
                best_seq, best_cost, best_per_edge, split_info, threshold,
            )
        return best_seq

    def _extract_split_info(self, best_seq, best_per_edge, divide_optimizer, threshold):
        if divide_optimizer is None:
            return None
        verified = getattr(divide_optimizer, 'verified_locally_free', None) or []
        if not verified:
            return None
        # verified is sorted by divide-optimizer score descending; only the best matters.
        S, R, dscore = verified[0]
        S_set, R_set = set(S), set(R)
        graph = getattr(divide_optimizer, 'obstruction_graph', None)
        if graph is not None:
            present_R = [p for p in R if graph.has_node(p)]
            if len(present_R) <= 1:
                R_components = [frozenset(R_set)]
            else:
                R_components = [
                    frozenset(c) for c in nx.connected_components(graph.subgraph(present_R))
                ]
        else:
            R_components = [frozenset(R_set)]

        sub_sequences = [[p for p in best_seq if p in S_set]] + [
            [p for p in best_seq if p in comp] for comp in R_components
        ]
        sub_scores = []
        for sub_seq in sub_sequences:
            sub_seq_set = set(sub_seq)
            sub_scores.append(
                sum(s for p, s in zip(best_seq, best_per_edge) if p in sub_seq_set)
            )

        return {
            'S': S, 'R_components': R_components,
            'divide_score': dscore,
            'passes_threshold': dscore >= threshold,
            'sub_sequences': sub_sequences, 'sub_scores': sub_scores,
        }

    @staticmethod
    def _print_optimize_scored_diagnostic(best_seq, best_cost, best_per_edge,
                                           split_info, threshold):
        bar = '=' * 64
        print('\n' + bar)
        print('[optimize_scored] Sequence + Split diagnostic')
        print(bar)
        print(f'  Chosen sequence:      {best_seq}')
        print(f'  Sequence cost (sum):  {best_cost:.4f}')
        print(f'  Per-edge costs:       {[round(c, 4) for c in best_per_edge]}')
        if split_info is None:
            print('  Split:                (no verified divide-optimizer candidate)')
        else:
            verdict = 'ACCEPTED' if split_info['passes_threshold'] else 'REJECTED'
            print(f'  Best split score:     {split_info["divide_score"]:.4f}  '
                  f'(threshold={threshold:.4f}) -> {verdict}')
            print(f'  S        = {sorted(split_info["S"])}')
            for i, comp in enumerate(split_info['R_components']):
                print(f'  R[{i}]    = {sorted(comp)}')
            print('  Sub-sequences (filtered from full sequence):')
            labels = ['S'] + [f'R[{i}]' for i in range(len(split_info['R_components']))]
            for label, sub, sscore in zip(labels, split_info['sub_sequences'],
                                          split_info['sub_scores']):
                print(f'    {label:<6}: {sub}  (score = {sscore:.4f})')
            print(f'  Sum of sub-sequence scores: '
                  f'{sum(split_info["sub_scores"]):.4f}')
        print(bar)
