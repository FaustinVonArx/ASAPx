import random


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
