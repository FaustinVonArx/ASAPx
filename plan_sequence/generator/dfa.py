import os
import tempfile

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from plan_sequence.feasibility_check import get_contact_graph
from .base import Generator

class DFAGenerator(Generator):
    G = None

    def _build_contact_graph(self, parts=None):
        if parts is None:
            parts = self.parts
        return get_contact_graph(self.asset_folder, self.assembly_dir, parts=parts, save_sdf=self.save_sdf)

    def _init_G(self):
        self.G = nx.DiGraph()
        self.G.add_node(tuple(self.parts))

    def scan_for_subassemblies(self, current_parts, dof_info, successful, unsuccessful, save_path=None):
        '''
        Build a contact graph over current_parts annotated with per-part DoF and
        removability info, then visualize it (green = removable, red = not).
        Args:
            current_parts: parts currently in the (sub)assembly
            dof_info: dict mapping part_id -> 6-element DoF array (or None if unknown)
            successful: list of part_ids that were successfully disassembled this step
            unsuccessful: list of part_ids that failed to disassemble this step
            save_path: optional output png path; if None, a temp file is used
        Returns:
            The annotated networkx Graph.
        '''
        G = self._build_contact_graph(parts=current_parts)
        successful_set = set(successful)
        attempted_set = set(successful) | set(unsuccessful)
        for part in current_parts:
            if part not in G.nodes:
                G.add_node(part)
            G.nodes[part]['dof'] = dof_info.get(part)
            G.nodes[part]['removable'] = part in successful_set
            G.nodes[part]['attempted'] = part in attempted_set
        self._plot_subassembly_graph(G, save_path=save_path)
        return G

    def _plot_subassembly_graph(self, G, save_path=None):
        nodes = list(G.nodes)
        node_colors = ['green' if G.nodes[n].get('removable') else 'red' for n in nodes]
        labels = {}
        for n in nodes:
            dof = G.nodes[n].get('dof')
            if dof is not None:
                dof_str = ''.join(str(int(x)) for x in np.asarray(dof).flatten())
                labels[n] = f'{n}\n{dof_str}'
            else:
                labels[n] = f'{n}\n--'
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(G, pos, ax=ax, nodelist=nodes, node_color=node_colors, labels=labels,
                with_labels=True, node_size=900, font_size=8)
        if save_path is None:
            with tempfile.NamedTemporaryFile(suffix='.png', prefix='dfa_subassembly_', delete=False) as tmp:
                save_path = tmp.name
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[scan_for_subassemblies] saved to {save_path}')