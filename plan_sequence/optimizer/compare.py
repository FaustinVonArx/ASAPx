"""Compare an original disassembly sequence's cost against a split-based
sequence cost: prefix (parts not in S∪R from the original) + one "remove R as
a unified body from S" step + re-planned S and R as independent assemblies.

The cost decomposition mirrors HeuristicDFASequencePlanner._features_child so
weights from settings.heuristic_weights drive both the original and split
walks consistently.
"""
import json
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile

import networkx as nx
import numpy as np

from plan_sequence.physics_planner import CONTACT_EPS, get_contact_graph
from .base import BaseSequenceOptimizer


FEATURE_ORDER = ('contact_distance', 'free_dof', 'z_alignment', 'pose_change',
                 'hold_count')

DEFAULT_WEIGHTS = {
    'contact_distance': 0.8,
    'free_dof': 0.4,
    'z_alignment': 1.0,
    'pose_change': 1.0,
    'hold_count': 3.0,
}


def _load_weights():
    try:
        import settings as user_settings
        cfg = getattr(user_settings, 'heuristic_weights', None)
    except ImportError:
        cfg = None
    weights = dict(DEFAULT_WEIGHTS)
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k in weights:
                weights[k] = float(v)
    return weights


class CostComputer:
    """Standalone replica of HeuristicDFASequencePlanner._features_child /
    _cost_child that doesn't need a SequencePlanner instance — just the asset
    folder, assembly dir, and the parts list of the assembly being walked."""

    def __init__(self, asset_folder, assembly_dir, parts, save_sdf=False, weights=None):
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.parts = list(parts)
        self.save_sdf = save_sdf
        self.weights = weights or _load_weights()
        self._cg = None

    def _contact_graph(self):
        if self._cg is None:
            self._cg = get_contact_graph(
                self.asset_folder, self.assembly_dir, self.parts,
                contact_eps=CONTACT_EPS, save_sdf=self.save_sdf,
            )
        return self._cg

    def features(self, G_prime, sim_info, parent_G, parent_pose=None):
        cg = self._contact_graph()
        moved = list(set(parent_G) - set(G_prime))
        if not moved:
            return np.zeros(len(FEATURE_ORDER), dtype=float)
        candidate = moved[0]
        already_removed = set(self.parts) - set(parent_G)

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

        dof = sim_info.get('dof')
        if dof is None:
            c2 = 0.0
        else:
            arr = np.asarray(dof, dtype=float)
            c2 = float(arr.size - arr.sum())

        action = sim_info.get('action')
        if action is None:
            c3 = 1.0
        else:
            a = np.asarray(action, dtype=float)
            n = float(np.linalg.norm(a))
            c3 = (1.0 - float(a[2] / n)) if n > 1e-9 else 1.0

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

        parts_fix = sim_info.get('parts_fix')
        if parts_fix is None:
            c5 = 0.0
        else:
            c5 = float(len(parts_fix))

        return np.array([c1, c2, c3, c4, c5], dtype=float)

    def cost(self, G_prime, sim_info, parent_G, parent_pose=None):
        phi = self.features(G_prime, sim_info, parent_G, parent_pose=parent_pose)
        w = np.array([self.weights[k] for k in FEATURE_ORDER], dtype=float)
        components = {k: float(phi[i] * w[i]) for i, k in enumerate(FEATURE_ORDER)}
        return float(w @ phi), components


def _tree_root(tree):
    for n in tree.nodes:
        if tree.in_degree(n) == 0:
            return n
    return None


def cost_sequence(sequence, tree, cost_computer):
    """Walk `sequence` through `tree` summing per-edge _cost_child. Returns
    (total_cost, components_dict, n_steps) or None on incompatible sequence."""
    root = _tree_root(tree)
    if root is None:
        return None
    G = root
    total = 0.0
    components = {k: 0.0 for k in FEATURE_ORDER}
    parent_pose = None
    n_steps = 0
    for part in sequence:
        if part not in G:
            continue
        G_prime = tuple(p for p in G if p != part)
        if not tree.has_edge(G, G_prime):
            return None
        edge = tree.edges[G, G_prime]
        sim_info = edge.get('sim_info') or {}
        if not sim_info.get('feasible'):
            return None
        c, comp = cost_computer.cost(G_prime, sim_info, G, parent_pose=parent_pose)
        total += c
        for k, v in comp.items():
            components[k] += v
        parent_pose = sim_info.get('pose')
        G = G_prime
        n_steps += 1
    return total, components, n_steps


def _prepare_subassembly_dir(assembly_dir, parts_subset):
    """Create a temp dir containing only the .obj files and config.json
    entries for `parts_subset`. Caller is responsible for cleanup."""
    temp_dir = tempfile.mkdtemp(prefix='subasm_')
    for part in parts_subset:
        src = os.path.join(assembly_dir, f'{part}.obj')
        if not os.path.exists(src):
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(f'Missing part .obj: {src}')
        shutil.copy(src, os.path.join(temp_dir, f'{part}.obj'))

    config_path = os.path.join(assembly_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        sub_config = {p: config[p] for p in parts_subset if p in config}
        with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
            json.dump(sub_config, f)
    return temp_dir


def _run_subassembly_into_queue(args, queue, idx):
    """Top-level wrapper so the child process can pickle the target. Runs
    `_plan_subassembly_worker(args)`, captures any exception, and pushes
    `(idx, result_dict)` onto `queue`."""
    try:
        result = _plan_subassembly_worker(args)
    except Exception as e:
        import traceback
        label = args[5] if len(args) > 5 else '?'
        result = {'label': label,
                  'error': f'{e}\n{traceback.format_exc()}',
                  'temp_dir': ''}
    queue.put((idx, result))


def _plan_subassembly_worker(args):
    """Multiprocessing worker: re-plans one subassembly. Returns a dict with
    'sequence', 'tree' (pickled bytes), and 'temp_dir' (assembly dir used,
    needed by the caller's CostComputer for the contact graph)."""
    asset_folder, parts_subset, assembly_dir, num_proc, planner_kwargs, label = args
    temp_dir = _prepare_subassembly_dir(assembly_dir, parts_subset)
    log_dir = tempfile.mkdtemp(prefix=f'subasm_{label}_log_')
    try:
        from plan_sequence.run_seq_plan import seq_plan
        # Treat the subassembly like a normal assembly: no base_part override
        # (planner runs the full self-stable-pose precheck + per-step stable-
        # pose computation), and stability defers to settings.skip_stability
        # so gravity / parts_fix behaves consistently with non-split runs.
        try:
            import settings as _user_settings
            _skip_stab = bool(getattr(_user_settings, 'skip_stability', False))
        except ImportError:
            _skip_stab = False
        seq_plan(
            asset_folder=asset_folder,
            assembly_dir=temp_dir,
            generator_name=planner_kwargs.get('generator', 'rand'),
            planner_name=planner_kwargs.get('planner', 'dfa'),
            num_proc=num_proc,
            seed=planner_kwargs.get('seed', 0),
            budget=planner_kwargs.get('budget', 400),
            max_gripper=planner_kwargs.get('max_gripper', 3),
            max_pose=planner_kwargs.get('max_pose', 3),
            pose_reuse=planner_kwargs.get('pose_reuse', 0),
            early_term=False,
            timeout=planner_kwargs.get('timeout', 600),
            base_part=planner_kwargs.get('base_part'),
            save_sdf=False, clear_sdf=False,
            plan_grasp=False, plan_arm=False,
            gripper_type=None, gripper_scale=None,
            optimizer='L-BFGS-B', debug=0, render=False,
            record_dir=None, log_dir=log_dir,
            get_dof=True, skip_stability=_skip_stab,
        )
        stats_path = os.path.join(log_dir, 'stats.json')
        tree_path = os.path.join(log_dir, 'tree.pkl')
        if not (os.path.exists(stats_path) and os.path.exists(tree_path)):
            return {'label': label, 'error': 'no stats/tree produced', 'temp_dir': temp_dir}
        with open(stats_path) as f:
            stats = json.load(f)
        with open(tree_path, 'rb') as f:
            tree_bytes = f.read()
        return {
            'label': label,
            'sequence': stats.get('sequence'),
            'success': stats.get('success', False),
            'tree_bytes': tree_bytes,
            'temp_dir': temp_dir,  # caller cleans this up after costing
        }
    except Exception as e:
        import traceback
        return {'label': label, 'error': f'{e}\n{traceback.format_exc()}',
                'temp_dir': temp_dir}
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def _split_step_cost(asset_folder, assembly_dir, parts_S, parts_R, weights):
    """Cost the "remove unified R from unified S" step. Reuses
    verify_separation's combine-and-scan approach to find a feasible
    direction; cost is computed as if R were a single moving part with that
    direction as the action. Returns (cost, components, direction) or
    (None, None, None) on failure."""
    from plan_sequence.physics_planner import (FORCE_MAG, MultiPartPathPlanner,
                                               _VERIFY_DIRECTIONS)
    from plan_sequence.stable_pose import get_combined_mesh

    if not parts_S or not parts_R:
        return None, None, None

    tmp_dir = tempfile.mkdtemp(prefix='split_step_cost_')
    try:
        get_combined_mesh(assembly_dir, list(parts_S)).export(os.path.join(tmp_dir, 'S.obj'))
        get_combined_mesh(assembly_dir, list(parts_R)).export(os.path.join(tmp_dir, 'R.obj'))
        planner = MultiPartPathPlanner(
            asset_folder=asset_folder, assembly_dir=tmp_dir,
            parts_fix=['S'], part_move='R', pose=None,
            force_mag=FORCE_MAG, save_sdf=False,
        )
        planner.max_time = 30
        chosen = None
        for direction in _VERIFY_DIRECTIONS:
            if planner.check_success(direction):
                chosen = direction
                break
        if chosen is None:
            return None, None, None

        z = float(chosen[2]) / float(np.linalg.norm(chosen))
        z_alignment = 1.0 - z

        # No previously-removed parts in this two-body view, no pose, no
        # parts_fix, no DoF probe — leave those features at 0.
        components = {
            'contact_distance': 0.0,
            'free_dof': 0.0,
            'z_alignment': z_alignment * weights['z_alignment'],
            'pose_change': 0.0,
            'hold_count': 0.0,
        }
        return float(z_alignment * weights['z_alignment']), components, list(chosen)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compare_with_split(tree, asset_folder, assembly_dir, split,
                       original_sequence=None, num_proc=8, planner_kwargs=None,
                       debug=0):
    """Compare original plan cost vs split plan cost.

    Split plan structure:
      1. Prefix: parts of the original sequence that aren't in S∪R, in order.
      2. Split step: R (unified body) separates from S (unified body).
      3. S-internal: re-planned sequence for S as an independent assembly.
      4. R-internal: re-planned sequence for R as an independent assembly.
    S and R are re-planned in parallel.

    Returns a dict with both sequences and their decomposed costs.
    """
    parts_S = sorted(split['S'])
    parts_R = sorted(split['R'])
    weights = _load_weights()

    root = _tree_root(tree)
    full_parts = sorted(root) if root else sorted(set(parts_S) | set(parts_R))

    if original_sequence is None:
        try:
            original_sequence = BaseSequenceOptimizer(tree).optimize()
        except Exception as e:
            print(f'[compare_with_split] BaseSequenceOptimizer error: {e}')
            original_sequence = None

    cc_full = CostComputer(asset_folder, assembly_dir, full_parts, weights=weights)

    original_cost_result = (cost_sequence(original_sequence, tree, cc_full)
                            if original_sequence else None)

    in_split = set(parts_S) | set(parts_R)
    prefix_sequence = [p for p in (original_sequence or []) if p not in in_split]
    prefix_cost_result = cost_sequence(prefix_sequence, tree, cc_full)

    # Re-plan S and R in parallel.
    planner_kwargs = dict(planner_kwargs or {})
    sub_num_proc = max(num_proc // 2, 1)

    worker_args = [
        (asset_folder, parts_S, assembly_dir, sub_num_proc, planner_kwargs, 'S'),
        (asset_folder, parts_R, assembly_dir, sub_num_proc, planner_kwargs, 'R'),
    ]
    if debug:
        print(f'[compare_with_split] re-planning S ({len(parts_S)} parts) and '
              f'R ({len(parts_R)} parts) in parallel; sub-num_proc={sub_num_proc}')

    # Raw non-daemonic Process workers (NOT Pool): Pool workers are daemonic by
    # default, and daemonic procs can't spawn children — but the inner DFA
    # planner uses parallel_execute → mp.Process to parallelise its own per-step
    # feasibility sims, so workers MUST be allowed to spawn. Fork context so the
    # parent's already-evicted-ATA module state carries over (see the spawn
    # vs fork note above the original commit).
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    procs = []
    for idx, wargs in enumerate(worker_args):
        p = ctx.Process(target=_run_subassembly_into_queue, args=(wargs, queue, idx))
        p.daemon = False
        p.start()
        procs.append(p)

    received = {}
    for _ in worker_args:
        idx, result = queue.get()
        received[idx] = result
    for p in procs:
        p.join()

    s_result, r_result = received[0], received[1]

    def _cost_sub(parts_sub, sub_result):
        if 'error' in sub_result and sub_result.get('error'):
            return None
        seq = sub_result.get('sequence')
        if not seq:
            return None
        try:
            sub_tree = pickle.loads(sub_result['tree_bytes'])
            cc_sub = CostComputer(asset_folder, sub_result['temp_dir'], parts_sub, weights=weights)
            return cost_sequence(seq, sub_tree, cc_sub)
        finally:
            shutil.rmtree(sub_result.get('temp_dir', ''), ignore_errors=True)

    s_cost_result = _cost_sub(parts_S, s_result)
    r_cost_result = _cost_sub(parts_R, r_result)

    # Cost the split step (unified R separates from unified S).
    split_cost, split_components, split_dir = _split_step_cost(
        asset_folder, assembly_dir, parts_S, parts_R, weights,
    )

    # Aggregate the split-plan total.
    def _safe(c):
        return c if c is not None else (0.0, {k: 0.0 for k in FEATURE_ORDER}, 0)

    p_cost, p_comp, p_n = _safe(prefix_cost_result)
    s_cost, s_comp, s_n = _safe(s_cost_result)
    r_cost, r_comp, r_n = _safe(r_cost_result)
    sc = split_cost or 0.0
    sc_comp = split_components or {k: 0.0 for k in FEATURE_ORDER}

    total_split = p_cost + sc + s_cost + r_cost
    total_components = {
        k: p_comp[k] + sc_comp[k] + s_comp[k] + r_comp[k] for k in FEATURE_ORDER
    }
    total_steps = p_n + (1 if split_cost is not None else 0) + s_n + r_n

    return {
        'weights': weights,
        'original_sequence': original_sequence,
        'original_cost': original_cost_result,
        'prefix_sequence': prefix_sequence,
        'prefix_cost': prefix_cost_result,
        'split_S': parts_S,
        'split_R': parts_R,
        'split_step_cost': split_cost,
        'split_step_components': split_components,
        'split_step_direction': split_dir,
        'S_sequence': s_result.get('sequence'),
        'S_cost': s_cost_result,
        'S_error': s_result.get('error'),
        'R_sequence': r_result.get('sequence'),
        'R_cost': r_cost_result,
        'R_error': r_result.get('error'),
        'split_total_cost': total_split,
        'split_total_components': total_components,
        'split_total_steps': total_steps,
    }


def print_comparison(result):
    """Pretty-print a compare_with_split result to stdout."""
    print('=' * 70)
    print('Sequence cost comparison: original vs subassembly-split')
    print('=' * 70)
    print(f'Weights: {result["weights"]}')
    print()

    orig = result.get('original_cost')
    print('--- Original (flat) sequence ---')
    print(f'  sequence: {result.get("original_sequence")}')
    if orig is not None:
        c, comp, n = orig
        print(f'  total cost: {c:.4f}  ({n} steps)')
        for k in FEATURE_ORDER:
            print(f'    {k:>18}: {comp[k]:.4f}')
    else:
        print('  cost: N/A')

    print()
    print('--- Split sequence ---')
    print(f'  prefix:   {result.get("prefix_sequence")}')
    print(f'  split:    S={result["split_S"]}  R={result["split_R"]}  '
          f'dir={result.get("split_step_direction")}')
    print(f'  S inner:  {result.get("S_sequence")}')
    if result.get('S_error'):
        print(f'  S error: {result["S_error"].splitlines()[0]}')
    print(f'  R inner:  {result.get("R_sequence")}')
    if result.get('R_error'):
        print(f'  R error: {result["R_error"].splitlines()[0]}')

    print()
    print('  cost breakdown:')

    def _row(label, cost_result, sc=None, sc_comp=None):
        if cost_result is None and sc is None:
            print(f'    {label:>14}: N/A')
            return
        if cost_result is None:
            c, comp, n = sc, sc_comp, 1
        elif sc is not None:
            c = cost_result[0] + sc
            comp = {k: cost_result[1][k] + sc_comp[k] for k in FEATURE_ORDER}
            n = cost_result[2] + 1
        else:
            c, comp, n = cost_result
        parts = '  '.join(f'{k[:4]}={comp[k]:.3f}' for k in FEATURE_ORDER)
        print(f'    {label:>14}: {c:.4f}  ({n} steps)  {parts}')

    _row('prefix', result.get('prefix_cost'))
    if result.get('split_step_cost') is not None:
        _row('split step', None, result['split_step_cost'], result['split_step_components'])
    else:
        print(f'    {"split step":>14}: N/A')
    _row('S inner', result.get('S_cost'))
    _row('R inner', result.get('R_cost'))
    print(f'    {"TOTAL split":>14}: {result["split_total_cost"]:.4f}  '
          f'({result["split_total_steps"]} steps)')

    print()
    if orig is not None:
        diff = result['split_total_cost'] - orig[0]
        verdict = ('split CHEAPER by ' + f'{-diff:.4f}') if diff < 0 else (
                  'split COSTLIER by ' + f'{diff:.4f}' if diff > 0 else 'TIE')
        print(f'  Δ (split − original) = {diff:+.4f}   →  {verdict}')
    print('=' * 70)
