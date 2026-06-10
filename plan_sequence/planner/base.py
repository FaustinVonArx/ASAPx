import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import random
import torch
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from time import time
import traceback

from assets.load import load_part_ids
from plan_sequence.sim_string import get_body_color_dict
from plan_sequence.physics_planner import MultiPartPathPlanner, get_body_mass, find_stable_initial_poses
from plan_sequence.feasibility_check import check_assemblable, get_stable_plan_1pose_parallel, get_stable_plan_1pose_serial, check_tool
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses, translation_pose_to_ground
from plan_robot.run_grasp_plan import GraspPlanner
from plan_robot.run_grasp_arm_plan import GraspArmPlanner
import settings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _below_ground_after_disassembly(assembly_dir, part_move, pose, path):
    """Return True if the moving part's COG ends up at z<0 after the disassembly
    path, i.e. the action drove it through the resting plane. Assumes pure
    translation along the path (no body rotation), which is a reasonable
    approximation for short disassembly motions."""
    if path is None or len(path) == 0:
        return False
    part_mesh = get_combined_mesh(assembly_dir, [part_move])
    cog_local = np.asarray(part_mesh.center_mass, dtype=float)
    if pose is not None:
        cog_world_init = pose[:3, :3] @ cog_local + pose[:3, 3]
    else:
        cog_world_init = cog_local
    displacement = np.asarray(path[-1][:3], dtype=float) - np.asarray(path[0][:3], dtype=float)
    return float(cog_world_init[2] + displacement[2]) < 0.0


def _simulate_standalone(asset_folder, assembly_dir, save_sdf, base_part,
                          part_move, parts_rest, parts_removed, pose,
                          max_grippers, timeout, optimizer, debug, render,
                          allow_gap=False, get_dof=False, tools=None,
                          skip_stability=False, ignore_unstable=()):
    """Module-level wrapper around _simulate logic, picklable for use with parallel_execute."""
    _t0 = time()
    diag_assembly = {}
    if get_dof:
        action, path, dof = check_assemblable(
            asset_folder, assembly_dir, parts_rest, part_move,
            pose=pose, save_sdf=save_sdf, return_path=True, debug=debug, render=render, get_dof=True,
            diagnostics=diag_assembly,
        )
    else:
        action, path = check_assemblable(
            asset_folder, assembly_dir, parts_rest, part_move,
            pose=pose, save_sdf=save_sdf, return_path=True, debug=debug, render=render,
            diagnostics=diag_assembly,
        )
        dof = None
    dt_path = time() - _t0

    if action is not None and settings.filter_below_ground and _below_ground_after_disassembly(assembly_dir, part_move, pose, path):
        action = None

    dt_stab = None
    diag_stability = {}
    if action is None:
        parts_fix = None
    elif skip_stability:
        # Caller opted out of the multi-part stability check — treat the
        # subassembly as if no extra fixes are needed.
        parts_fix = []
    else:
        max_fix = max_grippers - 1 if max_grippers is not None else None
        _t0 = time()
        parts_fix = get_stable_plan_1pose_serial(
            asset_folder, assembly_dir, parts_rest, base_part,
            pose=pose, max_fix=max_fix, save_sdf=save_sdf,
            timeout=timeout, allow_gap=allow_gap, debug=debug, render=render,
            ignore_unstable=ignore_unstable, diagnostics=diag_stability,
        )
        dt_stab = time() - _t0

    # `tools` may be either a flat list (legacy) or a per-part dict produced
    # by the analyze_assembly_tools pipeline: {part_id: [chosen_tool] or []}.
    # An empty list means the VLM decided no tool is needed for this part —
    # tool check should not gate feasibility.
    if isinstance(tools, dict):
        effective_tools = tools.get(str(part_move), [])
    else:
        effective_tools = tools

    tool_result = None
    dt_tool = None
    diag_tool = {}
    if effective_tools and action is not None and parts_fix is not None:
        _t0 = time()
        tool_result = check_tool(
            asset_folder=asset_folder, assembly_dir=assembly_dir,
            parts_fix=parts_rest, part_move=part_move, tools=effective_tools, debug=debug,
            diagnostics=diag_tool,
        )
        dt_tool = time() - _t0

    tool_required = effective_tools is not None and len(effective_tools) > 0
    feasible = action is not None and parts_fix is not None and (not tool_required or tool_result is not None)
    if feasible:
        fail_reason = None
        fail_evidence = None
    elif action is None:
        fail_reason = 'assembly'
        fail_evidence = diag_assembly or None
    elif parts_fix is None:
        fail_reason = 'stability'
        fail_evidence = diag_stability or None
    else:
        fail_reason = 'tool'
        fail_evidence = diag_tool or None

    return {
        'feasible': feasible,
        'fail_reason': fail_reason,
        'fail_evidence': fail_evidence,
        'action': action,
        'base_part': base_part,
        'parts_fix': parts_fix,
        'part_move': part_move,
        'pose': pose,
        'grasp': None,
        'dof': dof,
        'tool': None if tool_result is None else {'tool_id': tool_result['tool_id'], 'inverted': tool_result['inverted']},
        '_dt_path': dt_path,
        '_dt_stab': dt_stab,
        '_dt_tool': dt_tool,
    }


def _simulate_standalone_tagged(*args):
    """Variant of `_simulate_standalone` whose last positional arg is a parent_idx
    tag. The tag is stripped before the real call and re-attached on the returned
    sim_info dict as `_parent_idx`. Used by the multi-frontier DFA loop so the
    termination callback (which only sees sim_info) can track per-parent quotas."""
    parent_idx = args[-1]
    sim_info = _simulate_standalone(*args[:-1])
    sim_info['_parent_idx'] = parent_idx
    return sim_info


class SequencePlanner:
    '''
    Disassembly sequence planning (without ground, 1 direction gravity, 1 part at a time)
    '''
    def __init__(self, seq_generator, num_proc=1, save_sdf=False, allow_gap=False, get_dof=False, tools=None, skip_stability=False, ignore_unstable=()):
        self.seq_generator = seq_generator
        self.asset_folder = seq_generator.asset_folder
        self.assembly_dir = seq_generator.assembly_dir
        self.base_part = seq_generator.base_part
        self.parts = sorted(load_part_ids(self.assembly_dir))
        assert len(self.parts) >= 2
        self.num_proc = num_proc
        self.save_sdf = save_sdf
        self.allow_gap = allow_gap
        self.get_dof = get_dof
        self.tools = tools
        self.skip_stability = skip_stability
        # Populated by plan() when settings.no_stable_pose_action == 'ignore_unstable'
        # and the precheck observed parts falling. Forwarded into every downstream
        # stability check so those parts no longer count as failures.
        self._ignored_unstable_parts = frozenset(ignore_unstable)
        self.part_mass = get_body_mass(self.asset_folder, self.assembly_dir, self.parts, save_sdf=self.save_sdf)
        self.t_start = None
        self.n_eval = None
        self.stop_msg = None
        self._timing = defaultdict(float)
        self._timing_counts = defaultdict(int)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _parent_pose_for(tree, parent_G):
        """Look up the pose used to reach `parent_G` from its tree predecessor.
        Returns None for the root (no in-edge) or when the in-edge has no
        stored pose. Lives on the base class so any planner can use it without
        having to inherit from HeuristicDFASequencePlanner."""
        if tree is None or parent_G is None:
            return None
        key = tuple(parent_G)
        if not tree.has_node(key):
            return None
        for pred in tree.predecessors(key):
            sim_info = tree.edges[pred, key].get('sim_info', {})
            return sim_info.get('pose')
        return None

    @staticmethod
    def _rotation_angle_between(pose_a, pose_b):
        """Angular distance in radians between the rotation blocks of two
        4x4 SE(3) poses. Returns +inf when either pose is missing or non-
        conformable, so callers using this as a sort key naturally push
        bad / None poses to the end."""
        if pose_a is None or pose_b is None:
            return float('inf')
        try:
            Ra = np.asarray(pose_a, dtype=float)[:3, :3]
            Rb = np.asarray(pose_b, dtype=float)[:3, :3]
            R_diff = Ra.T @ Rb
            trace = float(np.trace(R_diff))
            cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
            return float(np.arccos(cos_theta))
        except (ValueError, TypeError):
            return float('inf')

    @staticmethod
    def _sort_poses_by_proximity(poses, ref_pose):
        """Return poses ordered by ascending rotation-angle distance to
        ref_pose. When ref_pose is None or poses is empty, returns a shallow
        copy unchanged so the trimesh probability ordering is preserved."""
        if ref_pose is None or not poses:
            return list(poses)
        return sorted(
            poses,
            key=lambda p: SequencePlanner._rotation_angle_between(ref_pose, p),
        )

    def _pick_initial_pose_interactive(self, parts, initial_poses, log_dir=None,
                                       fallback_candidates=None, per_pose=None):
        """When settings.interactive_initial_pose is True, render every
        candidate initial pose, list them to stdin, and ask the user to pick
        one as the assembly's starting orientation. Returns a single-element
        list when the user picks, or the original verified-pose list when the
        user skips / runs in a non-interactive shell / the flag is off.

        fallback_candidates: optional broader list (typically the unverified
        trimesh stable poses for the full assembly). When the verified
        `initial_poses` has fewer than 2 entries (e.g. the gravity precheck
        with no_stable_pose_action='continue' rejected most candidates), the
        picker falls back to this list so the user still has options to
        compare. Without this fallback the flag would silently do nothing
        whenever the precheck filter is aggressive.

        per_pose: optional list of {'pose_idx', 'pose', 'success', 'fallen'}
        from `find_stable_initial_poses`. When the picker is off, the
        non-interactive auto-pick uses this to pick the single pose with the
        fewest falling parts; ties are broken by pose_idx (trimesh's own
        highest-probability ordering). Without per_pose the function keeps
        the legacy "return initial_poses unchanged" behavior (used by the
        'skip' branch where no gravity check ran).
        """
        if not getattr(settings, 'interactive_initial_pose', False):
            return self._auto_pick_initial_pose(initial_poses, per_pose)
        interactive = bool(getattr(sys.stdin, 'isatty', lambda: False)())
        if not interactive:
            if len(initial_poses) > 1:
                print('[init_pose] non-interactive shell; auto-picking by fewest-fallen + trimesh probability.')
            return self._auto_pick_initial_pose(initial_poses, per_pose)

        # Build the candidate set shown to the user. Prefer the verified
        # initial_poses; if too few, augment with the unverified fallback so
        # the user always has a meaningful set to choose from.
        candidates = list(initial_poses)
        fallback = list(fallback_candidates) if fallback_candidates else []
        augmented = False
        if len(candidates) < 2 and len(fallback) > len(candidates):
            if len(candidates) == 0:
                print(f'[init_pose] precheck found no self-stable initial pose; '
                      f'offering {len(fallback)} unverified trimesh candidate(s) '
                      f'so the picker still has something to show.')
            else:
                print(f'[init_pose] precheck verified only {len(candidates)} pose(s); '
                      f'augmenting with {len(fallback) - len(candidates)} unverified '
                      f'trimesh candidate(s) for the picker.')
            candidates = list(fallback)
            augmented = True

        if not candidates:
            print('[init_pose] no candidate poses available; nothing to pick from.')
            return initial_poses

        from plan_sequence.planner._renders import render_part_in_context
        from pathlib import Path as _Path

        cache_base = _Path(log_dir) if log_dir else _Path('/tmp/init_pose_renders')
        cache_dir = cache_base / 'initial_pose_renders'

        print('\n' + '=' * 70)
        label = 'candidate' if len(candidates) == 1 else 'candidates'
        marker = ' (unverified — gravity precheck filtered)' if augmented else ''
        print(f'[init_pose] choose initial stable pose ({len(candidates)} {label}){marker}:')
        for i, pose in enumerate(candidates):
            try:
                png = render_part_in_context(
                    self.assembly_dir, list(parts), None,
                    cache_dir=str(cache_dir), size=(512, 512), pose=pose,
                )
            except Exception as e:
                png = f'(render failed: {e})'
            print(f'  [{i}] {png}')

        try:
            raw = input('[init_pose] pick index (Enter to use [0]): ').strip()
        except EOFError:
            print('[init_pose] EOF; using [0].')
            return [candidates[0]]
        if raw == '':
            print('[init_pose] using pose [0].')
            return [candidates[0]]
        try:
            idx = int(raw)
        except ValueError:
            print(f'[init_pose] invalid input {raw!r}; using [0].')
            return [candidates[0]]
        if not (0 <= idx < len(candidates)):
            print(f'[init_pose] index {idx} out of range; using [0].')
            return [candidates[0]]
        print(f'[init_pose] using pose [{idx}].')
        return [candidates[idx]]

    def _auto_pick_initial_pose(self, initial_poses, per_pose):
        """Non-interactive single-pose pick when settings.interactive_initial_pose
        is off (or stdin is not a tty). Selects the candidate with the fewest
        falling parts; ties are broken by trimesh's own probability ordering
        (lower pose_idx = higher prob). Falls back to `initial_poses`
        unchanged when `per_pose` is unavailable (e.g. the 'skip' branch
        skipped the gravity check entirely).

        Crucially, when NO candidate is verified-stable and the user asked
        the planner to abort on that condition (no_stable_pose_action='exit'),
        we return `initial_poses` (which is the empty verified-stable list)
        so the exit gate in plan() triggers. Falling back to the 'least bad'
        unstable pose would silently override the user's setting.
        """
        if not per_pose:
            return initial_poses
        # Sort by (fallen_count, pose_idx). Success poses (empty fallen) come first,
        # then those with the least falling, with trimesh-probability as tiebreaker.
        ranked = sorted(per_pose, key=lambda e: (len(e.get('fallen') or []), e['pose_idx']))
        best = ranked[0]
        n_fallen = len(best.get('fallen') or [])
        action = getattr(settings, 'no_stable_pose_action', 'exit')
        # Only accept an unstable fallback when the user explicitly opted in.
        if not best['success'] and action not in ('continue', 'ignore_unstable'):
            print(f"[init_pose] auto-pick: no verified-stable pose; respecting "
                  f"settings.no_stable_pose_action={action!r} (will trigger the "
                  f"plan() exit gate).")
            return initial_poses
        marker = 'verified-stable' if best['success'] else f'{n_fallen} fallen'
        print(f"[init_pose] auto-picked pose [{best['pose_idx']}] "
              f"({marker}; trimesh rank {best['pose_idx']} of {len(per_pose)}).")
        return [best['pose']]

    def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers, timeout=None, grasp_planner=None, optimizer='L-BFGS-B', debug=0, render=False, log_dir=None):
        assert len(parts_rest) > 0
        _t0 = time()
        diag_assembly = {}
        if self.get_dof:
            action, path, dof = check_assemblable(self.asset_folder, self.assembly_dir, parts_rest, part_move, pose=pose, save_sdf=self.save_sdf,
                return_path=True, debug=debug, render=render, get_dof=True, diagnostics=diag_assembly)
        else:
            action, path = check_assemblable(self.asset_folder, self.assembly_dir, parts_rest, part_move, pose=pose, save_sdf=self.save_sdf,
                return_path=True, debug=debug, render=render, diagnostics=diag_assembly)
            dof = None
        _dt_path = time() - _t0
        self._timing['path_finding'] += _dt_path
        self._timing_counts['path_finding'] += 1
        if debug > 0:
            print(f'[planner._simulate] path_finding: {_dt_path:.3f}s  found={action is not None}')

        #NOTE when on, a theoretically feasible policy might not be found
        if action is not None and settings.filter_below_ground and _below_ground_after_disassembly(self.assembly_dir, part_move, pose, path):
            action = None

        diag_stability = {}
        if action is None:
            parts_fix_list = None
        elif self.skip_stability:
            # Caller opted out of the multi-part stability check.
            parts_fix_list = [[]]
        else:
            max_fix = max_grippers - 1 if max_grippers is not None else None
            _t0 = time()
            step_label = f'part{part_move}_eval{self.n_eval}' if log_dir is not None else None
            parts_fix_list = get_stable_plan_1pose_serial(self.asset_folder, self.assembly_dir, parts_rest, self.base_part, pose=pose, max_fix=max_fix, save_sdf=self.save_sdf, timeout=timeout, debug=debug, render=render, log_dir=log_dir, step_label=step_label, ignore_unstable=self._ignored_unstable_parts, diagnostics=diag_stability)
            _dt_stab = time() - _t0
            self._timing['stability_check'] += _dt_stab
            self._timing_counts['stability_check'] += 1
            if debug > 0:
                print(f'[planner._simulate] stability_check: {_dt_stab:.3f}s  stable={parts_fix_list is not None}')
            if parts_fix_list is not None: parts_fix_list = [parts_fix_list]

        if parts_fix_list is None:
            parts_fix = None
        elif len(parts_fix_list) == 0:
            parts_fix = []
        else:
            parts_fix = parts_fix_list[0] # NOTE: only pick the first feasible fix list, can be changed

        feasible = action is not None and parts_fix is not None

        if feasible and grasp_planner is not None:
            _t0 = time()
            grasps = grasp_planner.plan(part_move, parts_rest, parts_removed, pose, path, optimizer)
            _dt_grasp = time() - _t0
            self._timing['grasp_planning'] += _dt_grasp
            self._timing_counts['grasp_planning'] += 1
            if debug > 0:
                print(f'[planner._simulate] grasp_planning: {_dt_grasp:.3f}s')
            if len(grasps) == 0:
                grasps = None
                feasible = False
        else:
            grasps = None

        # Resolve per-part tool decision when self.tools is a dict (see
        # _simulate_standalone for the format); otherwise treat as flat list.
        if isinstance(self.tools, dict):
            effective_tools = self.tools.get(str(part_move), [])
        else:
            effective_tools = self.tools

        tool_result = None
        diag_tool = {}
        if feasible and effective_tools:
            _t0 = time()
            tool_result = check_tool(
                asset_folder=self.asset_folder, assembly_dir=self.assembly_dir,
                parts_fix=parts_rest, part_move=part_move, tools=effective_tools, debug=debug,
                diagnostics=diag_tool,
            )
            self._timing['tool_check'] += time() - _t0
            self._timing_counts['tool_check'] += 1
            if tool_result is None:
                feasible = False

        if feasible:
            fail_reason = None
            fail_evidence = None
        elif action is None:
            fail_reason = 'assembly'
            fail_evidence = diag_assembly or None
        elif parts_fix is None:
            fail_reason = 'stability'
            fail_evidence = diag_stability or None
        elif effective_tools and tool_result is None:
            fail_reason = 'tool'
            fail_evidence = diag_tool or None
        else:
            fail_reason = 'grasp'
            fail_evidence = None

        sim_info = {
            'feasible': feasible,
            'fail_reason': fail_reason,
            'fail_evidence': fail_evidence,
            'action': action,
            'base_part': self.base_part,
            'parts_fix': parts_fix,
            'part_move': part_move,
            'pose': pose,
            'grasp': grasps,
            'dof': dof,
            'tool': None if tool_result is None else {'tool_id': tool_result['tool_id'], 'inverted': tool_result['inverted']},
        }
        return sim_info

    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        if sim_info['feasible']: 
            assert tree.nodes[tuple(parts_parent)]['n_gripper'] is not None, f'[error] parent {parts_parent} not feasible'
        if tree.has_node(tuple(parts_child)):
            assert tree.nodes[tuple(parts_child)]['n_eval'] < n_eval,  f'[error] child {parts_child} incorrectly updated'
            if sim_info['feasible']:
                child_n_gripper, parent_n_gripper = tree.nodes[tuple(parts_child)]['n_gripper'], tree.nodes[tuple(parts_parent)]['n_gripper']
                child_n_gripper_new = max(parent_n_gripper, len(sim_info['parts_fix']) + 1)
                if child_n_gripper is None:
                    tree.nodes[tuple(parts_child)]['n_gripper'] = child_n_gripper_new
                else:
                    tree.nodes[tuple(parts_child)]['n_gripper'] = min(child_n_gripper, child_n_gripper_new)
                for pose in tree.nodes[tuple(parts_child)]['poses']: # check if pose is already in node attr
                    if sim_info['pose'] is None:
                        if pose is None:
                            break # both None -> in node attr
                    else:
                        if pose is not None and np.allclose(pose, sim_info['pose']):
                            break # both not None and allclose -> in node attr
                else:
                    tree.nodes[tuple(parts_child)]['poses'].insert(0, sim_info['pose']) # not in node attr, prioritize later poses
        else:
            if sim_info['feasible']:
                tree.add_node(tuple(parts_child), n_eval=n_eval, n_gripper=\
                    max(tree.nodes[tuple(parts_parent)]['n_gripper'], len(sim_info['parts_fix']) + 1), poses=[sim_info['pose']])
            else:
                tree.add_node(tuple(parts_child), n_eval=n_eval, n_gripper=None, poses=[])
        tree.add_edge(tuple(parts_parent), tuple(parts_child), n_eval=n_eval, sim_info=sim_info)

    def _check_fully_explored(self, tree, root_node): # code can be optimized
        # if all childs are recursively explored and failed, then fully explored
        assert tree.has_node(tuple(root_node))
        assert len(root_node) >= 2
        if len(root_node) == 2: return True # NOTE: assume max_gripper >= 2

        for part_move in root_node:
            if self.base_part is not None and part_move == self.base_part: continue
            child_node = root_node.copy()
            child_node.remove(part_move)
            if tree.has_edge(tuple(root_node), tuple(child_node)):
                if tree.edges[tuple(root_node), tuple(child_node)]['sim_info']['feasible']:
                    if not self._check_fully_explored(tree, child_node): return False # child not fully explored
            else:
                return False # no such child
        return True

    def plan(self, budget, max_grippers, max_poses=3, pose_reuse=0, early_term=False, timeout=None, plan_grasp=False, plan_arm=False, gripper_type=None, gripper_scale=None, optimizer='L-BFGS-B', debug=0, render=False, log_dir=None, connect_path=False):
        '''
        Main planning function
        Input:
            budget: max # simulations allowed
            max_grippers: max # grippers allowed to use
        Output:
            tree: disassembly tree including all disassembly attempts
        '''
        self.t_start = time()
        self.connect_path = connect_path
        solution_found = False
        self.stop_msg = None
        self._timing = defaultdict(float)
        self._timing_counts = defaultdict(int)
        assert budget is not None or timeout is not None

        self._reset()

        self.n_eval = 0
        G0 = self.parts.copy()
        tree = nx.DiGraph()
        action = getattr(settings, 'no_stable_pose_action', 'exit')
        # Compute the unverified trimesh stable-pose candidate set once. Used
        # directly as the seed in the 'skip' branch, AND as a fallback for the
        # interactive picker on the non-'skip' branches when the gravity
        # precheck rejects most/all candidates (otherwise the picker has
        # nothing to show and silently no-ops).
        G0_mesh = get_combined_mesh(self.assembly_dir, G0)
        trimesh_candidates = get_stable_poses(G0_mesh, max_num=max_poses)
        _per_pose = None
        if action == 'skip' and self.base_part is None:
            initial_poses = list(trimesh_candidates)
            if not initial_poses:
                initial_poses = [translation_pose_to_ground(G0_mesh)]
                print("[planner.base.plan] no trimesh stable pose for full assembly; "
                      "using translation-only ground-lift fallback "
                      "(settings.no_stable_pose_action='skip')")
            else:
                print(f"[planner.base.plan] skipping initial stable-pose gravity check "
                      f"(settings.no_stable_pose_action='skip'); seeded {len(initial_poses)} "
                      f"trimesh stable pose(s) on root node")
            observed_fallen = frozenset()
        else:
            initial_poses, observed_fallen, _per_pose = self._initial_stable_poses(G0, max_poses, log_dir=log_dir)
            # When the precheck rejected every candidate AND observed parts
            # falling, render one PNG per attempted pose (precheck_unstable_<i>.png),
            # each rendered in that pose's orientation with that pose's fallen
            # parts highlighted. Single-file overwrite was hiding the fact that
            # the CLI reported per-pose fallen sets but only one image was
            # surfacing. Gated by render_sequence so a global render-off run
            # doesn't produce the diagnostic PNGs either; the textual ID list
            # still prints so the user knows which parts fell.
            if not initial_poses and observed_fallen:
                print(f"[planner.base.plan] precheck observed {len(observed_fallen)} "
                      f"part(s) falling: {sorted(observed_fallen)}")
                if getattr(settings, 'render_sequence', True):
                    from plan_sequence.planner._renders import render_unstable_parts
                    from pathlib import Path as _Path
                    _save_dir = _Path(log_dir) if log_dir else _Path('/tmp')
                    for _entry in _per_pose:
                        if _entry['success'] or not _entry['fallen']:
                            continue
                        _idx = _entry['pose_idx']
                        _png = render_unstable_parts(
                            self.assembly_dir, G0, _entry['fallen'],
                            save_path=_save_dir / f'precheck_unstable_{_idx:02d}.png',
                            pose=_entry['pose'],
                        )
                        if _png is not None:
                            print(f"[planner.base.plan] unstable-parts visualisation pose "
                                  f"{_idx}: {_png}  (fallen={_entry['fallen']})")
        # When settings.interactive_initial_pose is True (and we're on a tty),
        # let the user pick which stable pose seeds the search. The fallback
        # list ensures the picker has options to show even when the gravity
        # precheck filtered everything out under 'continue' / 'ignore_unstable'.
        initial_poses = self._pick_initial_pose_interactive(
            G0, initial_poses, log_dir=log_dir,
            fallback_candidates=trimesh_candidates,
            per_pose=_per_pose,
        )
        tree.add_node(tuple(G0), n_eval=0, n_gripper=1, poses=initial_poses)

        if self.base_part is None and not initial_poses and action != 'skip':
            if action == 'exit':
                self.stop_msg = 'no self-stable initial pose'
                print('[planner.base.plan] aborting: no self-stable initial pose found '
                      "(settings.no_stable_pose_action='exit')")

                # Persist a precheck-stability failures.json so downstream
                # feedback (Feedback.generate_failure_feedback) can still
                # surface "these parts fall" diagnostics even though no
                # part-removal was attempted. Pairs with the
                # precheck_unstable_XX.png files written above.
                if log_dir is not None and observed_fallen:
                    try:
                        import json as _json
                        import os as _os
                        fallen_sorted = sorted(observed_fallen)
                        subject = fallen_sorted[0]
                        evidence = None
                        if _per_pose:
                            for _e in _per_pose:
                                if _e.get('success') or not _e.get('fallen'):
                                    continue
                                png = f"precheck_unstable_{_e['pose_idx']:02d}.png"
                                if _os.path.exists(_os.path.join(log_dir, png)):
                                    evidence = png
                                    break
                        payload = {
                            'depth': 0,
                            'deepest_nodes': [list(G0)],
                            'failures': [{
                                'child_part': subject,
                                'fail_reason': 'stability',
                                'evidence_image': evidence,
                                'unstable_parts': fallen_sorted,
                            }],
                        }
                        with open(_os.path.join(log_dir, 'failures.json'), 'w') as fp:
                            _json.dump(payload, fp, indent=2)
                        print(f"[planner.base.plan] wrote precheck-stability "
                              f"failures.json: {len(fallen_sorted)} unstable "
                              f"part(s), evidence={evidence}")
                    except Exception as _e:
                        print(f"[planner.base.plan] failed to persist precheck "
                              f"failures.json: {_e}")
                return tree
            elif action == 'ignore_unstable':
                self._ignored_unstable_parts = frozenset(observed_fallen)
                print(f"[planner.base.plan] no self-stable initial pose; "
                      f"ignoring {sorted(self._ignored_unstable_parts)} in all future stability checks "
                      f"(settings.no_stable_pose_action='ignore_unstable')")
            else:  # 'continue' or anything unrecognised
                print(f"[planner.base.plan] no self-stable initial pose; continuing with no seed "
                      f"(settings.no_stable_pose_action={action!r})")

        if plan_arm:
            grasp_planner = GraspArmPlanner(self.asset_folder, self.assembly_dir, gripper_type, gripper_scale)
        elif plan_grasp:
            grasp_planner = GraspPlanner(self.asset_folder, self.assembly_dir, gripper_type, gripper_scale)
        else:
            grasp_planner = None

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
                
                if self.base_part is not None:
                    poses = [None]
                else:
                    poses = tree.nodes[tuple(G)]['poses'][:pose_reuse]
                    G_mesh = get_combined_mesh(self.assembly_dir, G)
                    _t0 = time()
                    fresh = get_stable_poses(G_mesh, max_num=max_poses - pose_reuse)
                    _dt_pose = time() - _t0
                    self._timing['stable_pose'] += _dt_pose
                    self._timing_counts['stable_pose'] += 1
                    # Default behaviour: when multiple stable poses exist for
                    # this subassembly, prefer the one closest (by rotation
                    # angle) to the previous step's pose. Keeps the operator
                    # from re-orienting the assembly when an equivalent pose
                    # is available. When no parent pose is known (root, or
                    # first-time visit before any feasible edge) the trimesh
                    # probability ordering is preserved.
                    parent_pose = self._parent_pose_for(tree, G)
                    if parent_pose is not None and len(fresh) > 1:
                        fresh = self._sort_poses_by_proximity(fresh, parent_pose)
                    poses.extend(fresh)
                    if debug > 1:
                        print(f'[planner.base.plan] stable_pose: {_dt_pose:.3f}s  n_poses={len(poses)}')
                    if len(poses) == 0:
                        # Deterministic ground-lift fallback so parts never clip
                        # the ground when trimesh can't find a stable pose for
                        # this subassembly. Replaces the previous `[None]`
                        # (which left the assembly straddling z=0).
                        poses = [translation_pose_to_ground(G_mesh)]

                for p in self.seq_generator.generate_candidate_part(G): # NOTE: maybe can specify which parts to exclude
                    G_prime = G.copy()
                    G_prime.remove(p)

                    if tree.has_edge(tuple(G), tuple(G_prime)): continue

                    for pose in poses:

                        sim_timeout = None if timeout is None else timeout - (time() - self.t_start) # allocate for this step of simulation
                        if sim_timeout is not None and sim_timeout < 0:
                            break

                        sim_info = self._simulate(p, G_prime, parts_removed, pose, max_grippers=max_grippers, timeout=sim_timeout,
                            grasp_planner=grasp_planner, optimizer=optimizer, debug=debug - 1, render=render, log_dir=log_dir)
                        self.n_eval += 1
                        self._update_tree(tree, G, G_prime, self.n_eval, sim_info)

                        if sim_info['feasible']:
                            if len(G_prime) == 2:
                                solution_found = True
                            break

                    if debug > 0:
                        print(f'[planner.base.plan] add edge: ({G}, {G_prime}), feasible: {sim_info["feasible"]}')
                        print(f'[planner.base.plan] progress: {self.n_eval}/{budget} evaluations')
                    if debug > 1:
                        t_elapsed = time() - self.t_start
                        _timing_str = '  '.join(
                            f'{k}: {self._timing[k]:.1f}s ({100*self._timing[k]/t_elapsed:.0f}%)'
                            for k in ['path_finding', 'stability_check', 'stable_pose', 'grasp_planning']
                            if k in self._timing
                        )
                        print(f'[planner.base.plan] elapsed: {t_elapsed:.1f}s  [{_timing_str}]')

                    if debug > 1:
                        self.plot_tree(tree)
                    break

                if log_dir is not None:
                    stats = self.get_stats(tree)
                    self.log(tree, stats, log_dir)
            
            self._expand_leaf(tree, max_poses, pose_reuse, grasp_planner, optimizer, debug, render)
            self.plot_tree(tree)

        except (Exception, KeyboardInterrupt) as e:
            if type(e) == KeyboardInterrupt:
                self.stop_msg = 'interrupt'
            else:
                self.stop_msg = 'exception'
            print(e, f'from {self.assembly_dir}')
            print(traceback.format_exc())

        assert self.stop_msg is not None, '[planner.base.plan] bug: unexpectedly stopped'
        if debug > 0:
            print(f'[planner.base.plan] stopped: {self.stop_msg}')
            self._print_timing_summary()

        return tree

    def _print_timing_summary(self):
        t_total = time() - self.t_start
        print(f'  timing summary (wall-clock total: {t_total:.2f}s):')
        order = ['path_finding', 'stability_check', 'stable_pose', 'grasp_planning']
        for key in order + [k for k in self._timing if k not in order]:
            if key not in self._timing:
                continue
            t = self._timing[key]
            n = self._timing_counts[key]
            avg = t / n if n > 0 else 0.0
            print(f'  {key:<20} total {t:8.2f}s  avg {avg:7.3f}s  ({n} calls)')

        iter_timings = getattr(self, '_iter_timings', None)
        if iter_timings:
            print(f'  per-iteration breakdown ({len(iter_timings)} iterations):')
            print(f'    {"iter":>4}  {"parents":>7}  {"sims":>5}  {"duration":>9}  {"cum":>8}')
            cum = 0.0
            for idx, n_parents, n_received, dt in iter_timings:
                cum += dt
                print(f'    {idx:>4}  {n_parents:>7}  {n_received:>5}  {dt:>8.2f}s  {cum:>7.2f}s')

    def _reset(self):
        pass

    def _initial_stable_poses(self, parts, max_poses, log_dir=None):
        '''
        Pre-flight pose seed for the full assembly. When base_part is set the
        sim pose is always identity, so this is a no-op. Otherwise compute
        trimesh stable poses and keep only those for which the assembly is
        self-supporting under gravity (no parts would need fixing).
        When `log_dir` is provided, pyvista debug screenshots of each candidate
        pose are written to `<log_dir>/precheck_poses/`.

        Returns (poses, observed_fallen, per_pose).
          - `observed_fallen` is a frozenset of part IDs that were seen falling
            in at least one candidate pose check — the caller uses it when
            settings.no_stable_pose_action == 'ignore_unstable'.
          - `per_pose` is the per-pose result list from
            find_stable_initial_poses; used by callers to render one
            diagnostic image per candidate (precheck_unstable_<i>.png).
        '''
        if self.base_part is not None:
            return [], frozenset(), []
        debug_dir = os.path.join(log_dir, 'precheck_poses') if log_dir is not None else None
        # settings.debug_stability turns on per-pose gravity-sim replays.
        # Paths under <log_dir>/precheck_stability/ — produced by reusing the
        # sim's already-populated state history, so no re-simulation.
        stability_debug_dir = None
        if log_dir is not None and getattr(settings, 'debug_stability', False):
            stability_debug_dir = os.path.join(log_dir, 'precheck_stability')
            os.makedirs(stability_debug_dir, exist_ok=True)
            print(f'[planner] precheck stability-debug GIFs will land in {stability_debug_dir}')
        poses, observed_fallen, per_pose = find_stable_initial_poses(
            self.asset_folder, self.assembly_dir, parts,
            max_poses=max_poses, save_sdf=self.save_sdf, allow_gap=self.allow_gap,
            num_proc=self.num_proc, first_only=False, debug_dir=debug_dir,
            stability_debug_dir=stability_debug_dir,
        )
        if not poses:
            print(f'[planner] WARNING: no self-stable initial pose found for full assembly — parts would fall without fixing (observed_fallen={sorted(observed_fallen)})')
        else:
            print(f'[planner] precheck: seeded {len(poses)} self-stable initial pose(s) on root node')
        return poses, observed_fallen, per_pose

    def wh_select_node(self, tree):
        raise NotImplementedError

    def _expand_leaf(self, tree, max_poses, pose_reuse, grasp_planner, optimizer, debug, render):
        for node in tree.nodes:
            assert len(node) >= 2

        node_expand_list = []
        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 2 and node_info['n_gripper'] is not None:
                node_expand_list.append(node)

        G0 = self.parts.copy()

        # Use a monotonic counter past any existing n_eval. Parallel DFA shares
        # n_eval across a whole batch of siblings, so multiple 2-part leaves can
        # carry identical n_eval; when several of them expand to the same 1-part
        # child the strict `<` assertion in _update_tree would otherwise fail.
        next_n_eval = max((tree.nodes[n]['n_eval'] for n in tree.nodes), default=0) + 1

        for node in node_expand_list:
            part_a, part_b = node
            mass_a, mass_b = self.part_mass[part_a], self.part_mass[part_b]
            # Primary role assignment: lighter part moves, heavier stays fixed.
            primary_fix, primary_move = (part_a, part_b) if mass_a > mass_b else (part_b, part_a)
            # Try both role assignments before giving up. The mass-based pick is
            # usually right (less inertia on the gripper actuating the move) but
            # for some geometries the only feasible separation direction is
            # blocked under the primary assignment (e.g. settings.filter_below_ground
            # rejects every pose because the lighter part naturally extracts
            # downward in all stable orientations). Swapping roles is cheap and
            # catches those cases. Skip the swap when one of the two is the
            # base_part (it must remain fixed by definition).
            role_pairs = [(primary_fix, primary_move)]
            if self.base_part is None:
                role_pairs.append((primary_move, primary_fix))
            else:
                # base_part must be fixed; flip to enforce it if the mass heuristic
                # chose otherwise. No further swap to try.
                if primary_move == self.base_part:
                    role_pairs = [(primary_move, primary_fix)]

            parts_removed = [part for part in G0 if part != part_a and part != part_b]
            poses = tree.nodes[tuple(node)]['poses'][:pose_reuse]
            node_mesh = get_combined_mesh(self.assembly_dir, node)
            fresh = get_stable_poses(node_mesh, max_num=max_poses - pose_reuse)
            # Same proximity-prefer pass as in plan(): when the parent step's
            # pose is known, order fresh stable poses closest-first.
            parent_pose = self._parent_pose_for(tree, node)
            if parent_pose is not None and len(fresh) > 1:
                fresh = self._sort_poses_by_proximity(fresh, parent_pose)
            poses.extend(fresh)
            if self.base_part is not None:
                poses = [None]
            elif len(poses) == 0:
                poses = [translation_pose_to_ground(node_mesh)]

            matched = False
            for part_fix, part_move in role_pairs:
                for pose in poses:
                    sim_info = self._simulate(part_move, [part_fix], parts_removed, pose=pose, max_grippers=2, grasp_planner=grasp_planner, optimizer=optimizer, debug=debug - 1, render=render)
                    if sim_info['feasible']:
                        SequencePlanner._update_tree(self, tree, list(node), [part_fix], next_n_eval, sim_info)
                        next_n_eval += 1
                        matched = True
                        break
                if matched:
                    break

    @staticmethod
    def plot_tree(tree, save_path=None):
        from networkx.drawing.nx_agraph import graphviz_layout
        node_colors = ['g' if tree.nodes[node]['n_gripper'] is not None else 'r' for node in tree.nodes]
        edge_colors = ['g' if tree.edges[edge]['sim_info']['feasible'] else 'r' for edge in tree.edges]
        edge_labels = {edge: ','.join([str(x) for x in set(edge[0]) - set(edge[1])]) for edge in tree.edges}
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def plot_tree_with_budget(tree, budget, save_path=None):
        if budget is None: return SequencePlanner.plot_tree(tree, save_path=save_path)

        from networkx.drawing.nx_agraph import graphviz_layout
        budget_tree = nx.DiGraph()
        # for node in tree.nodes:
        #     node_info = tree.nodes[node]
        #     if node_info['n_eval'] <= budget:
        #         budget_tree.add_node(node)
        for edge in tree.edges:
            edge_info = tree.edges[edge]
            if edge_info['n_eval'] <= budget:
                budget_tree.add_edge(*edge, feasible=edge_info['sim_info']['feasible'])
                
        # node_colors = ['g' if budget_tree.nodes[node]['feasible'] else 'r' for node in budget_tree.nodes]
        edge_colors = ['g' if budget_tree.edges[edge]['feasible'] else 'r' for edge in budget_tree.edges]
        edge_labels = {edge: ','.join([str(x) for x in set(edge[0]) - set(edge[1])]) for edge in budget_tree.edges}
        pos = graphviz_layout(budget_tree, prog='dot')
        nx.draw(budget_tree, pos, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(budget_tree, pos, edge_labels=edge_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def find_sequence(tree):

        # find leaf node
        leaf_node = None
        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 1 and node_info['n_gripper'] is not None:
                leaf_node = node
                break
        else:
            return None

        # find sequence in tree from bottom to top
        sequence = []
        node = leaf_node
        while tree.in_degree(node) > 0:
            node_info = tree.nodes[node]
            for parent_node in tree.predecessors(node):
                parent_node_info = tree.nodes[parent_node]
                if parent_node_info['n_gripper'] <= node_info['n_gripper']:
                    part_move = tree.edges[parent_node, node]['sim_info']['part_move']
                    sequence.insert(0, part_move)
                    node = parent_node
                    break
        return sequence

    @staticmethod
    def find_partial_sequence(tree):
        '''
        Best-effort sequence for a run that got stuck before fully
        disassembling. Returns the removal order along the deepest reachable
        feasible path from the root (the smallest subassembly that was reached
        through a chain of feasible edges). Returns [] when no feasible edge
        exists at all (no progress past the root).

        Unlike find_sequence (which requires a size-1 leaf), this accepts any
        feasible node, so the result may be a prefix of a full sequence.
        '''
        # Deepest feasible node = smallest reachable subassembly. Break ties by
        # fewer grippers, then by earlier discovery (n_eval).
        best = None
        for node in tree.nodes:
            info = tree.nodes[node]
            if info['n_gripper'] is None:
                continue
            if best is None:
                best = node
                continue
            best_info = tree.nodes[best]
            key = (len(node), info['n_gripper'], info['n_eval'])
            best_key = (len(best), best_info['n_gripper'], best_info['n_eval'])
            if key < best_key:
                best = node

        if best is None or tree.in_degree(best) == 0:
            return []  # no feasible progress past the root

        # Walk up to the root along feasible edges, collecting removed parts.
        sequence = []
        node = best
        while tree.in_degree(node) > 0:
            node_info = tree.nodes[node]
            for parent_node in tree.predecessors(node):
                parent_info = tree.nodes[parent_node]
                if parent_info['n_gripper'] is not None and parent_info['n_gripper'] <= node_info['n_gripper']:
                    part_move = tree.edges[parent_node, node]['sim_info']['part_move']
                    sequence.insert(0, part_move)
                    node = parent_node
                    break
            else:
                break  # no feasible parent found (shouldn't happen for a reachable node)
        return sequence

    @staticmethod
    def find_deepest_reached_nodes(tree):
        '''
        All feasibly-reached nodes (n_gripper is not None) of minimum length,
        i.e. the deepest reached nodes in the search tree (most parts already
        disassembled). Returns a list of node tuples (may have >1 entries on
        ties). Returns [] when no feasible progress past the root exists.
        '''
        feasible = [(node, tree.nodes[node]) for node in tree.nodes
                    if tree.nodes[node].get('n_gripper') is not None]
        if not feasible:
            return []
        min_len = min(len(n) for n, _ in feasible)
        return [n for n, _ in feasible if len(n) == min_len]

    @staticmethod
    def check_success(tree):

        success = False
        n_eval = None # min n_eval
        n_gripper = None # min n_gripper

        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 1 and node_info['n_gripper'] is not None:
                success = True

                if n_eval is None:
                    n_eval = node_info['n_eval']
                else:
                    n_eval = min(n_eval, node_info['n_eval'])

                if n_gripper is None:
                    n_gripper = node_info['n_gripper']
                else:
                    n_gripper = min(n_gripper, node_info['n_gripper'])

        return success, n_eval, n_gripper

    @staticmethod
    def get_stats(tree):
        success, n_eval, n_gripper = SequencePlanner.check_success(tree)
        if success:
            sequence = SequencePlanner.find_sequence(tree)
            partial = False
        else:
            # Stuck before full disassembly — return the deepest feasible
            # prefix so the caller can still report/render partial progress.
            sequence = SequencePlanner.find_partial_sequence(tree)
            partial = bool(sequence)
        return {
            'success': success,
            'partial': partial,
            'n_eval': n_eval,
            'n_gripper': n_gripper,
            'sequence': [x for x in sequence] if sequence else None,
        }

    def log(self, tree, stats, log_dir, plot=False):
        '''
        Log planned disassembly sequence and gripper statistics
        '''
        t_plan = time() - self.t_start # NOTE: a bit hacky
        stats['time'] = round(t_plan, 2)
        stats['total_n_eval'] = self.n_eval
        stats['stop_msg'] = self.stop_msg

        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'tree.pkl'), 'wb') as fp:
            pickle.dump(tree, fp)
        with open(os.path.join(log_dir, 'stats.json'), 'w') as fp:
            json.dump(stats, fp, cls=NumpyEncoder)
        if plot:
            self.plot_tree(tree, save_path=os.path.join(log_dir, 'tree.png'))

    def render(self, sequence, tree, record_dir=None):
        '''
        Render planned disassembly sequence
        '''
        parts_assembled = self.parts.copy()
        parts_removed = []

        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        for i, part_move in enumerate(sequence):
            parts_rest = parts_assembled.copy()
            parts_rest.remove(part_move)

            sim_info = tree.edges[tuple(parts_assembled), tuple(parts_rest)]['sim_info']
            assert part_move == sim_info['part_move']
            parts_fix = sim_info['parts_fix']
            parts_free = [part_i for part_i in parts_rest if part_i not in parts_fix] + [part_move]
            action = np.array(sim_info['action'])
            pose = np.array(sim_info['pose']) if sim_info['pose'] is not None else None

            if record_dir is not None:
                record_path = os.path.join(record_dir, f'{i}_{part_move}.gif')
            else:
                record_path = None

            path_planner = MultiPartPathPlanner(self.asset_folder, self.assembly_dir, parts_rest, part_move, parts_removed=parts_removed, pose=pose, save_sdf=self.save_sdf)
            # path_planner.check_success(action)
            success, path = path_planner.plan_path(action, rotation=True, connect_path=getattr(self, 'connect_path', False))

            body_color_dict = get_body_color_dict(parts_fix, parts_free) # visualize fixes
            path_planner.sim.set_body_color_map(body_color_dict)
            path_planner.render(path=path, record_path=record_path)

            parts_assembled = parts_rest
            parts_removed.append(part_move)
