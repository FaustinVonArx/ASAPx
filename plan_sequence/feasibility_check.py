import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
from itertools import combinations
from time import time

from utils.renderer import SimRenderer
from utils.parallel import parallel_execute
from plan_sequence.physics_planner import (
    MultiPartPathPlanner, MultiPartStabilityPlanner, MultiPartNoForceStabilityPlanner,
    get_contact_graph, CONTACT_EPS,
    check_tool as _check_tool_physics,
)


def get_R3_actions():
    actions = [
        np.array([0, 0, 1]), # +Z
        np.array([0, 0, -1]), # -Z
        np.array([1, 0, 0]), # +X
        np.array([-1, 0, 0]), # -X
        np.array([0, 1, 0]), # +Y
        np.array([0, -1, 0]), # -Y   
    ]
    return actions

def check_tool(asset_folder, assembly_dir, parts_fix, part_move, tools,
               asset_folder_bfs=None, output_dir=None, show=False, debug=0,
               diagnostics=None, failure_record_dir=None):
    '''
    Tool-feasibility wrapper used by sequence planning: check whether any of the
    candidate `tools` can geometrically remove `part_move` from the sub-assembly
    formed by `parts_fix + [part_move]`. Delegates to physics_planner.check_tool.

    diagnostics: optional mutable dict; when supplied, populated with
        {'subkind': 'collision'|'access'|None, 'colliding_parts': [...], 'tried_tools': [...]}.

    Returns:
        dict {'tool_id', 'tool_mesh', 'inverted'} on success, or None on failure.
    '''
    result = _check_tool_physics(
        asset_folder=asset_folder,
        assembly_dir=assembly_dir,
        parts_fix=parts_fix,
        part_move=part_move,
        tools=tools,
        asset_folder_bfs=asset_folder_bfs,
        output_dir=output_dir,
        show=show,
        verbose=debug > 0,
        diagnostics=diagnostics,
        failure_record_dir=failure_record_dir,
    )
    if debug > 0:
        if result is None:
            print(f'[check_tool] no feasible tool for part_move={part_move}, parts_fix={parts_fix}')
        else:
            print(f'[check_tool] feasible tool={result["tool_id"]} (inverted={result["inverted"]}) for part_move={part_move}')
    return result

def check_assemblable(asset_folder, assembly_dir, parts_fix, part_move, pose=None, save_sdf=False, debug=0, render=False, return_path=False, optimize_path=False, min_sep=None, get_dof=False, diagnostics=None):
    '''
    Check if certain parts are disassemblable

    diagnostics: optional mutable dict; when supplied, it is populated with
        {'directions': [{'action': [..], 'success': bool, 'path_len': int}, ...]}
        capturing per-axis probe results that would otherwise be discarded.
    '''
    planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move, pose=pose, save_sdf=save_sdf)

    dof = planner.compute_dof() if get_dof else None

    # Probe along assembly-local axes: rotate the world-frame 6 unit actions by pose.
    R = pose[:3, :3] if pose is not None else np.eye(3)
    actions = [R @ a for a in get_R3_actions()]
    # Try actions whose world-z component is most positive first; ground-directed
    # actions (negative world-z) are tried last so a floor-blocked sim stalls only
    # after other directions have already had a chance.
    actions.sort(key=lambda a: -float(a[2]))
    best_action = None
    best_path = None
    best_path_len = np.inf
    if diagnostics is not None:
        diagnostics['directions'] = []
    for action in actions:
        success, path = planner.check_success(action, return_path=True, min_sep=None if optimize_path else min_sep)
        if diagnostics is not None:
            diagnostics['directions'].append({
                'action': [float(x) for x in action],
                'success': bool(success),
                'path_len': int(len(path)) if path is not None else 0,
            })
        if debug > 0:
            print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action: {action}, path_len: {len(path)}')
            if render:
                SimRenderer().replay(planner.sim)
        if success:
            if len(path) < best_path_len:
                best_path_len = len(path)
                best_path = path
                best_action = action

    if best_path is not None:
        best_path = np.array(best_path)
        if optimize_path: # optimize action based on the path found
            best_dirs = best_path[1:, :3] - best_path[0, :3]
            opt_action = (best_dirs / np.linalg.norm(best_dirs, axis=1)[:, None]).mean(axis=0)
            opt_action = opt_action / np.linalg.norm(opt_action)
            success, opt_path = planner.check_success(opt_action, return_path=True, min_sep=min_sep)
            if debug > 0:
                print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action (optimized): {opt_action}, path_len (optimized): {len(opt_path)}')
                if render:
                    SimRenderer().replay(planner.sim)
            if success:
                best_path_len = len(opt_path)
                best_path = opt_path
                best_action = opt_action
            else: # just in case, plan again with min_sep
                success, best_path = planner.check_success(best_action, return_path=True, min_sep=min_sep)
                assert success
        best_path = np.array(best_path)

    if return_path and get_dof:
        return best_action, best_path, dof
    elif return_path:
        return best_action, best_path
    elif get_dof:
        return best_action, dof
    else:
        return best_action


def check_all_connection_assemblable(asset_folder, assembly_dir, parts=None, contact_eps=CONTACT_EPS, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if all connected pairs of parts are disassemblable
    '''
    G = get_contact_graph(asset_folder, assembly_dir, parts, contact_eps=contact_eps, save_sdf=save_sdf)

    worker_args = []
    for pair in G.edges:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc, show_progress=debug > 0, desc='check_all_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_all_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_given_connection_assemblable(asset_folder, assembly_dir, part_pairs, bidirection=False, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if given connected pairs of parts are disassemblable
    '''
    worker_args = []
    for pair in part_pairs:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])
        if bidirection:
            worker_args.append([asset_folder, assembly_dir, [part_b], part_a, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc, show_progress=debug > 0, desc='check_given_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_given_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_stable_noforce(asset_folder, assembly_dir, parts, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False):
    '''
    Check if stable without any external force
    '''
    planner = MultiPartNoForceStabilityPlanner(asset_folder, assembly_dir, parts, save_sdf=save_sdf, allow_gap=allow_gap)
    
    success, G = planner.check_success(timeout=timeout)
    if debug > 0:
        print(f'[check_stable_noforce] success: {success}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, G


def check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False, record_path=None, ignore_unstable=()):
    '''
    Check if gravitationally stable for a given fixed part
    '''
    planner = MultiPartStabilityPlanner(asset_folder, assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, allow_gap=allow_gap, ignore_unstable=ignore_unstable)

    success, parts_fall = planner.check_success(timeout=timeout, record_path=record_path)
    if debug > 0:
        print(f'[check_stable] success: {success}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, parts_fall


def get_stable_plan_1pose_serial(asset_folder, assembly_dir, parts, base_part, pose, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False, return_count=False, log_dir=None, step_label=None, ignore_unstable=(), diagnostics=None):
    '''
    Get all gravitationally stable plans given 1 pose through serial greedy search

    diagnostics: optional mutable dict; when supplied, populated with
        {'unstable_parts': [...]} listing the union of all parts_fall observed
        across the greedy iterations (only meaningful on failure).
    '''
    t_start = time()
    count = 0

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))
    parts_fix = [] if base_part is None else [base_part]
    fell_seen = []

    while True:

        parts_move = parts.copy()
        for part_fix in parts_fix:
            parts_move.remove(part_fix)

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                if diagnostics is not None:
                    diagnostics['unstable_parts'] = list(dict.fromkeys(fell_seen))
                    diagnostics['timed_out'] = True
                if return_count:
                    return None, count
                else:
                    return None
            t_start = time()

        if log_dir is not None and step_label is not None:
            record_path = os.path.join(log_dir, 'stability_debug', f'{step_label}_fix{len(parts_fix)}.gif')
        else:
            record_path = None

        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render, record_path=record_path, ignore_unstable=ignore_unstable)
        count += 1

        if debug > 0:
            print(f'[get_stable_plan_1pose_serial] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')

        if success:
            break
        else:
            if parts_fall is None:
                if diagnostics is not None:
                    diagnostics['unstable_parts'] = list(dict.fromkeys(fell_seen))
                    diagnostics['timed_out'] = True
                if return_count:
                    return None, count # timeout
                else:
                    return None
            fell_seen.extend(parts_fall)
            parts_fix.extend(parts_fall)

        if len(parts_fix) > max_fix:
            if diagnostics is not None:
                diagnostics['unstable_parts'] = list(dict.fromkeys(fell_seen))
                diagnostics['max_fix_exceeded'] = True
            if return_count:
                return None, count # failed
            else:
                return None

    if base_part is not None:
        parts_fix.remove(base_part)

    if return_count:
        return parts_fix, count
    else:
        return parts_fix


def get_stable_plan_1pose_parallel(asset_folder, assembly_dir, parts, base_part, pose=None, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, num_proc=1, debug=0, render=False):
    '''
    Get all gravitationally stable plans given 1 pose through parallel greedy search
    '''
    t_start = time()

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))

    if pose is not None:
        parts_fix = [] if base_part is None else [base_part]
        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts, pose, save_sdf, timeout, allow_gap, debug, render) # check if stable without any grippers
        if debug > 0:
            print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: 0, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts}')
        if success:
            return []
        else:
            if parts_fall is None:
                return None # timeout

    if base_part is None:
        parts_fix_list = [[part_fix] for part_fix in parts]
    else:
        parts_fix_list = [[part_fix, base_part] for part_fix in parts if part_fix != base_part]
    
    while True:
        success_any = False

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                return None
            t_start = time()

        worker_args = []
        for parts_fix in parts_fix_list:
            if len(parts_fix) > max_fix: continue
            parts_move = parts.copy()
            for part_fix in parts_fix:
                parts_move.remove(part_fix)
            worker_args.append([asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render])

        if len(worker_args) == 0:
            return None # failed

        for (success, parts_fall), args in parallel_execute(check_stable, worker_args, num_proc, show_progress=debug > 0, desc='get_stable_plan_1pose_parallel', return_args=True):
            parts_fix, parts_move = args[2], args[3]
            if debug > 0:
                print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
            if success:
                success_any = True
            else:
                if parts_fall is None:
                    return None # timeout
                index = parts_fix_list.index(parts_fix)
                parts_fix_list[index].extend(parts_fall)
            if timeout is not None and time() - t_start > timeout:
                return None

        if success_any:
            break

    parts_fix_list = [parts_fix for parts_fix in parts_fix_list if len(parts_fix) <= max_fix]
    for parts_fix in parts_fix_list:
        if base_part is not None:
            parts_fix.remove(base_part)
    parts_fix_list = sorted(parts_fix_list, key=lambda x: len(x))
    return parts_fix_list
