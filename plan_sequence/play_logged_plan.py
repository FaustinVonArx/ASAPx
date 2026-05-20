import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import traceback
import shutil
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from plan_sequence.sim_string import get_body_color_dict
from plan_sequence.physics_planner import MultiPartPathPlanner, MultiPartStabilityPlanner
from plan_sequence.planner.base import SequencePlanner
from plan_robot.render_grasp import render_path_with_grasp
from plan_robot.render_grasp_arm import render_path_with_grasp_and_arm
from plan_robot.geometry import load_part_meshes, load_gripper_meshes, load_arm_meshes, save_meshes
from assets.save import clear_saved_sdfs, save_path_all_objects
from assets.load import load_part_ids


def _fit_camera_to_path(path_planner, path, camera_pos, camera_lookat):
    """Reposition the sim camera to frame the start and end of the path.

    Uses only path[0] (assembled position) and path[-1] (ground resting position)
    to define the scene extent — ignoring intermediate disassembly positions that
    may fly far off-screen.  The original viewing direction is preserved; only the
    lookat and distance are adjusted.
    """
    if len(path) < 2:
        return

    start = np.array(path[0][:3])
    end   = np.array(path[-1][:3])

    center = (start + end) / 2
    span   = float(np.linalg.norm(end - start))
    if span < 1e-6:
        return

    if camera_pos is not None and camera_lookat is not None:
        cam_dir = np.array(camera_pos) - np.array(camera_lookat)
        norm = np.linalg.norm(cam_dir)
        cam_dir = cam_dir / norm if norm > 1e-6 else np.array([1., -1., 1.]) / np.sqrt(3)
    else:
        cam_dir = np.array([1., -1., 1.]) / np.sqrt(3)

    # Keep the same camera-to-lookat distance as the original, scaled by how much
    # further the ground position is from the assembly centre.
    if camera_pos is not None and camera_lookat is not None:
        orig_dist = float(np.linalg.norm(np.array(camera_pos) - np.array(camera_lookat)))
    else:
        orig_dist = span
    dist = max(orig_dist, span * 0.8)

    path_planner.sim.viewer_options.camera_lookat = list(center)
    path_planner.sim.viewer_options.camera_pos    = list(center + cam_dir * dist)


def interpolate_path(states):
    interpolated_path = []

    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        interpolated_path.append(current_state)

        if len(current_state) == 3:
            average_state = (current_state + next_state) / 2
            interpolated_path.append(average_state)
        elif len(current_state) == 6:
            current_pos, current_euler = current_state[:3], current_state[3:]
            next_pos, next_euler = next_state[:3], next_state[3:]
            average_pos = (current_pos + next_pos) / 2
            rotations = R.from_euler('xyz', [current_euler, next_euler])
            slerp = Slerp([0, 1], rotations)
            average_euler = slerp([0.5]).as_euler('xyz')[0]
            average_state = np.concatenate([average_pos, average_euler])
            interpolated_path.append(average_state)
        else:
            raise NotImplementedError

    interpolated_path.append(states[-1])

    return interpolated_path


def _render_step_worker(asset_folder, assembly_dir, step_arg, result_paths, options):
    """Render one disassembly step in isolation. Top-level so it is picklable and
    can be dispatched across multiprocessing workers. All cross-step state must
    be pre-resolved by the caller and passed via ``step_arg`` — this worker does
    not touch the planning tree or any accumulating list, so concurrent calls
    for different steps share no mutable Python state."""
    import time as _time
    _t0 = _time.time()
    print(f'[play_logged_plan] pid={os.getpid()} step {step_arg["i"]} ({step_arg["part_move"]}) START', flush=True)
    i = step_arg['i']
    part_move = step_arg['part_move']
    parts_rest = step_arg['parts_rest']
    parts_removed = step_arg['parts_removed']
    sim_info = step_arg['sim_info']
    tool_mesh = step_arg.get('tool_mesh')

    record_dir = result_paths.get('record_dir')
    pose_dir = result_paths.get('pose_dir')
    part_dir = result_paths.get('part_dir')
    path_dir = result_paths.get('path_dir')
    record_dir_grasp = result_paths.get('record_dir_grasp')
    extra_views = result_paths.get('extra_views') or []

    save_path = options['save_path']
    save_record = options['save_record']
    save_all = options['save_all']
    save_sdf = options['save_sdf']
    make_video = options['make_video']
    reverse = options['reverse']
    show_fix = options['show_fix']
    show_grasp = options['show_grasp']
    show_arm = options['show_arm']
    gripper_type = options['gripper_type']
    gripper_scale = options['gripper_scale']
    optimizer = options['optimizer']
    camera_pos = options['camera_pos']
    camera_lookat = options['camera_lookat']
    connect_path = options['connect_path']
    n_frame = options.get('n_frame', 300)

    action = np.array(sim_info['action'])
    pose = np.array(sim_info['pose']) if sim_info['pose'] is not None else None
    grasps = sim_info['grasp'] if show_grasp else None
    parts_fix = sim_info['parts_fix']
    parts_free = [p for p in parts_rest if parts_fix is None or p not in parts_fix] + [part_move]

    if show_fix:
        body_color_dict = get_body_color_dict(parts_fix, parts_free, parts_moving=[part_move])
    else:
        body_color_dict = get_body_color_dict([], parts_rest + [part_move], parts_moving=[part_move])

    if record_dir is not None:
        record_path = os.path.join(record_dir, f'{i}_{part_move}.mp4' if make_video else f'{i}_{part_move}.gif')
    else:
        record_path = None

    if save_path or save_record or save_all:
        path_planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_rest, part_move,
            parts_removed=parts_removed, pose=pose, save_sdf=save_sdf,
            camera_pos=camera_pos, camera_lookat=camera_lookat)
        success, path = path_planner.plan_path(action, rotation=True, connect_path=connect_path)
        assert success, f'[play_logged_plan] step {i} part_move {part_move}: path planner failed'

        min_path_len = 300
        while len(path) < min_path_len:
            path = interpolate_path(path)

        if (show_grasp or show_arm) and grasps is not None:
            n_render = min(3, len(grasps))
            random_indices = np.random.choice(len(grasps), n_render, replace=False)
            for idx in random_indices:
                grasp = grasps[idx][0]
                if record_dir_grasp is not None:
                    record_path_grasp = os.path.join(record_dir_grasp, f'{i}_{part_move}_g{idx}.mp4' if make_video else f'{i}_{part_move}_g{idx}.gif')
                else:
                    record_path_grasp = None
                if show_arm:
                    body_matrices = render_path_with_grasp_and_arm(asset_folder, assembly_dir, part_move, parts_rest, parts_removed, pose, path, gripper_type, gripper_scale, grasp, optimizer, camera_lookat, camera_pos,
                        body_color_dict, reverse, save_record or save_all, record_path_grasp, make_video)
                else:
                    body_matrices = render_path_with_grasp(asset_folder, assembly_dir, part_move, parts_rest, parts_removed, pose, path, gripper_type, gripper_scale, grasp, camera_lookat, camera_pos,
                        body_color_dict, reverse, save_record or save_all, record_path_grasp, make_video)

                if path_dir is not None:
                    path_i_dir = os.path.join(path_dir, f'{i}_{part_move}_g{idx}')
                    save_path_all_objects(path_i_dir, body_matrices, n_frame=n_frame)

        if connect_path:
            _fit_camera_to_path(path_planner, path, camera_pos, camera_lookat)

        path_planner.sim.set_body_color_map(body_color_dict)
        if record_path is not None:
            if tool_mesh is not None:
                body_matrices = path_planner.render_with_tool(
                    tool_mesh, path=path, reverse=reverse,
                    record_path=record_path, make_video=make_video,
                    body_color_dict=body_color_dict,
                )
            else:
                body_matrices = path_planner.render(path=path, reverse=reverse, record_path=record_path, make_video=make_video)

            if path_dir is not None:
                path_i_dir = os.path.join(path_dir, f'{i}_{part_move}')
                save_path_all_objects(path_i_dir, body_matrices, n_frame=n_frame)

        for view_record_dir, view_camera_pos, view_camera_lookat in extra_views:
            view_record_path = os.path.join(view_record_dir,
                f'{i}_{part_move}.mp4' if make_video else f'{i}_{part_move}.gif')
            if tool_mesh is not None:
                path_planner.render_with_tool(
                    tool_mesh, path=path, reverse=reverse,
                    record_path=view_record_path, make_video=make_video,
                    camera_pos=list(view_camera_pos), camera_lookat=list(view_camera_lookat),
                    body_color_dict=body_color_dict,
                )
            else:
                if connect_path:
                    _fit_camera_to_path(path_planner, path, view_camera_pos, view_camera_lookat)
                else:
                    path_planner.sim.viewer_options.camera_pos = list(view_camera_pos)
                    path_planner.sim.viewer_options.camera_lookat = list(view_camera_lookat)
                path_planner.render(path=path, reverse=reverse, record_path=view_record_path, make_video=make_video)

    if pose_dir is not None:
        pose_path = os.path.join(pose_dir, f'{i}_{part_move}.npy')
        np.save(pose_path, pose, allow_pickle=True)

    if part_dir is not None:
        part_path = os.path.join(part_dir, f'{i}_{part_move}.json')
        with open(part_path, 'w') as fp:
            json.dump(parts_fix, fp)

    print(f'[play_logged_plan] pid={os.getpid()} step {i} ({part_move}) DONE in {_time.time() - _t0:.1f}s', flush=True)
    return i


def play_logged_plan(asset_folder, assembly_dir, sequence, tree, result_dir, save_mesh, save_pose, save_part, save_path, save_record, save_all,
    reverse=False, show_fix=False, show_grasp=False, show_arm=False, gripper_type=None, gripper_scale=None, optimizer='L-BFGS-B', save_sdf=False, clear_sdf=False, make_video=False, budget=None, camera_pos=None, camera_lookat=None, connect_path=False,
    extra_views=None, tool_meshes_per_step=None, num_proc=1, n_frame=300):
    # extra_views: list of (record_dir, camera_pos, camera_lookat) — each entry
    # re-renders the already-simulated path from a different camera, writing
    # GIFs/MP4s into its record_dir. No additional physics simulation is run.
    #
    # tool_meshes_per_step: optional dict {part_id: trimesh.Trimesh}. When an entry exists
    # for the step's moving part, the tool mesh (already positioned in the part's OBJ frame,
    # e.g. via ToolAnalyzer._apply_tool_geometric) is attached as a fixed child of the
    # moving part and replayed alongside the part trajectory.
    #
    # num_proc: when > 1, render each disassembly step in a separate process. Steps share
    # no mutable state (parts_assembled/parts_removed are pre-resolved per step before
    # dispatch), so they can be rendered fully in parallel. Falls back to a serial loop
    # when num_proc <= 1 or the sequence has fewer than 2 steps.

    parts_assembled = sorted(load_part_ids(assembly_dir))

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)

    if save_mesh or save_all: # save object centric mesh
        mesh_dir = os.path.join(result_dir, 'mesh')
        os.makedirs(mesh_dir, exist_ok=True)
        all_meshes = load_part_meshes(assembly_dir, transform='none')
        # shutil.copyfile(os.path.join(assembly_dir, 'config.json'), os.path.join(mesh_dir, 'config.json'))
        if show_grasp:
            gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, visual=True)
            all_meshes.update(gripper_meshes)
        if show_arm:
            arm_meshes = load_arm_meshes(asset_folder, visual=True, convex=False)
            all_meshes.update(arm_meshes)
        save_meshes(all_meshes, mesh_dir)
    else:
        mesh_dir = None

    if save_pose or save_all:
        pose_dir = os.path.join(result_dir, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
    else:
        pose_dir = None
    
    if save_part or save_all:
        part_dir = os.path.join(result_dir, 'part_fix')
        os.makedirs(part_dir, exist_ok=True)
    else:
        part_dir = None

    if save_path or save_all:
        path_dir = os.path.join(result_dir, 'path')
        os.makedirs(path_dir, exist_ok=True)
    else:
        path_dir = None

    if save_record or save_all:
        record_dir = os.path.join(result_dir, 'record')
        os.makedirs(record_dir, exist_ok=True)
        record_dir_grasp = None
        if show_grasp:
            record_dir_grasp = os.path.join(result_dir, 'record_grasp')
            os.makedirs(record_dir_grasp, exist_ok=True)
    else:
        record_dir = None
        record_dir_grasp = None

    extra_views = extra_views or []
    for view_record_dir, _, _ in extra_views:
        os.makedirs(view_record_dir, exist_ok=True)

    try:
        # Pre-resolve per-step state so workers see no mutable accumulators.
        # parts_assembled / parts_removed at step i are fully determined by sequence[:i],
        # which is exactly what makes the loop parallelizable.
        step_args = []
        _parts_assembled = list(parts_assembled)
        _parts_removed = []
        for i, part_move in enumerate(sequence):
            parts_rest = [p for p in _parts_assembled if p != part_move]
            sim_info = tree.edges[tuple(_parts_assembled), tuple(parts_rest)]['sim_info']
            assert part_move == sim_info['part_move']
            step_args.append({
                'i': i,
                'part_move': part_move,
                'parts_rest': parts_rest,
                'parts_removed': list(_parts_removed),
                'sim_info': sim_info,
                'tool_mesh': (tool_meshes_per_step or {}).get(part_move),
            })
            _parts_assembled = parts_rest
            _parts_removed.append(part_move)

        result_paths = {
            'record_dir': record_dir,
            'pose_dir': pose_dir,
            'part_dir': part_dir,
            'path_dir': path_dir,
            'record_dir_grasp': record_dir_grasp,
            'extra_views': extra_views,
        }
        options = {
            'save_path': save_path or save_all,
            'save_record': save_record or save_all,
            'save_all': save_all,
            'save_sdf': save_sdf,
            'make_video': make_video,
            'reverse': reverse,
            'show_fix': show_fix,
            'show_grasp': show_grasp,
            'show_arm': show_arm,
            'gripper_type': gripper_type,
            'gripper_scale': gripper_scale,
            'optimizer': optimizer,
            'camera_pos': camera_pos,
            'camera_lookat': camera_lookat,
            'connect_path': connect_path,
            'n_frame': n_frame,
        }

        if num_proc and num_proc > 1 and len(step_args) > 1:
            print(f'[play_logged_plan] dispatching {len(step_args)} steps across up to {num_proc} workers')
            from utils.parallel import parallel_execute
            worker_args = [(asset_folder, assembly_dir, sa, result_paths, options) for sa in step_args]
            for _ in parallel_execute(_render_step_worker, worker_args, num_proc, desc='play_logged_plan'):
                pass
        else:
            print(f'[play_logged_plan] running {len(step_args)} steps serially (num_proc={num_proc})')
            for sa in tqdm(step_args, desc='play_logged_plan'):
                _render_step_worker(asset_folder, assembly_dir, sa, result_paths, options)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[play_logged_plan] interrupt')
        else:
            print('[play_logged_plan] exception:', e, f'from {assembly_dir}')
            print(traceback.format_exc())
        
        if clear_sdf:
            clear_saved_sdfs(assembly_dir)
        raise e

    if clear_sdf:
        clear_saved_sdfs(assembly_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--save-mesh', default=False, action='store_true')
    parser.add_argument('--save-pose', default=False, action='store_true')
    parser.add_argument('--save-part', default=False, action='store_true')
    parser.add_argument('--save-path', default=False, action='store_true')
    parser.add_argument('--save-record', default=False, action='store_true')
    parser.add_argument('--save-all', default=False, action='store_true')
    parser.add_argument('--reverse', default=False, action='store_true')
    parser.add_argument('--show-fix', default=False, action='store_true')
    parser.add_argument('--show-grasp', default=False, action='store_true')
    parser.add_argument('--show-arm', default=False, action='store_true')
    parser.add_argument('--gripper', type=str, default='robotiq-140', choices=['panda', 'robotiq-85', 'robotiq-140'])
    parser.add_argument('--scale', type=float, default=0.4)
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B')
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    parser.add_argument('--clear-sdf', default=False, action='store_true')
    parser.add_argument('--plot-tree', default=False, action='store_true')
    parser.add_argument('--make-video', default=False, action='store_true')
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[-1, 1, 0], help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=[1.25, -1.5, 1.5], help='camera position')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.log_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    with open(os.path.join(args.log_dir, 'stats.json'), 'r') as fp:
        stats = json.load(fp)
        sequence = stats['sequence']

    if args.plot_tree:
        SequencePlanner.plot_tree_with_budget(tree, budget=args.budget)

    if sequence is None:
        print('[play_logged_plan] failed plan')
    else:
        asset_folder = os.path.join(project_base_dir, './assets')
        play_logged_plan(asset_folder, args.assembly_dir, sequence, tree, args.result_dir, args.save_mesh, args.save_pose, args.save_part, args.save_path, args.save_record, args.save_all, 
            args.reverse, args.show_fix, args.show_grasp, args.show_arm, args.gripper, args.scale, args.optimizer, not args.disable_save_sdf, args.clear_sdf, args.make_video, args.budget, args.camera_pos, args.camera_lookat)
