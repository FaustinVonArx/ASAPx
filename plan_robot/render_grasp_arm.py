import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
import json
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils.renderer import SimRenderer
from assets.load import load_pos_quat_dict, load_assembly_all_transformed, load_part_ids
from assets.color import get_color
from assets.transform import get_pos_quat_from_pose, get_transform_matrix_euler
from plan_robot.util_arm import get_arm_chain, get_arm_path_from_gripper_path, get_gripper_path_from_arm_path, get_gripper_part_path_from_arm_path, get_default_arm_rest_q, get_gripper_qm_from_arm_q
from plan_robot.util_grasp import get_gripper_finger_states, get_gripper_base_name, get_gripper_path_from_part_path
from plan_robot.motion_plan_arm import ArmMotionPlanner


def _flatten_finger_states(finger_states):
    # Concatenate joint values across all fingers. Returns an empty 1-D array
    # when the gripper has no fingers (e.g. the rod contact model), so
    # downstream `np.concatenate([..., fingers, ...])` calls don't blow up.
    if not finger_states:
        return np.zeros(0, dtype=float)
    return np.concatenate(list(finger_states.values()))


def _build_static_meshes(assembly_dir, move_id, still_ids, removed_ids, pose, move_delta_T=None):
    """Return a list of trimesh meshes representing the static environment at
    one moment in time, in world frame, used for arm-motion collision checks.

    move_delta_T: optional 4x4 relative transform applied to the move part's
    posed mesh. None => place it at its assembled (pre-disassembly) pose, used
    by the reach phase. For the retreat phase, the caller passes
        T_end @ inv(T_start)
    where T_start / T_end come from `get_transform_matrix_euler(part_path[i][:3], part_path[i][3:])`
    so the move part rides along to its post-disassembly world pose.
    """
    assembly = load_assembly_all_transformed(assembly_dir)
    pose_arr = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
    meshes = []
    for sid in still_ids:
        if sid not in assembly:
            continue
        m = assembly[sid].get('mesh_final')
        if m is None:
            continue
        m = m.copy()
        m.apply_transform(pose_arr)
        meshes.append(m)
    for rid in removed_ids:
        if rid not in assembly:
            continue
        m = assembly[rid].get('mesh_initial')
        if m is None:
            continue
        meshes.append(m.copy())
    if move_id in assembly:
        m = assembly[move_id].get('mesh_final')
        if m is not None:
            m = m.copy()
            m.apply_transform(pose_arr)
            if move_delta_T is not None:
                m.apply_transform(move_delta_T)
            meshes.append(m)
    return meshes


def _states_from_arm_path(sim, gripper_type, gripper_scale, arm_chain, arm_path_full,
                          fixed_part_local, gripper_base_name, finger_open_state):
    """Build per-frame redmax state vectors for an arm trajectory in which the
    moving part stays put (its world-frame local state never changes) and the
    arm/gripper move according to the planned `arm_path_full` (each entry is a
    full-Q list including the base link at index 0).

    fixed_part_local: the local sim state for the move part, repeated every frame.
    finger_open_state: the local sim state for the gripper fingers (open ratio
    matching the reach/retreat phase), repeated every frame.

    Returns a list of np.ndarray states ready to feed sim.set_state_his(...).
    """
    states = []
    for arm_q_full in arm_path_full:
        gripper_qm = get_gripper_qm_from_arm_q(arm_chain, arm_q_full, gripper_type)
        gripper_local = sim.get_joint_q_from_qm(gripper_base_name, gripper_qm)
        arm_active = arm_q_full[1:]  # strip the immutable base link
        states.append(np.concatenate([fixed_part_local, gripper_local, finger_open_state, arm_active]))
    return states


def render_arm_next_to_assembly(asset_folder, assembly_dir, move_id, still_ids, removed_ids,
                                pose, gripper_type, gripper_scale,
                                save_path, size=(900, 700)):
    """Off-screen pyvista snapshot of the arm at its default rest pose placed
    next to the (posed) assembly. Use this to diagnose scale / reachability
    issues without needing to step through redmax. Renders:
      - All static parts colored gray (assembly_final + parts_initial for the
        ones removed in earlier steps).
      - The arm meshes posed at rest_q with base at the first candidate from
        get_arm_pos_candidates (gripper imagined to be at the assembly center).
      - A ground plane outline at z=0 for reference.

    Returns the saved path on success, or None on failure (e.g. pyvista import
    error, missing meshes). Non-fatal.
    """
    try:
        import pyvista as pv
        from plan_robot.geometry import load_arm_meshes, transform_arm_meshes
        from plan_robot.util_arm import (
            get_arm_chain, get_arm_pos_candidates, get_arm_euler,
            get_default_arm_rest_q,
        )

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        # Static parts at their world-frame poses (matches what GraspArmPlanner
        # would see at the start of this disassembly step).
        still_meshes = _build_static_meshes(
            assembly_dir, move_id, still_ids, removed_ids, pose, move_delta_T=None,
        )

        # Assembly center / gripper imagined position. The arm-base heuristic
        # uses get_arm_pos_candidates which projects toward a gripper pose;
        # use the assembly's bounding-box center as a stand-in for "where the
        # arm would try to reach."
        if still_meshes:
            all_v = np.concatenate([m.vertices for m in still_meshes], axis=0)
            center = all_v.mean(axis=0)
            center[2] = 0.0
        else:
            center = np.zeros(3)
        gripper_pos = center + np.array([0.0, 0.0, 2.0 * max(1.0, gripper_scale * 5)])  # 5cm above center
        gripper_ori = np.array([0.0, 0.0, 1.0])  # straight down

        arm_pos_candidates = get_arm_pos_candidates(
            gripper_pos, gripper_ori, gripper_scale, center=center,
        )
        arm_pos = arm_pos_candidates[0]
        arm_euler = get_arm_euler(arm_pos, center=center)

        arm_scale = gripper_scale
        chain = get_arm_chain(base_pos=arm_pos, base_euler=arm_euler, scale=arm_scale)
        rest_q_active = get_default_arm_rest_q()
        q_full = chain.active_to_full(rest_q_active, initial_position=[0] * len(chain.links))

        # Load arm meshes (visual for the snapshot). Posed via transform_arm_meshes.
        arm_meshes_raw = load_arm_meshes(asset_folder, visual=True, convex=False)
        arm_meshes_posed = transform_arm_meshes(arm_meshes_raw, chain, q_full, scale=arm_scale)

        # Compose the scene.
        plotter = pv.Plotter(off_screen=True, window_size=tuple(size))
        for m in still_meshes:
            plotter.add_mesh(pv.wrap(m), color='lightgray', opacity=0.9, show_edges=False)
        for arm_part_name, arm_mesh in arm_meshes_posed.items():
            color = 'steelblue' if arm_part_name != 'linkbase' else 'darkslategray'
            plotter.add_mesh(pv.wrap(arm_mesh), color=color, opacity=1.0, show_edges=False)

        # Ground plane reference. Sized to encompass both arm and assembly.
        if still_meshes:
            extent = max(abs(all_v).max(), float(np.linalg.norm(arm_pos))) * 2.2
        else:
            extent = max(20.0, float(np.linalg.norm(arm_pos))) * 2.2
        ground = pv.Plane(center=(center[0], center[1], 0.0),
                          direction=(0, 0, 1), i_size=extent, j_size=extent)
        plotter.add_mesh(ground, color='wheat', opacity=0.25, show_edges=True)

        # Annotate scale.
        try:
            asm_bbox = np.ptp(all_v, axis=0) if still_meshes else np.zeros(3)
            arm_reach_approx = 0.85 * arm_scale * 100  # Panda ~85 cm at scale=100
            text = (f'assembly bbox: {asm_bbox.round(2).tolist()}\n'
                    f'arm scale: {arm_scale}  approx reach: {arm_reach_approx:.1f}\n'
                    f'arm base: {arm_pos.round(2).tolist()}\n'
                    f'assembly center: {center.round(2).tolist()}')
            plotter.add_text(text, position='upper_left', font_size=8)
        except Exception:
            pass

        plotter.camera_position = 'iso'
        plotter.screenshot(str(save_path))
        plotter.close()
        return str(save_path)
    except Exception as e:
        print(f'[render_arm_next_to_assembly] failed: {e}')
        return None


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def create_gripper_arm_with_assembly_posed_xml(assembly_dir, move_id, still_ids, removed_ids, pose=None, gripper_type=None, gripper_pos=[0, 0, 5], gripper_quat=[1, 0, 0, 0], gripper_scale=1,
    arm_pos=[-10, 10, 0], arm_euler=[0, 0, 0], arm_scale=1):
    part_ids = [move_id] + still_ids + removed_ids
    all_part_ids = load_part_ids(assembly_dir)
    color_map = get_color(all_part_ids)
    pos_dict_final, quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    pos_dict_initial, quat_dict_initial = load_pos_quat_dict(assembly_dir, transform='initial')
    arm_quat = R.from_euler('xyz', arm_euler).as_quat()[[3, 0, 1, 2]]
    string = f'''
<redmax model="gripper_arm">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>
<ground pos="0 0 0" normal="0 0 1"/>
'''
    for part_id in part_ids:
        joint_type = 'free3d-exp' if part_id == move_id else 'fixed'
        if part_id in removed_ids:
            pos, quat = pos_dict_initial[part_id], quat_dict_initial[part_id]
            if pos is None or quat is None:
                continue
        else:
            pos, quat = get_pos_quat_from_pose(pos_dict_final[part_id], quat_dict_final[part_id], pose)
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="mesh" filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="{arr_to_str(color_map[part_id])}"/>
    </link>
</robot>
'''
    gripper_pos, gripper_quat = get_pos_quat_from_pose(gripper_pos, gripper_quat, pose)
    if gripper_type == 'panda':
        string += f'''
<robot>
    <link name="panda_hand">
        <joint name="panda_hand" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name="panda_hand" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/hand.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        <link name="panda_leftfinger">
            <joint name="panda_leftfinger" type="prismatic" axis="0 1 0" pos="0 0 {5.84 * gripper_scale}" quat="1 0 0 0" lim="0.0 {4 * gripper_scale}"/>
            <body name="panda_leftfinger" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        </link>
        <link name="panda_rightfinger">
            <joint name="panda_rightfinger" type="prismatic" axis="0 -1 0" pos="0 0 {5.84 * gripper_scale}" quat="1 0 0 0" lim="0.0 {4 * gripper_scale}"/>
            <body name="panda_rightfinger" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="0 0 0 1" transform_type="OBJ_TO_JOINT"/>
        </link>
    </link>
</robot>
'''
    elif gripper_type == 'robotiq_85':
        string += f'''
<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_85/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="{3.06011444260539 * gripper_scale} 0.0 {6.27920162695395 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name="robotiq_left_outer_finger" type="fixed" pos="{3.16910442266543 * gripper_scale} 0.0 {-0.193396375724605 * gripper_scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="{1.27000000001501 * gripper_scale} 0.0 {6.93074999999639 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_inner_finger">
                <joint name = "robotiq_left_inner_finger" type="revolute" pos="{3.4585310861294003 * gripper_scale} 0.0 {4.5497019381797505 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="-0.8757 0.0"/>
                <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="{-3.06011444260539 * gripper_scale} 0.0 {6.27920162695395 * gripper_scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_finger">
                <joint name="robotiq_right_outer_finger" type="fixed" pos="{3.16910442266543 * gripper_scale} 0.0 {-0.193396375724605 * gripper_scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="{-1.27000000001501 * gripper_scale} 0.0 {6.93074999999639 * gripper_scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 1.0 0.0" lim="-0.8757 0.0"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_inner_finger">
                <joint name = "robotiq_right_inner_finger" type="revolute" pos="{3.4585310861294003 * gripper_scale} 0.0 {4.5497019381797505 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 1.0 0.0" lim="0.0 0.8757" damping="0.0"/>
                <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
    </link>
</robot>
'''
    elif gripper_type == 'robotiq-140':
        string += f'''
<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_140/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="0 {-3.0601 * gripper_scale} {5.4905 * gripper_scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="-1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name = "robotiq_left_outer_finger" type="fixed" pos="0 {1.821998610742 * gripper_scale} {2.60018192872234 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_left_inner_finger">
                    <joint name = "robotiq_left_inner_finger" type="revolute" pos="0 {8.17554015893473 * gripper_scale} {-2.82203446692936 * gripper_scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_left_pad">
                        <joint name = "robotiq_left_pad" type="fixed" pos="0 {3.8 * gripper_scale} {-2.3 * gripper_scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_left_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="0 {-1.27 * gripper_scale} {6.142 * gripper_scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="0 {3.0601 * gripper_scale} {5.4905 * gripper_scale}" quat="0.0 0.0 0.91190335 0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_knuckle">
                <joint name = "robotiq_right_outer_finger" type="fixed" pos="0 {1.821998610742 * gripper_scale} {2.60018192872234 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_right_inner_finger">
                    <joint name = "robotiq_right_inner_finger" type="revolute" pos="0 {8.17554015893473 * gripper_scale} {-2.82203446692936 * gripper_scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_right_pad">
                        <joint name = "robotiq_right_pad" type="fixed" pos="0 {3.8 * gripper_scale} {-2.3 * gripper_scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_right_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="0 {1.27 * gripper_scale} {6.142 * gripper_scale}" quat="0.0 0.0 -0.91190335 -0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
    </link>
</robot>
'''
    elif gripper_type == 'rod':
        # Single rod cylinder, base at the wrist (free3d-exp joint), tip
        # extending in +z of the local frame. No fingers. The mesh is
        # auto-generated by _ensure_rod_obj in geometry.py at first use
        # and lives at asset_folder/rod/visual/rod.obj.
        string += f'''
<robot>
    <link name="rod_base">
        <joint name="rod_base" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name="rod_base" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="rod/visual/rod.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" rgba="0.85 0.55 0.15 1.0"/>
    </link>
</robot>
'''
    else:
        raise NotImplementedError
    string += f'''
<robot>
    <link name="linkbase">
        <joint name="linkbase" type="fixed" pos="{arr_to_str(arm_pos)}" quat="{arr_to_str(arm_quat)}"/>
        <body name= "linkbase" type = "abstract" pos = "{-2.1131 * arm_scale} {-0.16302 * arm_scale} {5.6488 * arm_scale}" quat = "0.41928822390350623 -0.3384325829202692 0.5661965601674424 0.6237645608467303" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "885.5600000000001" inertia = "16772.46795287213 33528.24905872843 38202.28298839946" rgba = "0.8 0.8 0.8 1.0">
            <visual mesh = "xarm7/visual/linkbase_smooth.obj" pos = "{4.2037215880446706 * arm_scale} {-4.303182668106523 * arm_scale} {-0.4604913739852395 * arm_scale}" quat = "-0.4192882239035062 -0.3384325829202692 0.5661965601674422 0.6237645608467302"/>
            <collision contacts = "xarm7/contacts/linkbase.txt" pos = "{4.2037215880446706 * arm_scale} {-4.303182668106523 * arm_scale} {-0.4604913739852395 * arm_scale}" quat = "-0.4192882239035062 -0.3384325829202692 0.5661965601674422 0.6237645608467302"/>
        </body>
        <link name="joint1">
            <joint name = "joint1" type="revolute" pos="0.0 0.0 {26.700000000000003 * arm_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
            <body name= "link1" type = "abstract" pos = "{-0.42142 * arm_scale} {2.8209999999999997 * arm_scale} {-0.87788 * arm_scale}" quat = "0.6918679191340069 0.0005252148724432689 -0.6060703296327886 0.39242484906198305" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "426.03000000000003" inertia = "8235.113114059062 13775.66035044562 14455.126535495323" rgba = "0.9 0.9 0.9 1.0">
                <visual mesh = "xarm7/visual/link1_smooth.obj" pos = "{-0.8114216775604369 * arm_scale} {-2.5981972214308087 * arm_scale} {1.2236319587744608 * arm_scale}" quat = "0.6918679191340068 -0.0005252148724432888 0.6060703296327885 -0.392424849061983"/>
                <collision contacts = "xarm7/contacts/link1_vhacd.txt" pos = "{-0.8114216775604369 * arm_scale} {-2.5981972214308087 * arm_scale} {1.2236319587744608 * arm_scale}" quat = "0.6918679191340068 -0.0005252148724432888 0.6060703296327885 -0.392424849061983"/>
            </body>
            <link name="joint2">
                <joint name = "joint2" type="revolute" pos="0.0 0.0 0.0" quat="0.7071054825112364 -0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-2.059 2.0944" damping="0.0"/>
                <body name= "link2" type = "abstract" pos = "{-0.0033178 * arm_scale} {-12.849 * arm_scale} {2.6337 * arm_scale}" quat = "0.6337905179370223 -0.3150527964362565 0.6307030926183779 -0.3182215011472822" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "560.9499999999999" inertia = "9808.04401005623 31159.849558318598 31915.106431625172" rgba = "0.8 0.8 0.8 1.0">
                    <visual mesh = "xarm7/visual/link2_smooth.obj" pos = "{-8.711764384019158 * arm_scale} {9.804940501796535 * arm_scale} {-0.03861050843999272 * arm_scale}" quat = "0.6337905179370222 0.3150527964362564 -0.6307030926183778 0.31822150114728215"/>
                    <collision contacts = "xarm7/contacts/link2_vhacd.txt" pos = "{-8.711764384019158 * arm_scale} {9.804940501796535 * arm_scale} {-0.03861050843999272 * arm_scale}" quat = "0.6337905179370222 0.3150527964362564 -0.6307030926183778 0.31822150114728215"/>
                </body>
                <link name="joint3">
                    <joint name = "joint3" type="revolute" pos="0.0 {-29.299999999999997 * arm_scale} 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                    <body name= "link3" type = "abstract" pos = "{4.223 * arm_scale} {-2.3258 * arm_scale} {-0.9667399999999999 * arm_scale}" quat = "-0.24066027491433598 0.8530839593824141 -0.23989343952005776 0.39595647235247505" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "444.63000000000005" inertia = "7804.745076690154 11912.598616264928 13322.65630704492" rgba = "0.9 0.9 0.9 1.0">
                        <visual mesh = "xarm7/visual/link3_smooth.obj" pos = "{-3.266493961637458 * arm_scale} {-1.4456637070979548 * arm_scale} {-3.379013837226153 * arm_scale}" quat = "0.24066027491433598 0.8530839593824141 -0.23989343952005787 0.39595647235247505"/>
                        <collision contacts = "xarm7/contacts/link3_vhacd.txt" pos = "{-3.266493961637458 * arm_scale} {-1.4456637070979548 * arm_scale} {-3.379013837226153 * arm_scale}" quat = "0.24066027491433598 0.8530839593824141 -0.23989343952005787 0.39595647235247505"/>
                    </body>
                    <link name="joint4">
                        <joint name = "joint4" type="revolute" pos="{5.25 * arm_scale} 0.0 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-0.19198 3.927" damping="0.0"/>
                        <body name= "link4" type = "abstract" pos = "{6.7148 * arm_scale} {-10.732 * arm_scale} {2.4479 * arm_scale}" quat = "0.6707722696010192 0.47605887893032944 0.012741086884645855 -0.5685685278230704" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "523.8699999999999" inertia = "8944.094777271237 28270.539922454394 28898.36530027436" rgba = "0.8 0.8 0.8 1.0">
                            <visual mesh = "xarm7/visual/link4_smooth.obj" pos = "{-9.059983366567941 * arm_scale} {-7.802235142560355 * arm_scale} {-4.826842200415131 * arm_scale}" quat = "0.6707722696010191 -0.47605887893032933 -0.012741086884645859 0.5685685278230704"/>
                            <collision contacts = "xarm7/contacts/link4_vhacd.txt" pos = "{-9.059983366567941 * arm_scale} {-7.802235142560355 * arm_scale} {-4.826842200415131 * arm_scale}" quat = "0.6707722696010191 -0.47605887893032933 -0.012741086884645859 0.5685685278230704"/>
                        </body>
                        <link name="joint5">
                            <joint name = "joint5" type="revolute" pos="{7.75 * arm_scale} {-34.25 * arm_scale} 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                            <body name= "link5" type = "abstract" pos = "{-0.023397 * arm_scale} {3.6705 * arm_scale} {-8.0064 * arm_scale}" quat = "0.6892055355098681 -0.16046408568558085 0.698228216539248 -0.10827910535309972" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "185.54000000000002" inertia = "2471.2608713476575 9886.134859618787 9955.304269033553" rgba = "0.9 0.9 0.9 1.0">
                                <visual mesh = "xarm7/visual/link5_smooth.obj" pos = "{-6.057144261834268 * arm_scale} {-6.378684331187784 * arm_scale} {-0.4460361240938076 * arm_scale}" quat = "-0.6892055355098679 -0.16046408568558085 0.6982282165392479 -0.1082791053530997"/>
                                <collision contacts = "xarm7/contacts/link5_vhacd.txt" pos = "{-6.057144261834268 * arm_scale} {-6.378684331187784 * arm_scale} {-0.4460361240938076 * arm_scale}" quat = "-0.6892055355098679 -0.16046408568558085 0.6982282165392479 -0.1082791053530997"/>
                            </body>
                            <link name="joint6">
                                <joint name = "joint6" type="revolute" pos="0.0 0.0 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-1.69297 3.14159265359" damping="0.0"/>
                                <body name= "link6" type = "abstract" pos = "{5.8911 * arm_scale} {2.8469 * arm_scale} {0.68428 * arm_scale}" quat = "0.9529732922502667 0.01599291734637919 0.1692539717525629 0.250876909855057" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "313.44" inertia = "3867.077886736355 7688.706404074782 8278.915709188861" rgba = "1.0 1.0 1.0 1.0">
                                    <visual mesh = "xarm7/visual/link6_smooth.obj" pos = "{-5.973444004856806 * arm_scale} {0.21893372810568068 * arm_scale} {-2.747393798118139 * arm_scale}" quat = "0.9529732922502666 -0.01599291734637919 -0.16925397175256282 -0.250876909855057"/>
                                    <collision contacts = "xarm7/contacts/link6_vhacd.txt" pos = "{-5.973444004856806 * arm_scale} {0.21893372810568068 * arm_scale} {-2.747393798118139 * arm_scale}" quat = "0.9529732922502666 -0.01599291734637919 -0.16925397175256282 -0.250876909855057"/>
                                </body>
                                <link name="joint7">
                                    <joint name = "joint7" type="revolute" pos="{7.6 * arm_scale} {9.700000000000001 * arm_scale} 0.0" quat="0.7071054825112364 -0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                                    <body name= "link7" type = "abstract" pos = "{-0.0015846 * arm_scale} {-0.46376999999999996 * arm_scale} {-1.2705 * arm_scale}" quat = "-0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "314.68" inertia = "1192.0774998754055 1698.502197488278 2603.520302636317" rgba = "0.753 0.753 0.753 1.0">
                                        <visual mesh = "xarm7/visual/link7_smooth.obj" pos = "{-0.4828448608210097 * arm_scale} {-0.0019607575065535687 * arm_scale} {-1.2633734086428683 * arm_scale}" quat = "0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751"/>
                                        <collision contacts = "xarm7/contacts/link7_vhacd.txt" pos = "{-0.4828448608210097 * arm_scale} {-0.0019607575065535687 * arm_scale} {-1.2633734086428683 * arm_scale}" quat = "0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751"/>
                                    </body>
                                </link>
                            </link>
                        </link>
                    </link>
                </link>
            </link>
        </link>
    </link>
</robot>
</redmax>
    '''
    return string


def get_finger_open_path(gripper_type, gripper_scale, open_ratio_start, open_ratio_end, n_frame):
    finger_path = []
    for i in range(n_frame):
        open_ratio = open_ratio_start + (open_ratio_end - open_ratio_start) * (i + 1) / (n_frame + 1)
        finger_states = _flatten_finger_states(
            get_gripper_finger_states(gripper_type, open_ratio, gripper_scale)
        )
        finger_path.append(finger_states)
    return finger_path


def get_all_paths_from_arm_path(sim, arm_chain, arm_path, gripper_type, open_ratio, gripper_scale, gripper_to_part_transform=None, move_id=None, truncate=None):

    gripper_base_name = get_gripper_base_name(gripper_type)
    if gripper_to_part_transform is None:
        gripper_path = get_gripper_path_from_arm_path(arm_chain, arm_path, gripper_type)
        part_path = None
    else:
        gripper_path, part_path = get_gripper_part_path_from_arm_path(arm_chain, arm_path, gripper_type, gripper_to_part_transform)
        part_path = [sim.get_joint_q_from_qm(f'part{move_id}', qm) for qm in part_path]

    gripper_path = [sim.get_joint_q_from_qm(gripper_base_name, qm) for qm in gripper_path]
    finger_states = get_gripper_finger_states(gripper_type, open_ratio, gripper_scale)
    finger_flat = _flatten_finger_states(finger_states)
    finger_path = [finger_flat for _ in range(len(gripper_path))]
    arm_path = [arm_q[1:] for arm_q in arm_path]

    if truncate == 'start':
        gripper_path, finger_path, arm_path = gripper_path[1:], finger_path[1:], arm_path[1:]
        if part_path is not None: part_path = part_path[1:]
    elif truncate == 'end':
        gripper_path, finger_path, arm_path = gripper_path[:-1], finger_path[:-1], arm_path[:-1]
        if part_path is not None: part_path = part_path[:-1]
    
    if part_path is None:
        return arm_path, gripper_path, finger_path
    else:
        return arm_path, gripper_path, finger_path, part_path


def render_step_from_precomputed(asset_folder, assembly_dir, move_id, still_ids, removed_ids,
                                  pose, part_path, gripper_type, gripper_scale,
                                  step_plan, transition_in_path=None, transition_out_path=None,
                                  arm_pos=None, arm_euler=None,
                                  camera_lookat=None, camera_pos=None, body_color_map=None,
                                  reverse=False, record_path=None, make_video=False):
    """Replay a precomputed (transition_in + disassembly + transition_out)
    arm trajectory for a single sequence step.

    Inputs are arm_pipeline.plan_arm_sequence() outputs:
      step_plan = the dict from plan['steps'][i] — carries grasp, arm_path_full
                  (full Q per disassembly waypoint), and part_path_local.
      transition_in_path = inbound transition arm_path_full (rest -> grasp or
                  prev_step.end_q -> this.start_q). None to skip the inbound
                  phase.
      transition_out_path = outbound transition arm_path_full (typically only
                  set for the very last step, to retract to rest). None to skip.
      arm_pos / arm_euler = the shared world-frame arm base used by the
                  whole sequence (from plan['arm_pos'], plan['arm_euler']).

    No IK or RRT is run here — purely a state-vector replay through redmax.
    Returns export_replay_matrices() on success, None otherwise.
    """
    if part_path is None or step_plan is None or not step_plan.get('feasible'):
        return None

    grasp_pos = np.array(step_plan['grasp']['pos'], dtype=float)
    grasp_quat = np.array(step_plan['grasp']['quat'], dtype=float)
    open_ratio = float(step_plan['grasp']['open_ratio'])
    arm_path_full = [np.array(q, dtype=float) for q in step_plan['arm_path_full']]
    arm_scale = gripper_scale

    if arm_pos is None or arm_euler is None:
        # Shouldn't happen when called from the pipeline, but be defensive.
        print('[render_step_from_precomputed] missing arm_pos/arm_euler; aborting.')
        return None
    arm_pos = np.asarray(arm_pos, dtype=float)
    arm_euler = np.asarray(arm_euler, dtype=float)

    xml_string = create_gripper_arm_with_assembly_posed_xml(
        assembly_dir=assembly_dir, move_id=move_id, still_ids=still_ids,
        removed_ids=removed_ids, pose=pose,
        gripper_type=gripper_type, gripper_pos=grasp_pos, gripper_quat=grasp_quat,
        gripper_scale=gripper_scale,
        arm_pos=arm_pos, arm_euler=arm_euler, arm_scale=arm_scale,
    )
    sim = redmax.Simulation(xml_string, asset_folder)
    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None:
        sim.viewer_options.camera_pos = camera_pos

    grasp_finger_states = get_gripper_finger_states(gripper_type, open_ratio, gripper_scale)
    for finger_name, finger_state in grasp_finger_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)
    if body_color_map is not None:
        sim.set_body_color_map(body_color_map)

    finger_grasp_state = _flatten_finger_states(grasp_finger_states)
    finger_open_state = _flatten_finger_states(
        get_gripper_finger_states(gripper_type, 1.0, gripper_scale)
    )

    # Disassembly-phase state vectors built from the precomputed arm path
    # and the disassembly part path.
    arm_chain = get_arm_chain(base_pos=arm_pos, base_euler=arm_euler, scale=arm_scale)
    gripper_base_name = get_gripper_base_name(gripper_type)
    part_path_local = [sim.get_joint_q_from_qm(f'part{move_id}', np.asarray(qm, dtype=float)) for qm in part_path]
    gripper_path = [get_gripper_qm_from_arm_q(arm_chain, q, gripper_type) for q in arm_path_full]
    gripper_path_local = [sim.get_joint_q_from_qm(gripper_base_name, qm) for qm in gripper_path]
    # Pad / truncate gripper path to match part path length (rare mismatch
    # when the path got re-interpolated).
    n_steps = min(len(part_path_local), len(arm_path_full), len(gripper_path_local))
    states_disassembly = []
    for k in range(n_steps):
        arm_active = arm_path_full[k][1:]
        states_disassembly.append(np.concatenate([
            part_path_local[k], gripper_path_local[k], finger_grasp_state, arm_active,
        ]))

    # Inbound / outbound transitions: arm moves with no part held. Part stays
    # parked at its first / last disassembly state respectively; fingers open.
    states_in = []
    if transition_in_path:
        first_part_state = part_path_local[0]
        states_in = _states_from_arm_path(
            sim, gripper_type, gripper_scale, arm_chain,
            [np.array(q, dtype=float) for q in transition_in_path],
            fixed_part_local=first_part_state,
            gripper_base_name=gripper_base_name,
            finger_open_state=finger_open_state,
        )

    states_out = []
    if transition_out_path:
        last_part_state = part_path_local[-1]
        states_out = _states_from_arm_path(
            sim, gripper_type, gripper_scale, arm_chain,
            [np.array(q, dtype=float) for q in transition_out_path],
            fixed_part_local=last_part_state,
            gripper_base_name=gripper_base_name,
            finger_open_state=finger_open_state,
        )

    states = states_in + states_disassembly + states_out
    if reverse:
        states = states[::-1]
    sim.set_state_his(states, [np.zeros_like(states[0]) for _ in range(len(states))])

    SimRenderer.replay(sim, record=record_path is not None,
                       record_path=record_path, make_video=make_video)
    return sim.export_replay_matrices()


def render_path_with_grasp_and_arm(asset_folder, assembly_dir, move_id, still_ids, removed_ids, pose, part_path, gripper_type, gripper_scale, grasp, optimizer,
    camera_lookat=None, camera_pos=None, body_color_map=None, reverse=False, render=True, record_path=None, make_video=False):
    if part_path is None:
        print('no path found')
        return

    gripper_pos, gripper_quat = grasp.pos, grasp.quat
    arm_pos, arm_euler, arm_q_pre = grasp.arm_pos, grasp.arm_euler, grasp.arm_q # full
    arm_scale = gripper_scale

    xml_string = create_gripper_arm_with_assembly_posed_xml(
        assembly_dir=assembly_dir, move_id=move_id, still_ids=still_ids, removed_ids=removed_ids, pose=pose, 
        gripper_type=gripper_type, gripper_pos=gripper_pos, gripper_quat=gripper_quat, gripper_scale=gripper_scale,
        arm_pos=arm_pos, arm_euler=arm_euler, arm_scale=arm_scale)
    sim = redmax.Simulation(xml_string, asset_folder)
    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None:
        sim.viewer_options.camera_pos = camera_pos

    finger_states = get_gripper_finger_states(gripper_type, grasp.open_ratio, gripper_scale)
    for finger_name, finger_state in finger_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    if body_color_map is not None:
        sim.set_body_color_map(body_color_map)

    # get gripper path
    gripper_path = get_gripper_path_from_part_path(part_path, gripper_pos, gripper_quat)

    # get part to gripper transform
    part_transform_pre = get_transform_matrix_euler(part_path[0][:3], part_path[0][3:])
    gripper_transform_pre = get_transform_matrix_euler(gripper_path[0][:3], gripper_path[0][3:])
    gripper_to_part_transform = np.linalg.inv(gripper_transform_pre) @ part_transform_pre # TODO: check
    
    # transform from global coordinate to local coordinate
    part_path_local = [sim.get_joint_q_from_qm(f'part{move_id}', qm) for qm in part_path]
    gripper_base_name = get_gripper_base_name(gripper_type)
    gripper_path_local = [sim.get_joint_q_from_qm(gripper_base_name, qm) for qm in gripper_path]
    finger_flat = _flatten_finger_states(finger_states)
    finger_path_local = [finger_flat for _ in range(len(gripper_path_local))]

    # get arm path
    arm_chain = get_arm_chain(base_pos=arm_pos, base_euler=arm_euler, scale=arm_scale)
    arm_path_full = get_arm_path_from_gripper_path(gripper_path, gripper_type, arm_chain, arm_q_pre, optimizer)  # full
    arm_path_local = [arm_q[1:] for arm_q in arm_path_full]  # active

    # Disassembly-phase states (existing behavior).
    states_disassembly = [
        np.concatenate([part_state, gripper_state, finger_state, arm_state])
        for part_state, gripper_state, finger_state, arm_state
        in zip(part_path_local, gripper_path_local, finger_path_local, arm_path_local)
    ]

    # Reach + retreat: plan collision-free arm trajectories in joint space via
    # RRT-Connect. The reach trajectory takes the arm from rest_q to arm_q_pre
    # with the gripper open and no part held. The retreat trajectory takes it
    # from the final disassembly arm_q back to rest_q, again with no part held;
    # the just-removed part stays parked at its post-disassembly pose so it's
    # part of the collision environment.
    #
    # Both phases are best-effort: if RRT fails (collision at start/goal, no
    # connection within the iteration budget, or any backend error) we silently
    # skip that phase and fall back to rendering only the disassembly motion.
    rest_q_active = list(get_default_arm_rest_q())  # 7-element active
    arm_q_first_full = np.asarray(arm_path_full[0], dtype=float)
    arm_q_last_full = np.asarray(arm_path_full[-1], dtype=float)
    arm_q_first_active = arm_q_first_full[1:]
    arm_q_last_active = arm_q_last_full[1:]
    finger_open_state = _flatten_finger_states(
        get_gripper_finger_states(gripper_type, 1.0, gripper_scale)
    )
    part_state_first = part_path_local[0]
    part_state_last = part_path_local[-1]

    states_reach = []
    states_retreat = []
    try:
        motion_planner = ArmMotionPlanner(
            base_pos=arm_pos, base_euler=arm_euler,
            scale=arm_scale, gripper_type=gripper_type,
        )
    except Exception as e:
        print(f'[render_path_with_grasp_and_arm] ArmMotionPlanner init failed: {e}; '
              f'skipping reach/retreat phases.')
        motion_planner = None

    if motion_planner is not None:
        # Reach: rest -> grasp, environment = full assembly with move part at
        # its starting pose, no part attached to the gripper.
        try:
            still_meshes_reach = _build_static_meshes(
                assembly_dir, move_id, still_ids, removed_ids, pose, move_delta_T=None,
            )
            reach_path_full = motion_planner.plan_with_grasp(
                start=rest_q_active, goal=list(arm_q_first_active),
                move_mesh=None, move_transform=None,
                still_meshes=still_meshes_reach, open_ratio=1.0,
                verbose=False,
            )
            if reach_path_full is not None:
                states_reach = _states_from_arm_path(
                    sim, gripper_type, gripper_scale, arm_chain, reach_path_full,
                    fixed_part_local=part_state_first,
                    gripper_base_name=gripper_base_name,
                    finger_open_state=finger_open_state,
                )
            else:
                print('[render_path_with_grasp_and_arm] reach RRT returned no path; skipping reach phase.')
        except Exception as e:
            print(f'[render_path_with_grasp_and_arm] reach planning failed: {e}; skipping reach phase.')

        # Retreat: grasp_last -> rest, environment = full assembly with move
        # part parked at its post-disassembly pose, no part attached.
        try:
            T_start = get_transform_matrix_euler(part_path[0][:3], part_path[0][3:])
            T_end = get_transform_matrix_euler(part_path[-1][:3], part_path[-1][3:])
            move_delta_T = T_end @ np.linalg.inv(T_start)
            still_meshes_retreat = _build_static_meshes(
                assembly_dir, move_id, still_ids, removed_ids, pose, move_delta_T=move_delta_T,
            )
            retreat_path_full = motion_planner.plan_with_grasp(
                start=list(arm_q_last_active), goal=rest_q_active,
                move_mesh=None, move_transform=None,
                still_meshes=still_meshes_retreat, open_ratio=1.0,
                verbose=False,
            )
            if retreat_path_full is not None:
                states_retreat = _states_from_arm_path(
                    sim, gripper_type, gripper_scale, arm_chain, retreat_path_full,
                    fixed_part_local=part_state_last,
                    gripper_base_name=gripper_base_name,
                    finger_open_state=finger_open_state,
                )
            else:
                print('[render_path_with_grasp_and_arm] retreat RRT returned no path; skipping retreat phase.')
        except Exception as e:
            print(f'[render_path_with_grasp_and_arm] retreat planning failed: {e}; skipping retreat phase.')

    # Concatenate: reach (gripper opens at grasp), disassembly, retreat. We
    # don't insert separate finger-closing/opening interpolation frames — the
    # finger state simply switches between the open and grasp-closed values at
    # the phase boundaries. Good enough for visualization; if a smoother
    # transition is wanted later, `get_finger_open_path` is already wired up.
    states = states_reach + states_disassembly + states_retreat
    if reverse:
        states = states[::-1]
    sim.set_state_his(states, [np.zeros_like(states[0]) for _ in range(len(states))])

    if render:
        SimRenderer.replay(sim, record=record_path is not None, record_path=record_path, make_video=make_video)

    return sim.export_replay_matrices()
