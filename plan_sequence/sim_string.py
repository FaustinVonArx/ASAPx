import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import numpy as np
import distinctipy
import settings
import itertools
from scipy.spatial.transform import Rotation

from assets.load import load_config, load_pos_quat_dict, load_part_ids
from assets.transform import mat_dict_to_pos_quat_dict, pos_quat_to_mat


GROUND_Z = 0

KN = 1e6
KT = 1e3
MU = 0.5
DAMPING = 1e2


def _get_color(index, alpha=1.0):
    colors = [
        [210, 87, 89, 255 * alpha],
        [237, 204, 73, 255 * alpha],
        [60, 167, 221, 255 * alpha],
        [190, 126, 208, 255 * alpha],
        [108, 192, 90, 255 * alpha],
    ]
    colors = np.array(colors) / 255.0
    return colors[int(index) % 5]

# Per-run cache so the same part_id keeps the same base color across calls.
# Keyed by (scheme, normalize); value is dict[part_id -> base RGBA color] holding
# the pre-emphasis color (emphasis is reapplied per call since the moving set varies).
_BASE_COLOR_CACHE = {}


def _generate_base_color(scheme, normalize, existing_colors):
    """Produce one new base RGBA color that hasn't been handed out yet for this
    (scheme, normalize) cache bucket. `existing_colors` is the list of already-
    assigned base colors (same normalization)."""
    index = len(existing_colors)

    if scheme == "original" or scheme == "default":
        palette = np.array([
            [210, 87, 89, 255],
            [237, 204, 73, 255],
            [60, 167, 221, 255],
            [190, 126, 208, 255],
            [108, 192, 90, 255],
        ], dtype=float)
        if normalize:
            palette = palette / 255.0
        return palette[index % len(palette)].copy()

    if scheme == "distinctipy":
        exclude = [tuple(np.array(c[:3], dtype=float) / (1.0 if normalize else 255.0)) for c in existing_colors]
        # deterministic seed so repeated runs give the same sequence
        rng = index + 1
        raw = distinctipy.get_colors(1, exclude_colors=exclude, rng=rng)[0]
        r, g, b = raw
        a = settings.alpha * settings.brightness
        if normalize:
            return np.array([r, g, b, a], dtype=float)
        return np.array([int(r * 255), int(g * 255), int(b * 255), int(a * 255)], dtype=float)

    if scheme == "max_contrast":
        values = [0, 128, 255]
        rgb_combinations = list(itertools.product(values, repeat=3))
        filtered = [rgb for rgb in rgb_combinations if rgb not in [(0, 0, 0), (255, 255, 255)]]
        sorted_rgb = sorted(filtered, key=lambda x: x.count(128))
        r, g, b = sorted_rgb[index % len(sorted_rgb)]
        a = int(settings.opacity * 255)
        if normalize:
            return np.array([r / 255.0, g / 255.0, b / 255.0, a / 255.0], dtype=float)
        return np.array([r, g, b, a], dtype=float)

    raise ValueError(f"Unknown color scheme: {scheme}")


def _get_colors(part_fixed, parts_moving=None, normalize=True, scheme='default'):
    emphasize_moving = settings.emphasize_moving and parts_moving is not None
    part_ids = list(part_fixed) + (list(parts_moving) if parts_moving is not None else [])

    # Resolve "default" eagerly so the cache bucket doesn't depend on call-site part count.
    resolved_scheme = settings.color_scheme if scheme == "default" else scheme

    cache = _BASE_COLOR_CACHE.setdefault((resolved_scheme, normalize), {})
    for pid in part_ids:
        if pid not in cache:
            cache[pid] = _generate_base_color(resolved_scheme, normalize, list(cache.values()))

    max_val = 1.0 if normalize else 255
    gray = np.array([0.5, 0.5, 0.5]) * max_val
    bleak_blend = settings.bleak
    boost = settings.boost

    color_map = {}
    for part_id in part_ids:
        color = cache[part_id].astype(float).copy()
        if emphasize_moving:
            rgb = color[:3]
            if part_id in parts_moving:
                mean = rgb.mean()
                rgb = mean + (rgb - mean) * boost
                rgb = np.clip(rgb, 0, max_val)
            else:
                rgb = rgb * (1 - bleak_blend) + gray * bleak_blend
            color[:3] = rgb
        if not normalize:
            color = color.astype(int)
        color_map[part_id] = color
    return color_map

def _get_fixed_color():
    return np.array([120, 120, 120, 255]) / 255.0


def get_body_color_dict(parts_fix, parts_free, parts_moving=None): # parts_free = parts_rest - parts_fix + [part_move]
    body_color_dict = {}
    # Only `parts_moving` (typically just the currently-moving part) should be
    # emphasized; parts_free includes all non-fixed parts and must not be reused
    # as the moving-set or every part gets boosted.
    if parts_moving is None:
        parts_moving = []
    parts_moving = list(parts_moving)
    non_moving = [p for p in [*parts_fix, *parts_free] if p not in parts_moving]
    colors = _get_colors(non_moving, parts_moving=parts_moving)
    for part_id in [*parts_fix, *parts_free]:
        if part_id in parts_free:
            color = colors[part_id][:3]
        else:
            color = _get_fixed_color()[:3]
        body_color_dict[f'part{part_id}'] = color
    return body_color_dict


def _arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def _get_basic_sim_substring(gravity=False):
    substring = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. {-980 if gravity else 1e-12}"/>
<ground pos="0 0 {GROUND_Z}" normal="0 0 1"/>
'''
    return substring


def _get_path_sim_substring():
    substring = _get_basic_sim_substring(gravity=False)
    substring += f'''
<default>
    <ground_contact kn="{KN}" kt="0" mu="0" damping="{DAMPING}"/>
    <general_SDF_contact kn="{KN}" kt="0" mu="0.0" damping="{DAMPING}"/>
    <general_MultiSDF_contact kn="{KN}" kt="0" mu="0.0" damping="{DAMPING}"/>
</default>
'''
    return substring


def _get_stablility_sim_substring(gravity=True):
    substring = _get_basic_sim_substring(gravity=gravity)
    substring += f'''
<default>
    <ground_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
    <general_SDF_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
    <general_MultiSDF_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
</default>
'''
    return substring


def get_contact_sim_string(assembly_dir, parts=None, save_sdf=False, mat_dict=None):
    '''
    Simulation string for checking contact info
    '''
    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_basic_sim_substring()
    if parts is None: parts = load_part_ids(assembly_dir)
    colors = _get_colors(parts)
    for part_id in parts:
        joint_type = 'fixed'
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos_dict[part_id])}" quat="{_arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" mu="0" rgba="{_arr_to_str(colors[part_id])}"/>
    </link>
</robot>
'''
    string += f'''
</redmax>
'''
    return string


def get_path_sim_string(assembly_dir, parts_fix, part_move, parts_removed=[], save_sdf=False, pose=None, mat_dict=None, col_th=0.01, arm_string=None, tool_attach=None, floor=True):
    '''
    Simulation string for checking path assemblability.

    tool_attach: optional dict {"filename": <.obj path>, "color": "r g b a"}. When set,
    a child <link> is nested inside the moving part's link with a fixed joint at the
    parent body's origin, so the tool rides along the part's free3d-exp motion.

    floor: when False, omit the ground-contact binding on part_move so the floor no
    longer blocks world -Z motion. The <ground> element itself is kept for visual /
    sim-structure consistency.
    '''
    if pose is None: pose = np.eye(4)

    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    if len(parts_removed) > 0: # set removed parts to initial states
        pos_init_dict, quat_init_dict = load_pos_quat_dict(assembly_dir, transform='initial')
        for part_id in parts_removed:
            pos_dict[part_id] = pos_init_dict[part_id]
            quat_dict[part_id] = quat_init_dict[part_id]

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_path_sim_substring()
    # part_move must be in the id list so its colour is generated; parts_moving
    # must be a list/set so the `in` check is membership rather than substring.
    colors = _get_colors([*parts_fix, *parts_removed], [part_move])
    for part_id in [part_move, *parts_fix, *parts_removed]:

        if part_id in parts_removed:
            if pos_dict[part_id] is None or quat_dict[part_id] is None:
                continue

        if part_id == part_move:
            joint_type = 'free3d-exp'
            # color = _get_color(part_id)
            color = colors[part_id]
        else:
            joint_type = 'fixed'
            # color = _get_fixed_color()
            # color = _get_color(part_id)
            color = colors[part_id]

        matrix = pos_quat_to_mat(pos_dict[part_id], quat_dict[part_id])
        matrix = pose @ matrix
        pos = matrix[:3, 3] + np.array([0, 0, GROUND_Z]) # NOTE: pay attention when combining mat_dict and pose
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]

        if pos is None or quat is None:
            continue

        if type(col_th) == dict:
            col_th_i = col_th[part_id]
        else:
            col_th_i = col_th

        tool_inner = ""
        if part_id == part_move and tool_attach is not None:
            tool_color = tool_attach.get("color", "0.9 0.45 0.1 1.0")
            tool_inner = f'''
        <link name="tool">
            <joint name="tool" type="fixed" pos="0 0 0" quat="1 0 0 0"/>
            <body name="tool" type="mesh" filename="{tool_attach["filename"]}" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" rgba="{tool_color}"/>
        </link>'''

        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos)}" quat="{_arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" col_th="{col_th_i}" mu="0" rgba="{_arr_to_str(color)}"/>{tool_inner}
    </link>
</robot>
'''
    if arm_string is not None:
        string += arm_string
    string += f'''
<contact>
'''
    if floor:
        string += f'''
    <ground_contact body="part{part_move}"/>
'''
    for part_id in parts_fix:
        string += f'''
    <general_SDF_contact general_body="part{part_id}" SDF_body="part{part_move}"/>
    <general_SDF_contact general_body="part{part_move}" SDF_body="part{part_id}"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string


def get_stability_sim_string(assembly_dir, parts_fix, parts_move, gravity=True, save_sdf=False, pose=None, mat_dict=None, col_th=0.01):
    '''
    Simulation string for checking stability
    '''
    if pose is None: pose = np.eye(4)

    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_stablility_sim_substring(gravity=gravity)

    colors = _get_colors(parts_fix, parts_move)
    for part_id in [*parts_fix, *parts_move]:
        if part_id in parts_fix:
            joint_type = 'fixed'
            color = _get_fixed_color()
        else:
            joint_type = 'free3d-exp'
            # color = _get_color(part_id)
            color = colors[part_id]

        if mat_dict is None: # ground init
            matrix = pos_quat_to_mat(pos_dict[part_id], quat_dict[part_id])
            matrix = pose @ matrix
            pos = matrix[:3, 3] + np.array([0, 0, GROUND_Z]) # NOTE: pay attention when combining mat_dict and pose
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        else: # ground cont
            pos = pos_dict[part_id]
            quat = quat_dict[part_id]

        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos)}" quat="{_arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" col_th="{col_th}" mu="0" rgba="{_arr_to_str(color)}"/>
    </link>
</robot>
'''
    string += f'''
<contact>
'''
    part_pairs_in_contact = []
    for i, part_move in enumerate(parts_move):
        for part_fix in parts_fix:
            part_pairs_in_contact.append((part_move, part_fix))
        for j in range(i + 1, len(parts_move)):
            part_pairs_in_contact.append((part_move, parts_move[j]))
        string += f'''
    <ground_contact body="part{part_move}"/>
'''
    for part_pair in part_pairs_in_contact:
        string += f'''
    <general_SDF_contact general_body="part{part_pair[0]}" SDF_body="part{part_pair[1]}"/>
    <general_SDF_contact general_body="part{part_pair[1]}" SDF_body="part{part_pair[0]}"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string
