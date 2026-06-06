"""Two-stage arm trajectory planning over a full disassembly sequence.

Stage 1 (per-step, parallel): for each step, iterate the 4 base candidates
on the circle around that step's assembly centroid (same selection as the
original GraspArmPlanner) and keep the first base that yields a feasible
in-grasp trajectory. Per-step base is stored on the result so each step's
arm is independent — between steps the base "teleports" to a new position
with no inter-step joint-space transition needed (by design — user-elected
behavior).

Stage 2 (per-step, parallel): for each feasible step, plan two short
RRT-Connect trajectories at that step's chosen base:
  - reach:   rest_q -> step.start_q (no part held)
  - retreat: step.end_q -> rest_q  (no part held; the just-removed part is
             parked at its disassembly endpoint in the collision env)

There are no inter-step transitions in this design; each step's GIF
plays back as reach + disassembly + retreat, independent of every other
step.

Result is written to `<log_dir>/arm_plans.json` and consumed by
play_logged_plan render workers.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from time import time

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from assets.load import load_assembly_all_transformed, load_part_ids
from assets.transform import get_transform_matrix_euler
from plan_robot.util_arm import (
    get_arm_chain,
    get_arm_pos_candidates,
    get_arm_euler,
    get_default_arm_rest_q,
)
from utils.parallel import parallel_execute


# ---------------------------------------------------------------------------
# Time estimation (joint-space constant-velocity model)
# ---------------------------------------------------------------------------

# xarm7 active joints with continuous rotation — must match interpolate_q
# in motion_plan_arm.py. Continuous joints take the shortest mod-2π path.
_CONTINUOUS_ACTIVE_JOINTS = {0, 2, 4, 6}


def _shortest_angular_step(a, b, continuous):
    if continuous:
        d = (b - a) % (2.0 * np.pi)
        if d > np.pi:
            d -= 2.0 * np.pi
        return abs(d)
    return abs(b - a)


def _arm_path_distance_rad(arm_path_full):
    """Slowest-joint distance along the path: for each active joint, sum the
    shortest angular step over consecutive waypoints; then take the MAX across
    joints. Models concurrent joint motion (the whole arm finishes when the
    slowest joint does) rather than serialised motion (sum). Operates on the
    active 7-DoF slice (arm_q_full[1:]). Returns 0.0 for empty / single-
    waypoint paths so duration is well-defined."""
    if not arm_path_full or len(arm_path_full) < 2:
        return 0.0
    arm = [np.asarray(q, dtype=float)[1:] for q in arm_path_full]
    n_joints = min(len(q) for q in arm)
    if n_joints == 0:
        return 0.0
    per_joint = np.zeros(n_joints, dtype=float)
    prev = arm[0][:n_joints]
    for cur_full in arm[1:]:
        cur = cur_full[:n_joints]
        for j in range(n_joints):
            per_joint[j] += _shortest_angular_step(prev[j], cur[j], j in _CONTINUOUS_ACTIVE_JOINTS)
        prev = cur
    return float(per_joint.max())


def _annotate_durations(stage1_results, transitions, velocity_rad_s):
    """Compute and attach `path_distance_rad` + `duration_s` to every feasible
    step + transition. Infeasible entries get 0.0 for both. Results marked
    `simplified=True` already carry a pre-computed `duration_s` from the
    simplified worker — those are left untouched."""
    velocity = max(float(velocity_rad_s), 1e-9)
    for r in stage1_results or []:
        if not r:
            continue
        if r.get('simplified'):
            # Preserve the simplified worker's closed-form duration.
            r.setdefault('path_distance_rad', 0.0)
            r.setdefault('duration_s', 0.0)
            continue
        if r.get('feasible') and r.get('arm_path_full'):
            dist = _arm_path_distance_rad(r['arm_path_full'])
            r['path_distance_rad'] = dist
            r['duration_s'] = dist / velocity
        else:
            r['path_distance_rad'] = 0.0
            r['duration_s'] = 0.0
    for t in transitions or []:
        if not t:
            continue
        if t.get('feasible') and t.get('arm_path_full'):
            dist = _arm_path_distance_rad(t['arm_path_full'])
            t['path_distance_rad'] = dist
            t['duration_s'] = dist / velocity
        else:
            t['path_distance_rad'] = 0.0
            t['duration_s'] = 0.0


def _build_time_estimate(stage1_results, transitions, velocity_rad_s):
    step_secs = [float(r.get('duration_s', 0.0)) for r in (stage1_results or []) if r]
    trans_secs = [float(t.get('duration_s', 0.0)) for t in (transitions or []) if t]
    return {
        'velocity_rad_s': float(velocity_rad_s),
        'steps_total_s': sum(step_secs),
        'transitions_total_s': sum(trans_secs),
        'total_s': sum(step_secs) + sum(trans_secs),
    }


def _horizontal_angle(point_xy, center_xy):
    """Horizontal angle (rad) of point as seen from center, in [-pi, pi].
    Returns None when point coincides with center within a small tolerance."""
    dx = float(point_xy[0]) - float(center_xy[0])
    dy = float(point_xy[1]) - float(center_xy[1])
    if dx * dx + dy * dy < 1e-12:
        return None
    return float(np.arctan2(dy, dx))


def _shortest_angle_delta(a, b):
    """Shortest signed angular difference b - a, wrapped to [-pi, pi]."""
    d = (b - a) % (2.0 * np.pi)
    if d > np.pi:
        d -= 2.0 * np.pi
    return float(abs(d))


def _pose_rotation_angle(pose_a, pose_b):
    """Magnitude (rad) of the relative rotation that takes the assembly from
    pose_a to pose_b. Either may be None / identity. Translation is ignored."""
    Ra = np.asarray(pose_a, dtype=float)[:3, :3] if pose_a is not None else np.eye(3)
    Rb = np.asarray(pose_b, dtype=float)[:3, :3] if pose_b is not None else np.eye(3)
    R_rel = Rb @ Ra.T
    cos = (np.trace(R_rel) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.arccos(cos))


def _safe_median(values, fallback=0.0):
    finite = [float(v) for v in values
              if v is not None and np.isfinite(v) and v > 0.0]
    return float(np.median(finite)) if finite else float(fallback)


def _build_timing_overview(stage1_results, transitions, step_infos, sequence,
                           velocity_rad_s, base_travel_velocity,
                           reorient_velocity_rad_s, fail_multiplier,
                           time_per_held_part_s=0.0,
                           assembly_center=None):
    """Assemble the full per-step / per-transition timing breakdown.

    Components (all in seconds):
      step_disassembly_s[k]   from stage1_results[k]['duration_s'] (or fallback)
      transition_s[k]         from transitions table for inbound to step k
                              (kind 'prefix' for k=0, 'inter' otherwise)
      base_travel_s[k]        ARC LENGTH between consecutive part centroids,
                              measured around the shared assembly center on the
                              circle of average radius r_avg = (r_{k-1}+r_k)/2:
                                  arc = r_avg · |Δθ|
                                  base_travel_s[k] = arc / base_travel_velocity
                              Models the operator/robot walking around the
                              assembly between attention loci. Geometry-only —
                              defined regardless of arm planner outcome. 0 for
                              k=0 and whenever either centroid is unknown.
      reorientation_s[k]      rotation_angle(pose_{k-1}, pose_k) / velocity.
                              0 for k=0.
      hold_s[k]               len(step_infos[k]['parts_fix']) *
                              time_per_held_part_s. 0 when parts_fix is None
                              or empty.

    Failed step disassemblies and failed transitions are imputed as
    median(successful) * fail_multiplier.
    """
    n = len(sequence)
    base_velocity_linear = max(float(base_travel_velocity), 1e-9)
    reorient_velocity = max(float(reorient_velocity_rad_s), 1e-9)
    hold_penalty = float(time_per_held_part_s)
    asm_center = (np.asarray(assembly_center, dtype=float)
                  if assembly_center is not None else np.zeros(3))
    asm_center = asm_center.copy()
    if asm_center.size >= 3:
        asm_center[2] = 0.0

    # ----- step disassembly -----
    step_raw = []
    step_was_failure = []
    for k in range(n):
        r = stage1_results[k] if k < len(stage1_results) else None
        feasible = bool(r and r.get('feasible'))
        dur = float(r.get('duration_s', 0.0)) if r else 0.0
        step_raw.append(dur if feasible else 0.0)
        step_was_failure.append(not feasible)
    step_median = _safe_median(step_raw)
    step_fallback = step_median * float(fail_multiplier)
    step_disassembly_s = [
        step_fallback if step_was_failure[k] else step_raw[k]
        for k in range(n)
    ]

    # ----- stage-2 transition (inbound to step k) -----
    # to_step == k is the inbound transition for step k (prefix when k=0,
    # inter otherwise). Failed transitions get median * multiplier.
    inbound = [None] * n
    for t in (transitions or []):
        if not t:
            continue
        to_step = t.get('to_step')
        if to_step is None or not (0 <= to_step < n):
            continue
        kind = t.get('kind')
        if kind in ('prefix', 'inter', 'reach', None):
            inbound[to_step] = t
    trans_raw = []
    trans_was_failure = []
    for k in range(n):
        t = inbound[k]
        feasible = bool(t and t.get('feasible'))
        dur = float(t.get('duration_s', 0.0)) if t else 0.0
        trans_raw.append(dur if feasible else 0.0)
        # Only mark a failure when a transition entry exists but is infeasible.
        # Missing entries (no transition planned) count as 0 with no fallback.
        trans_was_failure.append(bool(t) and not feasible)
    trans_median = _safe_median(trans_raw)
    trans_fallback = trans_median * float(fail_multiplier)
    transition_s = [
        trans_fallback if trans_was_failure[k] else trans_raw[k]
        for k in range(n)
    ]

    # ----- focus-shift arc travel between consecutive part centroids -----
    # Pure geometry: defined for every step (including planner-failed ones)
    # because part centroids come from the tree's stored pose + the part's
    # local mesh, not from the arm planner.
    #
    # For step k>=1, with c_k = horizontal centroid of part_move at step k:
    #   r_k = ||c_k - assembly_center||
    #   Δθ_k = shortest angular delta between (c_{k-1}-C) and (c_k-C)
    #   r_avg = (r_{k-1} + r_k) / 2
    #   arc_k = r_avg * |Δθ_k|
    #   base_travel_s[k] = arc_k / base_velocity_linear
    #
    # When either centroid is missing (mesh-load failure), the step contributes
    # 0 and is marked imputed so the plot can show it.
    base_raw = [0.0] * n
    base_was_failure = [False] * n
    base_arcs = [0.0] * n
    base_thetas = [0.0] * n
    centroids = []
    for k in range(n):
        c = (step_infos[k].get('part_centroid_world')
             if k < len(step_infos) else None)
        centroids.append(np.asarray(c, dtype=float) if c is not None else None)
    for k in range(1, n):
        c_prev = centroids[k - 1]
        c_cur = centroids[k]
        if c_prev is None or c_cur is None:
            base_was_failure[k] = True
            continue
        v_prev = c_prev[:2] - asm_center[:2]
        v_cur = c_cur[:2] - asm_center[:2]
        r_prev = float(np.linalg.norm(v_prev))
        r_cur = float(np.linalg.norm(v_cur))
        # Angle around the shared center. _horizontal_angle returns None when
        # the part centroid coincides with the assembly center (r ≈ 0). In
        # that case Δθ is undefined but r_avg ≈ 0 makes the arc trivially 0,
        # so we treat the contribution as 0 rather than imputing.
        ang_prev = _horizontal_angle(c_prev[:2], asm_center[:2])
        ang_cur = _horizontal_angle(c_cur[:2], asm_center[:2])
        if ang_prev is None or ang_cur is None:
            base_arcs[k] = 0.0
            base_thetas[k] = 0.0
            continue
        delta = _shortest_angle_delta(ang_prev, ang_cur)
        r_avg = 0.5 * (r_prev + r_cur)
        arc = r_avg * float(delta)
        base_arcs[k] = float(arc)
        base_thetas[k] = float(delta)
        base_raw[k] = arc / base_velocity_linear
    base_median = _safe_median([v for k, v in enumerate(base_raw) if k > 0 and not base_was_failure[k]])
    base_fallback = base_median * float(fail_multiplier)
    base_travel_s = [
        base_fallback if (k > 0 and base_was_failure[k]) else base_raw[k]
        for k in range(n)
    ]

    # ----- assembly reorientation (k>=1) -----
    reorient_raw = [0.0] * n
    reorient_angles = [0.0] * n
    for k in range(1, n):
        pose_prev = step_infos[k - 1].get('pose')
        pose_cur = step_infos[k].get('pose')
        angle = _pose_rotation_angle(pose_prev, pose_cur)
        reorient_angles[k] = angle
        reorient_raw[k] = angle / reorient_velocity
    # No fallback for reorientation — angles are derived from poses, which
    # are always known regardless of robot planning success.

    # ----- extra-parts-held penalty (per step) -----
    # Sourced from sim_info['parts_fix'] which the planner records on each
    # tree edge — the count of EXTRA fixtures (beyond the moving gripper)
    # needed to keep the assembly stable during the step. None => stability
    # didn't resolve a fix list; treated as 0 (no penalty).
    hold_counts = [0] * n
    hold_raw = [0.0] * n
    for k in range(n):
        sa = step_infos[k] if k < len(step_infos) else None
        parts_fix = sa.get('parts_fix') if sa else None
        if parts_fix is None:
            continue
        hold_counts[k] = int(len(parts_fix))
        hold_raw[k] = float(len(parts_fix)) * hold_penalty

    # ----- per-step rollup -----
    per_step = []
    for k in range(n):
        per_step.append({
            'step': k,
            'part_move': sequence[k],
            'step_disassembly_s': step_disassembly_s[k],
            'step_disassembly_was_imputed': step_was_failure[k],
            'transition_s': transition_s[k],
            'transition_was_imputed': trans_was_failure[k],
            'base_travel_s': base_travel_s[k],
            'base_travel_was_imputed': base_was_failure[k],
            'base_travel_arc': base_arcs[k],
            'base_travel_angle_rad': base_thetas[k],
            'reorientation_s': reorient_raw[k],
            'reorientation_angle_rad': reorient_angles[k],
            'hold_s': hold_raw[k],
            'hold_count': hold_counts[k],
            'total_s': (step_disassembly_s[k] + transition_s[k]
                        + base_travel_s[k] + reorient_raw[k] + hold_raw[k]),
        })

    totals = {
        'step_disassembly_s': float(sum(step_disassembly_s)),
        'transitions_s': float(sum(transition_s)),
        'base_travel_s': float(sum(base_travel_s)),
        'reorientation_s': float(sum(reorient_raw)),
        'hold_s': float(sum(hold_raw)),
    }
    totals['total_s'] = sum(totals.values())

    return {
        'velocity_rad_s': float(velocity_rad_s),
        'base_travel_velocity': float(base_travel_velocity),
        'assembly_center': asm_center.tolist(),
        'reorientation_velocity_rad_s': float(reorient_velocity_rad_s),
        'failed_step_time_multiplier': float(fail_multiplier),
        'time_per_held_part_s': hold_penalty,
        'medians_used_for_fallback': {
            'step_disassembly_s': step_median,
            'transition_s': trans_median,
            'base_travel_s': base_median,
        },
        'per_step': per_step,
        'totals': totals,
    }


def _step_horizontal_center(sa):
    """Cheap horizontal-plane centroid from a step_infos entry without
    re-loading the mesh. Falls back to the pose translation when geometry is
    unavailable; matches what _compute_step_center stores at planning time
    closely enough for the timing model (angle differences, not absolute
    positions, are what matter)."""
    if sa.get('center') is not None:
        c = np.asarray(sa['center'], dtype=float)
    elif sa.get('pose') is not None:
        c = np.asarray(sa['pose'], dtype=float)[:3, 3]
    else:
        c = np.zeros(3)
    c = c.astype(float).copy()
    if c.size >= 3:
        c[2] = 0.0
    return c


def _log_timing_overview(overview):
    print(f'[arm_pipeline:timing] complete time estimate '
          f'(joint v={overview["velocity_rad_s"]:.2f} rad/s, '
          f'base v={overview["base_travel_velocity"]:.2f} u/s, '
          f'reorient v={overview["reorientation_velocity_rad_s"]:.2f} rad/s, '
          f'hold +{overview.get("time_per_held_part_s", 0.0):.2f} s/part, '
          f'fail x{overview["failed_step_time_multiplier"]:.2f})', flush=True)
    print(f'[arm_pipeline:timing]    step | part | disasm | trans  | base   | reorient | hold(n)     | total', flush=True)
    print(f'[arm_pipeline:timing]    -----+------+--------+--------+--------+----------+-------------+--------', flush=True)
    for e in overview['per_step']:
        flags = ''
        if e['step_disassembly_was_imputed']:
            flags += 'D'
        if e['transition_was_imputed']:
            flags += 'T'
        if e['base_travel_was_imputed']:
            flags += 'B'
        flag_str = f' [{flags}]' if flags else ''
        hold_label = f'{e.get("hold_s", 0.0):6.2f}({e.get("hold_count", 0):>2})'
        print(f'[arm_pipeline:timing]    {e["step"]:>4} | {str(e["part_move"]):>4} | '
              f'{e["step_disassembly_s"]:6.2f} | {e["transition_s"]:6.2f} | '
              f'{e["base_travel_s"]:6.2f} | {e["reorientation_s"]:8.2f} | '
              f'{hold_label:>11} | '
              f'{e["total_s"]:6.2f}{flag_str}', flush=True)
    t = overview['totals']
    print(f'[arm_pipeline:timing]    --------------------------------------------------------', flush=True)
    print(f'[arm_pipeline:timing]    disassembly:    {t["step_disassembly_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    transitions:    {t["transitions_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    base travel:    {t["base_travel_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    reorientation:  {t["reorientation_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    hold penalty:   {t.get("hold_s", 0.0):7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    TOTAL:          {t["total_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:timing]    (flags: D=disassembly imputed, T=transition imputed, B=base-travel imputed)', flush=True)


def _log_time_estimate(stage1_results, transitions, time_estimate):
    print(f'[arm_pipeline:time] joint-space constant-velocity model  '
          f'(v = {time_estimate["velocity_rad_s"]:.3f} rad/s)', flush=True)
    for r in stage1_results or []:
        if not r or not r.get('feasible'):
            continue
        print(f'[arm_pipeline:time]    step {r["step"]} ({r["part_move"]:>3}): '
              f'{r["path_distance_rad"]:7.2f} rad  →  {r["duration_s"]:6.2f} s', flush=True)
    for t in transitions or []:
        if not t or not t.get('feasible'):
            continue
        f_, to = t.get('from_step'), t.get('to_step')
        kind = t.get('kind', '')
        f_label = 'rest' if f_ == -1 else f's{f_}'
        to_label = 'rest' if to == -1 else f's{to}'
        print(f'[arm_pipeline:time]    {kind:<6} {f_label}→{to_label}: '
              f'{t["path_distance_rad"]:7.2f} rad  →  {t["duration_s"]:6.2f} s', flush=True)
    print(f'[arm_pipeline:time]    ------------------------------------------', flush=True)
    print(f'[arm_pipeline:time]    steps:       {time_estimate["steps_total_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:time]    transitions: {time_estimate["transitions_total_s"]:7.2f} s', flush=True)
    print(f'[arm_pipeline:time]    TOTAL:       {time_estimate["total_s"]:7.2f} s', flush=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_arm_sequence(asset_folder, assembly_dir, sequence, tree,
                     gripper_type='robotiq-140', gripper_scale=0.4,
                     log_dir=None, num_proc=8, optimizer='L-BFGS-B',
                     plan_suffix_to_rest=True, debug=False,
                     stage1_timeout_s=None, stage2_timeout_s=None,
                     debug_snapshot_dir=None):
    """Run the two-stage arm planner over `sequence` (a list of part_move ids
    in disassembly order) and the planner's `tree` (an nx.DiGraph with
    per-edge sim_info). Writes <log_dir>/arm_plans.json and returns the plan.

    Returns a dict with the schema documented in this module's docstring.
    Steps / transitions for which planning failed have feasible=False and
    no arm_path; downstream renderers must check that flag.

    Timeouts:
      stage1_timeout_s: per-step wall-clock budget. Defaults to
        settings.grasp_planner_timeout_s.
      stage2_timeout_s: per-transition wall-clock budget. Defaults to
        settings.arm_planner_timeout_s.
      Pass None to either to disable that timeout.
    """
    if not sequence:
        print('[arm_pipeline] empty sequence; nothing to plan.', flush=True)
        return _empty_plan(gripper_type, gripper_scale)

    # Resolve timeouts + timing-model parameters from settings.
    velocity_rad_s = 1.0
    base_travel_velocity = 2.0
    reorient_velocity_rad_s = 0.5
    fail_multiplier = 1.5
    time_per_held_part_s = 0.0
    try:
        import settings as _settings
        if stage1_timeout_s is None:
            stage1_timeout_s = getattr(_settings, 'grasp_planner_timeout_s', None)
        if stage2_timeout_s is None:
            stage2_timeout_s = getattr(_settings, 'arm_planner_timeout_s', None)
        velocity_rad_s = float(getattr(_settings, 'arm_joint_velocity_rad_s', 1.0))
        base_travel_velocity = float(getattr(_settings, 'arm_base_travel_velocity', 2.0))
        reorient_velocity_rad_s = float(getattr(_settings, 'assembly_reorientation_velocity_rad_s', 0.5))
        fail_multiplier = float(getattr(_settings, 'failed_step_time_multiplier', 1.5))
        time_per_held_part_s = float(getattr(_settings, 'time_per_held_part_s', 0.0))
    except ImportError:
        pass

    # Default debug snapshot dir: <log_dir>/arm_debug/. Stage 1 workers save
    # an arm-vs-assembly PNG here for any step that ends up infeasible, so
    # the user can eyeball scale/reachability immediately.
    if debug_snapshot_dir is None and log_dir is not None:
        debug_snapshot_dir = str(Path(log_dir) / 'arm_debug')

    t_start = time()
    print(f'[arm_pipeline] ============================================================', flush=True)
    print(f'[arm_pipeline]  Two-stage arm planning  |  {len(sequence)} step(s)', flush=True)
    print(f'[arm_pipeline]  assembly_dir: {assembly_dir}', flush=True)
    print(f'[arm_pipeline]  log_dir:      {log_dir}', flush=True)
    print(f'[arm_pipeline]  gripper_type: {gripper_type}   scale: {gripper_scale}'
          f'{"   (rod contact model)" if gripper_type == "rod" else ""}', flush=True)
    print(f'[arm_pipeline]  num_proc:     {num_proc}', flush=True)
    print(f'[arm_pipeline]  timeouts:     stage1={stage1_timeout_s}s  stage2={stage2_timeout_s}s', flush=True)
    print(f'[arm_pipeline]  sequence:     {list(sequence)}', flush=True)
    print(f'[arm_pipeline] ============================================================', flush=True)

    # ─────────── Simplification mode (read first so prep can short-circuit) ───────────
    # When simplified mode is on AND grasp checking is off, no downstream
    # consumer in this pipeline needs the disassembly path — distance comes
    # from a deterministic mesh-based proxy. Skip the physics-replay path
    # replanning entirely. This is the only "still touching the planner" step
    # otherwise; without it the pipeline is closed-form and instant.
    simplified_mode = False
    k_dist = 1.0
    k_vol = 0.01
    simplified_check_grasp = True
    try:
        import settings as _settings_simp
        simplified_mode = bool(getattr(_settings_simp, 'arm_simplified_mode', False))
        k_dist = float(getattr(_settings_simp, 'arm_simplified_k_dist', 1.0))
        k_vol = float(getattr(_settings_simp, 'arm_simplified_k_vol', 0.01))
        simplified_check_grasp = bool(getattr(_settings_simp, 'arm_simplified_check_grasp', True))
    except ImportError:
        pass
    need_paths = (not simplified_mode) or simplified_check_grasp

    # ─────────── Step info + path replanning ───────────
    print(f'[arm_pipeline:prep] preparing step infos for {len(sequence)} step(s) '
          f'(replan_paths={need_paths})...', flush=True)
    t_prep = time()
    step_infos = _extract_step_infos(asset_folder, assembly_dir, sequence, tree,
                                     num_proc=num_proc,
                                     replan_paths=need_paths)
    _log_step_info_summary(step_infos, t_prep)

    # One sequence-wide circle of evenly distributed base candidates. Every
    # stage 1 worker picks from this same world-frame list — base positions
    # don't drift with the per-step centroid anymore, so consecutive steps
    # default to the same base unless geometry forces a move.
    asm_bbox = _assembly_bbox(assembly_dir, sequence, step_infos)
    arm_reach_approx = 0.85 * gripper_scale * 100  # Panda ≈85cm at scale=100
    n_angle = 4
    shared_center, shared_base_candidates = _sequence_base_circle(
        assembly_dir, sequence, step_infos, gripper_scale, n_angle=n_angle,
    )
    print(f'[arm_pipeline:base]  assembly bbox: {[round(float(x), 2) for x in asm_bbox]}', flush=True)
    print(f'[arm_pipeline:base]  arm scale={gripper_scale}  approx reach: {arm_reach_approx:.1f} cm', flush=True)
    print(f'[arm_pipeline:base]  shared circle: center={[round(float(x), 2) for x in shared_center]}  '
          f'n_angle={n_angle}', flush=True)
    for ci, cp in enumerate(shared_base_candidates):
        print(f'[arm_pipeline:base]    candidate {ci}: {[round(float(x), 2) for x in cp]}', flush=True)

    # ─────────── Stage 1 ───────────
    # simplified_mode / k_dist / k_vol / simplified_check_grasp resolved
    # above (before path replanning, so that step could short-circuit too).
    if simplified_mode:
        grasp_tag = 'rod feasibility ON' if simplified_check_grasp else 'rod feasibility OFF'
        print(f'[arm_pipeline:stage1] SIMPLIFIED mode (settings.arm_simplified_mode=True) — '
              f'{grasp_tag}; closed-form cost  (k_dist={k_dist}, k_vol={k_vol})', flush=True)
        t_s1 = time()
        stage1_results = _run_stage1_simplified(
            asset_folder, assembly_dir, step_infos,
            gripper_type, gripper_scale,
            num_proc=num_proc,
            k_dist=k_dist, k_vol=k_vol,
            fail_multiplier=fail_multiplier,
            check_grasp=simplified_check_grasp,
        )
        _log_stage1_summary(stage1_results, t_s1)
    else:
        print(f'[arm_pipeline:stage1] starting per-step in-grasp planning '
              f'({len(step_infos)} steps, num_proc={num_proc})', flush=True)
        t_s1 = time()
        stage1_results = _run_stage1(asset_folder, assembly_dir, step_infos,
                                      gripper_type, gripper_scale,
                                      num_proc=num_proc, debug=debug,
                                      max_time=stage1_timeout_s,
                                      debug_snapshot_dir=debug_snapshot_dir,
                                      shared_base_candidates=shared_base_candidates)
        _log_stage1_summary(stage1_results, t_s1)

    # ─────────── Stage 2 ───────────
    if simplified_mode:
        print(f'[arm_pipeline:stage2] skipped (simplified mode); transitions=[]', flush=True)
        transitions = []
    else:
        print(f'[arm_pipeline:stage2] starting inter-step transition planning '
              f'(num_proc={num_proc})', flush=True)
        t_s2 = time()
        transitions = _run_stage2(asset_folder, assembly_dir, step_infos, stage1_results,
                                  gripper_type, gripper_scale,
                                  num_proc=num_proc, debug=debug,
                                  max_time=stage2_timeout_s)
        _log_stage2_summary(transitions, t_s2)

    # ─────────── Time estimate (joint-space constant-velocity) ───────────
    _annotate_durations(stage1_results, transitions, velocity_rad_s)
    time_estimate = _build_time_estimate(stage1_results, transitions, velocity_rad_s)
    _log_time_estimate(stage1_results, transitions, time_estimate)

    # ─────────── Full timing overview (per-step disasm + transition +
    # base-arc travel + assembly reorientation, with failure fallback) ───────
    timing_overview = _build_timing_overview(
        stage1_results, transitions, step_infos, list(sequence),
        velocity_rad_s=velocity_rad_s,
        base_travel_velocity=base_travel_velocity,
        reorient_velocity_rad_s=reorient_velocity_rad_s,
        fail_multiplier=fail_multiplier,
        time_per_held_part_s=time_per_held_part_s,
        assembly_center=shared_center,
    )
    _log_timing_overview(timing_overview)

    # ─────────── Persist ───────────
    plan = {
        'gripper_type': gripper_type,
        'gripper_scale': gripper_scale,
        'rest_q_active': list(get_default_arm_rest_q()),
        'sequence': list(sequence),
        'steps': stage1_results,
        'transitions': transitions,
        'time_estimate': time_estimate,
        'timing_overview': timing_overview,
    }

    if log_dir is not None:
        out = Path(log_dir) / 'arm_plans.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(plan, f, indent=2, default=_json_default)
        print(f'[arm_pipeline] wrote {out}', flush=True)

        # Standalone timing file, easier to consume than digging into the
        # larger arm_plans.json. Same content as plan['timing_overview'].
        out_t = Path(log_dir) / 'timing_overview.json'
        with open(out_t, 'w') as f:
            json.dump(timing_overview, f, indent=2, default=_json_default)
        print(f'[arm_pipeline] wrote {out_t}', flush=True)

    print(f'[arm_pipeline] ============================================================', flush=True)
    print(f'[arm_pipeline] DONE in {time()-t_start:.1f}s', flush=True)
    print(f'[arm_pipeline] ============================================================', flush=True)
    return plan


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_step_info_summary(step_infos, t_start):
    n_paths_ok = sum(1 for sa in step_infos if sa.get('path') is not None)
    n_paths_fail = len(step_infos) - n_paths_ok
    print(f'[arm_pipeline:prep] done in {time()-t_start:.1f}s   '
          f'paths: {n_paths_ok} ok, {n_paths_fail} replanning failed', flush=True)
    for sa in step_infos:
        if sa.get('path') is None:
            print(f'[arm_pipeline:prep]    step {sa["i"]} ({sa["part_move"]}): '
                  f'path replanning FAILED — stage 1 will be skipped', flush=True)


def _log_stage1_summary(results, t_start):
    n = len(results)
    n_ok = sum(1 for r in results if r and r.get('feasible'))
    print(f'[arm_pipeline:stage1] done in {time()-t_start:.1f}s   '
          f'feasible: {n_ok}/{n}', flush=True)
    for r in results:
        if not r:
            continue
        i = r.get('step')
        part = r.get('part_move')
        if r.get('feasible'):
            arm_path_len = len(r.get('arm_path_full') or [])
            print(f'[arm_pipeline:stage1]    ✓ step {i} ({part})   arm_path_len={arm_path_len}', flush=True)
        else:
            reason = r.get('fail_reason') or '?'
            print(f'[arm_pipeline:stage1]    ✗ step {i} ({part})   reason: {reason}', flush=True)


def _log_stage2_summary(transitions, t_start):
    n = len(transitions)
    n_ok = sum(1 for t in transitions if t and t.get('feasible'))
    print(f'[arm_pipeline:stage2] done in {time()-t_start:.1f}s   '
          f'feasible: {n_ok}/{n}', flush=True)
    for t in transitions:
        if not t:
            continue
        f_, to = t.get('from_step'), t.get('to_step')
        kind = t.get('kind', '')
        f_label = 'rest' if f_ == -1 else f'step {f_}'
        to_label = 'rest' if to == -1 else f'step {to}'
        arrow = f'{kind} {f_label} → {to_label}' if kind else f'{f_label} → {to_label}'
        if t.get('feasible'):
            arm_path_len = len(t.get('arm_path_full') or [])
            print(f'[arm_pipeline:stage2]    ✓ {arrow}   arm_path_len={arm_path_len}', flush=True)
        else:
            reason = t.get('fail_reason') or '?'
            print(f'[arm_pipeline:stage2]    ✗ {arrow}   reason: {reason}', flush=True)


def _assembly_bbox(assembly_dir, sequence, step_infos):
    """Bounding-box dimensions of the full assembly at the first step's pose.
    Best-effort; returns zeros on any failure."""
    try:
        from plan_sequence.stable_pose import get_combined_mesh
        all_parts = list(step_infos[0]['parts_rest']) + [step_infos[0]['part_move']]
        mesh = get_combined_mesh(assembly_dir, all_parts)
        if mesh is None or len(mesh.vertices) == 0:
            return np.zeros(3)
        pose = np.array(step_infos[0]['pose']) if step_infos[0].get('pose') is not None else np.eye(4)
        verts = (pose[:3, :3] @ mesh.vertices.T).T + pose[:3, 3]
        return np.ptp(verts, axis=0)
    except Exception:
        return np.zeros(3)


def _sequence_base_circle(assembly_dir, sequence, step_infos, gripper_scale, n_angle=4):
    """Return (shared_center, [arm_pos × n_angle]) — one circle of evenly
    distributed candidate base positions used by every stage 1 worker.

    Replaces the per-step `get_arm_pos_candidates` call so every step picks
    its base from the SAME N world-frame positions. Continuity across steps
    is then automatic (the only inter-step base motion is a discrete jump
    between two indices of this fixed list, often the same index).

    Center is the horizontal-plane centroid of the FULL assembly at the
    first step's pose. Radius is set by `get_arm_pos` from a straight-down
    stand-in gripper at that center (same formula the original code used,
    just evaluated once for the whole sequence instead of per-step).
    """
    from plan_sequence.stable_pose import get_combined_mesh
    if not step_infos:
        center = np.zeros(3)
    else:
        sa0 = step_infos[0]
        try:
            all_parts = list(sa0['parts_rest']) + [sa0['part_move']]
            mesh = get_combined_mesh(assembly_dir, all_parts)
        except Exception:
            mesh = None
        if mesh is not None and len(mesh.vertices) > 0:
            pose = (np.array(sa0['pose']) if sa0.get('pose') is not None
                    else np.eye(4))
            verts = (pose[:3, :3] @ mesh.vertices.T).T + pose[:3, 3]
            center = verts.mean(axis=0)
        else:
            center = (np.asarray(sa0['pose'], dtype=float)[:3, 3]
                      if sa0.get('pose') is not None else np.zeros(3))
    center = np.asarray(center, dtype=float).copy()
    if center.size >= 3:
        center[2] = 0.0
    sample_gripper_pos = center + np.array([0.0, 0.0, 5.0 * max(0.1, gripper_scale)])
    sample_gripper_ori = np.array([0.0, 0.0, 1.0])
    candidates = get_arm_pos_candidates(
        sample_gripper_pos, sample_gripper_ori, gripper_scale,
        center=center, n_angle=n_angle,
    )
    return center, [np.asarray(c, dtype=float) for c in candidates]


def load_arm_plan(log_dir):
    """Load arm_plans.json from log_dir. Returns None when the file is absent
    (e.g. settings.arm_continuous=False or planning hasn't been run yet)."""
    if log_dir is None:
        return None
    p = Path(log_dir) / 'arm_plans.json'
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f'[arm_pipeline] load_arm_plan({p}) failed: {e}')
        return None


# ---------------------------------------------------------------------------
# Step info extraction (re-plan disassembly paths in parallel)
# ---------------------------------------------------------------------------


def _extract_step_infos(asset_folder, assembly_dir, sequence, tree, num_proc,
                        replan_paths=True):
    """Build per-step planning context and re-plan the disassembly part path
    for each step. The part path is what GraspArmPlanner.plan needs as input
    and isn't stored on tree edges (the tree only stores `action`)."""
    parts_assembled_all = sorted(load_part_ids(assembly_dir))

    # Precompute each part's local-frame centroid once. Avoids reloading
    # meshes per step for the world-frame centroid lookup below.
    from plan_sequence.stable_pose import get_combined_mesh as _gcm
    _local_centroids = {}
    for p in parts_assembled_all:
        try:
            m = _gcm(assembly_dir, [p])
        except Exception:
            m = None
        if m is not None and len(m.vertices) > 0:
            _local_centroids[p] = np.asarray(m.vertices.mean(axis=0), dtype=float)
        else:
            _local_centroids[p] = None

    step_args = []
    _assembled = list(parts_assembled_all)
    _removed = []
    for i, part_move in enumerate(sequence):
        parts_rest = [p for p in _assembled if p != part_move]
        sim_info = tree.edges[tuple(_assembled), tuple(parts_rest)]['sim_info']
        assert part_move == sim_info['part_move']
        # World-frame horizontal centroid of the moving part AT THIS STEP'S
        # pose. Drives the arc-length focus-shift timing in
        # _build_timing_overview — pure geometry, defined regardless of arm
        # planner outcome.
        pose_arr = (np.array(sim_info['pose']) if sim_info.get('pose') is not None
                    else np.eye(4))
        local_c = _local_centroids.get(part_move)
        if local_c is not None:
            world_c = pose_arr[:3, :3] @ local_c + pose_arr[:3, 3]
            world_c = np.asarray(world_c, dtype=float).copy()
            world_c[2] = 0.0
            part_centroid_world = world_c.tolist()
        else:
            part_centroid_world = None
        step_args.append({
            'i': i,
            'part_move': part_move,
            'parts_rest': parts_rest,
            'parts_removed': list(_removed),
            'action': np.array(sim_info['action']).tolist(),
            'pose': (np.array(sim_info['pose']).tolist()
                     if sim_info.get('pose') is not None else None),
            # Extra parts that must be held to keep the assembly stable during
            # this step (beyond the moving gripper). None when the stability
            # check didn't resolve a fix list; [] when no extra holds needed.
            'parts_fix': (list(sim_info['parts_fix'])
                          if sim_info.get('parts_fix') is not None else None),
            'part_centroid_world': part_centroid_world,
        })
        _assembled = parts_rest
        _removed.append(part_move)

    # Parallel path replanning. Skipped when the caller passes
    # replan_paths=False (e.g. simplified mode with grasp check off — no
    # downstream consumer needs the path then, and the physics replay is the
    # only heavy step in this function).
    if replan_paths:
        worker_args = [(asset_folder, assembly_dir, sa) for sa in step_args]
        if num_proc > 1 and len(worker_args) > 1:
            results = list(parallel_execute(
                _path_worker, worker_args, num_proc=num_proc,
                show_progress=True, desc='arm_pipeline paths',
            ))
        else:
            results = [_path_worker(*wa) for wa in worker_args]
        for sa, path in zip(step_args, results):
            sa['path'] = path  # list of 6-vectors (qm), or None if replan failed
    else:
        for sa in step_args:
            sa['path'] = None
        print('[arm_pipeline:prep] path replanning skipped '
              '(replan_paths=False)', flush=True)
    return step_args


def _path_worker(asset_folder, assembly_dir, sa):
    """Re-plan a single step's disassembly path. Returns list[list[float]] or None."""
    t0 = time()
    try:
        from plan_sequence.physics_planner import MultiPartPathPlanner
        from plan_sequence.play_logged_plan import interpolate_path
        pose = np.array(sa['pose']) if sa['pose'] is not None else None
        action = np.array(sa['action'])
        planner = MultiPartPathPlanner(
            asset_folder, assembly_dir, sa['parts_rest'], sa['part_move'],
            parts_removed=sa['parts_removed'], pose=pose, save_sdf=False,
        )
        success, path = planner.plan_path(action, rotation=True, connect_path=False)
        if not success:
            print(f'[arm_pipeline:prep:worker] step {sa["i"]} ({sa["part_move"]}): '
                  f'plan_path success=False in {time()-t0:.1f}s', flush=True)
            return None
        raw_len = len(path)
        min_len = 300
        while len(path) < min_len:
            path = interpolate_path(path)
        print(f'[arm_pipeline:prep:worker] step {sa["i"]} ({sa["part_move"]}): '
              f'path planned ({raw_len} → {len(path)} frames) in {time()-t0:.1f}s', flush=True)
        return [list(map(float, q)) for q in path]
    except Exception as e:
        print(f'[arm_pipeline:prep:worker] step {sa["i"]} ({sa["part_move"]}) exception '
              f'after {time()-t0:.1f}s: {e}', flush=True)
        return None


# ---------------------------------------------------------------------------
# Arm base picker
# ---------------------------------------------------------------------------


def _compute_step_center(assembly_dir, sa):
    """Best-effort world-frame centroid of the assembly at step sa's pose,
    projected to z=0. Used by stage 1 workers to seed the 4-candidate arm
    base circle around the step's actual assembly position."""
    from plan_sequence.stable_pose import get_combined_mesh
    pose = np.array(sa['pose']) if sa.get('pose') is not None else np.eye(4)
    try:
        mesh = get_combined_mesh(assembly_dir, list(sa['parts_rest']) + [sa['part_move']])
    except Exception:
        mesh = None
    if mesh is not None and len(mesh.vertices) > 0:
        verts = (pose[:3, :3] @ mesh.vertices.T).T + pose[:3, 3]
        c = verts.mean(axis=0)
    else:
        c = pose[:3, 3].copy() if isinstance(pose, np.ndarray) else np.zeros(3)
    c = np.asarray(c, dtype=float)
    c[2] = 0.0
    return c


# ---------------------------------------------------------------------------
# Stage 1 — per-step in-grasp planning
# ---------------------------------------------------------------------------


def _run_stage1(asset_folder, assembly_dir, step_infos,
                gripper_type, gripper_scale, num_proc, debug=False,
                max_time=None, debug_snapshot_dir=None,
                shared_base_candidates=None):
    # shared_base_candidates: list of world-frame base positions used by every
    # step (sequence-wide circle). When None the worker falls back to the
    # legacy per-step circle around its own assembly centroid.
    shared_serialised = (
        [np.asarray(p, dtype=float).tolist() for p in shared_base_candidates]
        if shared_base_candidates is not None else None
    )
    worker_args = []
    for sa in step_infos:
        worker_args.append((
            asset_folder, assembly_dir, sa,
            gripper_type, gripper_scale, max_time, debug_snapshot_dir,
            shared_serialised,
        ))

    if num_proc > 1 and len(worker_args) > 1:
        out = list(parallel_execute(
            _stage1_worker, worker_args, num_proc=num_proc,
            show_progress=True, desc='arm_pipeline stage 1',
        ))
    else:
        out = [_stage1_worker(*wa) for wa in worker_args]

    # Sort by step index so the order matches sequence even when parallel
    # workers complete out of order.
    out_by_step = {r['step']: r for r in out if r is not None}
    return [out_by_step.get(i, _stage1_failed(step_infos[i])) for i in range(len(step_infos))]


def _run_stage1_simplified(asset_folder, assembly_dir, step_infos,
                            gripper_type, gripper_scale, num_proc,
                            k_dist, k_vol, fail_multiplier,
                            check_grasp=True):
    """Lightweight stage-1 replacement used when settings.arm_simplified_mode
    is True. Per step:
      - measures cartesian path length d of the disassembly motion
      - measures part mesh volume V
      - optionally runs a rod-grasp feasibility check (no IK, no RRT) when
        `check_grasp` is True; skipped entirely when False, in which case
        every step is treated as feasible and the bare cost is returned.
      - returns duration_s = k_dist · d · (1 + k_vol · V),
        multiplied by `fail_multiplier` when no rod contact survives.

    Output schema is compatible with `_build_timing_overview` and the
    renderer fallback: arm_path_full / arm_pos / arm_euler / grasp are all
    None (no path means downstream skips arm overlay and treats the step
    as part-only). The pre-computed duration is preserved through
    `_annotate_durations` via the `simplified=True` flag on each result.
    """
    worker_args = [
        (asset_folder, assembly_dir, sa, gripper_type, gripper_scale,
         float(k_dist), float(k_vol), float(fail_multiplier),
         bool(check_grasp))
        for sa in step_infos
    ]
    if num_proc > 1 and len(worker_args) > 1:
        out = list(parallel_execute(
            _stage1_simplified_worker, worker_args, num_proc=num_proc,
            show_progress=True, desc='arm_pipeline stage 1 (simplified)',
        ))
    else:
        out = [_stage1_simplified_worker(*wa) for wa in worker_args]
    out_by_step = {r['step']: r for r in out if r is not None}
    return [out_by_step.get(i, _stage1_failed(step_infos[i]))
            for i in range(len(step_infos))]


def _stage1_simplified_worker(asset_folder, assembly_dir, sa,
                               gripper_type, gripper_scale,
                               k_dist, k_vol, fail_multiplier,
                               check_grasp=True):
    """Picklable simplified stage-1 worker. See _run_stage1_simplified."""
    i = sa['i']
    part_move = sa['part_move']
    t0 = time()

    base_failed = {
        'step': i,
        'part_move': part_move,
        'feasible': True,            # always True in simplified mode — we
                                     # always have a deterministic cost, so
                                     # the timing layer's median-imputation
                                     # path stays inert.
        'simplified': True,
        'arm_pos': None, 'arm_euler': None,
        'arm_path_full': None,
        'start_q_full': None, 'end_q_full': None,
        'part_path_local': None, 'finger_open_ratio': None,
        'grasp': None,
        'parts_rest': list(sa['parts_rest']),
        'parts_removed': list(sa['parts_removed']),
        'pose': sa.get('pose'),
        'part_centroid_world': sa.get('part_centroid_world'),
        'path_endpoint': None,
    }

    # Part mesh volume + bbox diagonal — both pure mesh properties, fully
    # deterministic and independent of the physics planner. Used below for
    # the closed-form cost (V) and as a path-length proxy (bbox diagonal)
    # when the physics replanner was skipped.
    V = 0.0
    bbox_diag = 0.0
    try:
        from plan_sequence.stable_pose import get_combined_mesh as _gcm
        m = _gcm(assembly_dir, [part_move])
        if m is not None and len(m.vertices) > 0:
            V = float(abs(m.volume))
            extents = np.ptp(np.asarray(m.vertices, dtype=float), axis=0)
            bbox_diag = float(np.linalg.norm(extents))
    except Exception:
        V = 0.0
        bbox_diag = 0.0

    # Cartesian path length:
    #   - When the physics-replanned path is available, sum its cartesian
    #     waypoint distances (faithful to the actual extraction trajectory).
    #   - When it isn't (replan_paths=False, e.g. simplified mode with grasp
    #     check off), fall back to the part's bbox diagonal — a deterministic
    #     mesh-only proxy for "how far does the part have to travel to clear
    #     its neighbours".
    path = None
    d_cart = 0.0
    d_source = 'bbox_diag'
    if sa.get('path') is not None:
        path = [np.array(p, dtype=float) for p in sa['path']]
        for j in range(1, len(path)):
            d_cart += float(np.linalg.norm(path[j][:3] - path[j - 1][:3]))
        d_source = 'physics_path'
    if d_cart <= 0.0:
        d_cart = bbox_diag
        d_source = 'bbox_diag'

    # Rod grasp feasibility: reuse GraspPlanner (no IK/RRT). Returns a list
    # of feasible grasps along the path — non-empty = success. Skipped
    # entirely when `check_grasp` is False; in that case every step is
    # treated as feasible and the bare closed-form cost is returned.
    # Rod grasp feasibility: only meaningful when we actually have a path to
    # check against. When path is None (replan was skipped because grasp
    # checking is also off), we definitionally don't run the check.
    grasp_checked = bool(check_grasp) and (path is not None)
    if grasp_checked:
        grasp_feasible = False
        try:
            from plan_robot.run_grasp_plan import GraspPlanner
            pose = np.array(sa['pose']) if sa.get('pose') is not None else None
            planner = GraspPlanner(
                asset_folder, assembly_dir,
                gripper_type=gripper_type, gripper_scale=gripper_scale,
            )
            grasps = planner.plan(
                part_move, sa['parts_rest'], sa['parts_removed'], pose, path,
                early_terminate=True,
            )
            grasp_feasible = bool(grasps)
        except Exception as e:
            print(f'[arm_pipeline:stage1_simp] step {i} ({part_move}): '
                  f'grasp check raised: {e}; treating as infeasible.', flush=True)
            grasp_feasible = False
    else:
        grasp_feasible = True  # unchecked → cost stays at k_dist·d·(1+k_vol·V)

    cost = k_dist * d_cart * (1.0 + k_vol * V)
    duration_s = cost * (fail_multiplier if not grasp_feasible else 1.0)

    if not grasp_checked:
        status = 'NO-CHECK'
    else:
        status = 'OK' if grasp_feasible else 'FAIL'
    print(f'[arm_pipeline:stage1_simp] step {i} ({part_move}): {status}  '
          f'd={d_cart:.3f} ({d_source})  V={V:.3f}  duration={duration_s:.3f}s  '
          f'({time() - t0:.2f}s)', flush=True)

    base_failed.update({
        'fail_reason': None if grasp_feasible else 'rod_grasp_infeasible',
        'grasp_feasible': grasp_feasible,
        'grasp_checked': grasp_checked,
        'path_distance_cartesian': d_cart,
        'path_distance_source': d_source,
        'part_volume': V,
        'duration_s': duration_s,
        'path_distance_rad': 0.0,
        'path_endpoint': (list(map(float, sa['path'][-1]))
                          if sa.get('path') else None),
    })
    return base_failed


def _stage1_failed(sa, reason='unknown'):
    return {
        'step': sa['i'],
        'part_move': sa['part_move'],
        'feasible': False,
        'fail_reason': reason,
        'grasp': None,
        'arm_pos': None,
        'arm_euler': None,
        'arm_path_full': None,
        'start_q_full': None,
        'end_q_full': None,
        'part_path_local': None,
        'finger_open_ratio': None,
        'parts_rest': list(sa['parts_rest']),
        'parts_removed': list(sa['parts_removed']),
        'pose': sa.get('pose'),
        # Path endpoint is known even when stage 1 fails (path was already
        # replanned in prep). Future-park collision env in stage 2 uses it.
        'path_endpoint': (list(map(float, sa['path'][-1])) if sa.get('path') else None),
    }


def _stage1_worker(asset_folder, assembly_dir, sa,
                   gripper_type, gripper_scale, max_time=None,
                   debug_snapshot_dir=None,
                   shared_base_candidates=None):
    """Picklable worker. Iterate a set of arm base candidates and keep the
    first base that yields a feasible in-grasp trajectory for this step.

    shared_base_candidates: when provided (list of [x, y, z]), all steps in
    the sequence share this single circle of evenly distributed base
    positions — gives the planner sequence-wide base continuity (consecutive
    steps default to the same world-frame base unless geometry forces a
    move). When None the worker falls back to the legacy per-step circle
    centered on the step's own assembly centroid. Orientation (`arm_euler`)
    is always per-step: each candidate base faces the step's current
    centroid so the gripper is pointed where the action is.

    debug_snapshot_dir: if set, a PNG showing the arm at rest next to the
    posed assembly is saved here whenever the step ends up infeasible. Helps
    eyeball whether the arm scale even fits this assembly's geometry.
    """
    def _save_failure_snapshot(reason):
        if not debug_snapshot_dir:
            return
        try:
            from plan_robot.render_grasp_arm import render_arm_next_to_assembly
            import os as _os
            _os.makedirs(debug_snapshot_dir, exist_ok=True)
            arm_asset_folder = _os.path.join(project_base_dir, 'assets')
            pose_arr = np.array(sa['pose']) if sa.get('pose') is not None else None
            out = _os.path.join(debug_snapshot_dir, f'step_{sa["i"]}_{sa["part_move"]}_{reason}.png')
            saved = render_arm_next_to_assembly(
                arm_asset_folder, assembly_dir, sa['part_move'],
                sa['parts_rest'], sa['parts_removed'],
                pose_arr, gripper_type, gripper_scale, save_path=out,
            )
            if saved is not None:
                print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                      f'debug snapshot → {saved}', flush=True)
        except Exception as _e:
            print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                  f'debug snapshot failed: {_e}', flush=True)

    t0 = time()
    print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): start'
          f'{f" (timeout={max_time:.1f}s)" if max_time is not None else ""}', flush=True)
    try:
        from plan_robot.run_grasp_arm_plan import GraspArmPlanner

        if sa.get('path') is None:
            print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                  f'skipping — no path from prep stage', flush=True)
            _save_failure_snapshot('path_replan_failed')
            return _stage1_failed(sa, reason='path_replan_failed')

        path = [np.array(p, dtype=float) for p in sa['path']]
        pose = np.array(sa['pose']) if sa['pose'] is not None else None

        # Per-step centroid drives the orientation only; base positions come
        # from the sequence-wide shared circle when provided.
        center = _compute_step_center(assembly_dir, sa)
        if shared_base_candidates is not None:
            arm_pos_candidates = [np.asarray(p, dtype=float)
                                  for p in shared_base_candidates]
        else:
            sample_gripper_pos = center + np.array([0.0, 0.0, 5.0 * max(0.1, gripper_scale)])
            sample_gripper_ori = np.array([0.0, 0.0, 1.0])
            arm_pos_candidates = get_arm_pos_candidates(
                sample_gripper_pos, sample_gripper_ori, gripper_scale, center=center,
            )

        planner = GraspArmPlanner(
            asset_folder, assembly_dir,
            gripper_type=gripper_type, gripper_scale=gripper_scale,
        )

        # Try each base candidate in order. The first to produce a feasible
        # plan wins. Per-step budget covers the WHOLE candidate sweep (not
        # per-candidate) — once any candidate succeeds we exit, and a global
        # timeout still bounds the worst case.
        chosen = None
        for cand_idx, arm_pos in enumerate(arm_pos_candidates):
            if max_time is not None and (time() - t0) > max_time:
                print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                      f'timeout after {cand_idx} base candidate(s) '
                      f'({time()-t0:.1f}s > {max_time:.1f}s)', flush=True)
                _save_failure_snapshot('timeout')
                return _stage1_failed(sa, reason='timeout')
            arm_euler = get_arm_euler(arm_pos.copy(), center=center)
            remaining = None if max_time is None else max(0.5, max_time - (time() - t0))
            result = _plan_one_step_fixed_base(
                planner, sa['part_move'], sa['parts_rest'], sa['parts_removed'],
                pose, path, arm_pos, arm_euler,
                step_id=sa['i'], part_id=sa['part_move'], max_time=remaining,
                base_tag=f'base {cand_idx+1}/{len(arm_pos_candidates)}',
            )
            if result == 'timeout':
                print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                      f'FAIL — timeout at base {cand_idx+1}/{len(arm_pos_candidates)} ({time()-t0:.1f}s)',
                      flush=True)
                _save_failure_snapshot('timeout')
                return _stage1_failed(sa, reason='timeout')
            if result is not None:
                chosen = (arm_pos, arm_euler, cand_idx, result)
                break

        if chosen is None:
            print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
                  f'FAIL — no_feasible_grasp on any of {len(arm_pos_candidates)} base(s) '
                  f'({time()-t0:.1f}s)', flush=True)
            _save_failure_snapshot('no_feasible_grasp')
            return _stage1_failed(sa, reason='no_feasible_grasp')

        arm_pos, arm_euler, cand_idx, result = chosen
        endpoint = path[-1].tolist()
        print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}): '
              f'OK — base {cand_idx+1}/{len(arm_pos_candidates)}, '
              f'grasp {result["grasp_idx"]+1}/{result["n_grasps"]} survived, '
              f'arm_path_len={len(result["arm_path_full"])} ({time()-t0:.1f}s)', flush=True)
        return {
            'step': sa['i'],
            'part_move': sa['part_move'],
            'feasible': True,
            'fail_reason': None,
            'arm_pos': arm_pos.tolist(),
            'arm_euler': arm_euler.tolist(),
            'grasp': {
                'pos': result['grasp_pos'],
                'quat': result['grasp_quat'],
                'open_ratio': result['open_ratio'],
            },
            'arm_path_full': result['arm_path_full'],
            'start_q_full': result['arm_path_full'][0],
            'end_q_full': result['arm_path_full'][-1],
            'part_path_local': result['part_path_local'],
            'finger_open_ratio': result['open_ratio'],
            'parts_rest': list(sa['parts_rest']),
            'parts_removed': list(sa['parts_removed']),
            'pose': sa['pose'],
            'path_endpoint': endpoint,
        }
    except Exception as e:
        print(f'[arm_pipeline:stage1:worker] step {sa["i"]} ({sa["part_move"]}) '
              f'exception after {time()-t0:.1f}s: {e}', flush=True)
        print(traceback.format_exc(), flush=True)
        try:
            _save_failure_snapshot('exception')
        except Exception:
            pass
        return _stage1_failed(sa, reason=f'exception: {e}')


def _plan_one_step_fixed_base(planner, move_id, still_ids, removed_ids,
                              pose, path, arm_pos, arm_euler,
                              step_id=None, part_id=None, max_time=None,
                              base_tag=''):
    """Inline replica of GraspArmPlanner.plan() restricted to ONE arm base.
    Returns a dict with grasp + arm_path on success, or None.

    step_id / part_id are passed through purely for log lines.
    base_tag is an optional label (e.g. "base 2/4") appended to logs when
    multiple base candidates are being tried for one step.

    max_time: optional wall-clock budget (seconds). Checked between grasp
    candidates; bails out as soon as the budget is exhausted.
    """
    from assets.transform import get_transform_from_path
    from plan_robot.geometry import transform_part_meshes
    import time as _time

    if step_id is not None:
        tag = f'step {step_id} ({part_id})'
        if base_tag:
            tag += f' [{base_tag}]'
    else:
        tag = 'step ?'
    _t_budget_start = _time.time()

    part_meshes_final = transform_part_meshes(
        planner.part_meshes, planner.part_pos_dict, planner.part_quat_dict, pose,
    )
    move_mesh_final = part_meshes_final[move_id]
    still_mesh_final = trimesh.util.concatenate(
        [part_meshes_final[sid] for sid in still_ids]
    )

    part_transforms = get_transform_from_path(path, n_sample=None)
    part_rel_transforms = [T @ np.linalg.inv(part_transforms[0]) for T in part_transforms]

    disassembly_direction = np.asarray(path[-1], dtype=float)[:3] - np.asarray(path[0], dtype=float)[:3]
    if np.linalg.norm(disassembly_direction) < 1e-9:
        disassembly_direction = np.array([0.0, 0.0, 1.0])
    else:
        disassembly_direction /= np.linalg.norm(disassembly_direction)

    grasps_final = planner.generate_grasps(move_mesh_final, disassembly_direction)
    planner.center = trimesh.util.concatenate(list(part_meshes_final.values())).centroid.copy()
    planner.center[2] = 0.0

    n_grasps = len(grasps_final)
    if n_grasps == 0:
        print(f'[arm_pipeline:stage1:worker] {tag}: 0 grasps generated; '
              f'gripper may not fit any antipodal pair on the part mesh.', flush=True)
        return None

    # Buckets for diagnosing why grasps were rejected.
    fail_counts = {'grasp_collision': 0, 'arm_ik_or_collision': 0, 'finished_partial': 0, 'timeout': 0}

    for grasp_idx, grasp in enumerate(grasps_final):
        if max_time is not None and (_time.time() - _t_budget_start) > max_time:
            print(f'[arm_pipeline:stage1:worker] {tag}: timeout after grasp '
                  f'{grasp_idx}/{n_grasps} ({_time.time() - _t_budget_start:.1f}s > {max_time:.1f}s budget). '
                  f'breakdown so far: grasp_collision={fail_counts["grasp_collision"]}  '
                  f'arm_ik_or_collision={fail_counts["arm_ik_or_collision"]}', flush=True)
            return 'timeout'  # sentinel — caller maps to fail_reason='timeout'
        grasps_t = []
        arm_q_default = None
        n_waypoints_ok = 0
        fail_kind = None
        for part_transform in part_rel_transforms:
            grasp_t = planner.transform_grasp(grasp, part_transform)
            move_mesh_t = move_mesh_final.copy()
            move_mesh_t.apply_transform(part_transform)

            if not planner.check_grasp_feasible(grasp_t, move_mesh_t, still_mesh_final):
                fail_kind = 'grasp_collision'
                break
            if not planner.check_arm_feasible(grasp_t, move_mesh_t, still_mesh_final,
                                              arm_pos, arm_q_default):
                fail_kind = 'arm_ik_or_collision'
                break
            grasps_t.append(grasp_t)
            arm_q_default = grasp_t.arm_q  # within-step IK seed
            n_waypoints_ok += 1

        if fail_kind is None and grasps_t:
            return {
                'grasp_pos': list(map(float, grasps_t[0].pos)),
                'grasp_quat': list(map(float, grasps_t[0].quat)),
                'open_ratio': float(grasps_t[0].open_ratio),
                'arm_path_full': [list(map(float, g.arm_q)) for g in grasps_t],
                'part_path_local': [list(map(float, p)) for p in path],
                'grasp_idx': grasp_idx,
                'n_grasps': n_grasps,
            }
        if fail_kind is not None:
            fail_counts[fail_kind] += 1
        elif not grasps_t:
            fail_counts['finished_partial'] += 1

    # All grasps exhausted without success — emit the breakdown.
    print(f'[arm_pipeline:stage1:worker] {tag}: tried {n_grasps} grasp(s), all failed.  '
          f'breakdown: grasp_collision={fail_counts["grasp_collision"]}  '
          f'arm_ik_or_collision={fail_counts["arm_ik_or_collision"]}  '
          f'other={fail_counts["finished_partial"]}', flush=True)
    return None


# ---------------------------------------------------------------------------
# Stage 2 — per-transition planning
# ---------------------------------------------------------------------------


def _run_stage2(asset_folder, assembly_dir, step_infos, stage1_results,
                gripper_type, gripper_scale,
                num_proc, debug=False, max_time=None):
    """Build inter-step transitions and run RRT-Connect in parallel.

    Three transition kinds, all planned at the DESTINATION step's base
    (the arm's joint config is preserved across the base teleport; only
    the base changes between steps with no joint-space cost):

      prefix  rest_q -> step_0.start_q          at base_0          (to_step=0)
      inter   step_k.end_q -> step_{k+1}.start_q at base_{k+1}     (to_step=k+1)
      suffix  step_{N-1}.end_q -> rest_q        at base_{N-1}      (to_step=-1)

    No per-step reach/retreat — between consecutive disassembly steps the
    arm joint config flows directly from one step's end to the next step's
    start. The base teleports freely (no joint cost). The transition's
    arm_path_full length is the "cost of moving between steps" used by
    downstream time-estimation logic.

    Static collision envs:
      prefix:   assembly at step 0's start pose (all parts present).
      inter:    assembly at step k+1's start pose (parts_rest_{k+1} +
                part_move_{k+1}) + parts 0..k parked at endpoints.
      suffix:   assembly at step N-1's end state (parts_rest_{N-1} only) +
                parts 0..N-1 parked at endpoints.
    """
    rest_q_full = [0.0] + list(get_default_arm_rest_q())

    # Cumulative "previously-removed" endpoints across the sequence. Tracked
    # from step_infos paths (independent of stage 1 success), so transitions
    # always see earlier-removed parts at their parked positions.
    endpoints_so_far = []
    cumulative_endpoints = []  # cumulative_endpoints[k] = parked BEFORE step k
    for sa in step_infos:
        cumulative_endpoints.append(list(endpoints_so_far))
        path = sa.get('path')
        if path:
            endpoints_so_far.append((sa['part_move'], list(map(float, path[-1]))))
    endpoints_after_all = list(endpoints_so_far)  # parked AFTER the final step

    transitions_specs = []

    # Prefix: rest -> step_0.start_q at base_0.
    s0 = stage1_results[0] if stage1_results else None
    if s0 and s0.get('feasible'):
        transitions_specs.append({
            'from_step': -1, 'to_step': 0,
            'kind': 'prefix',
            'start_q_full': rest_q_full,
            'goal_q_full': s0['start_q_full'],
            'env_step': 0,
            'env_endpoints': list(cumulative_endpoints[0]),  # empty (no parts removed yet)
            'env_include_move_in_assembly': True,
            'arm_pos': s0['arm_pos'], 'arm_euler': s0['arm_euler'],
        })

    # Inter: step_k.end_q -> step_{k+1}.start_q at base_{k+1}.
    for k in range(len(stage1_results) - 1):
        sk = stage1_results[k]
        sk1 = stage1_results[k + 1]
        if not (sk and sk1 and sk.get('feasible') and sk1.get('feasible')):
            continue
        transitions_specs.append({
            'from_step': k, 'to_step': k + 1,
            'kind': 'inter',
            'start_q_full': sk['end_q_full'],
            'goal_q_full': sk1['start_q_full'],
            'env_step': k + 1,
            'env_endpoints': list(cumulative_endpoints[k + 1]),  # parts 0..k parked
            'env_include_move_in_assembly': True,  # part k+1 still in assembly
            'arm_pos': sk1['arm_pos'], 'arm_euler': sk1['arm_euler'],
        })

    # Suffix: step_{N-1}.end_q -> rest at base_{N-1}.
    sN = stage1_results[-1] if stage1_results else None
    if sN and sN.get('feasible'):
        transitions_specs.append({
            'from_step': len(stage1_results) - 1, 'to_step': -1,
            'kind': 'suffix',
            'start_q_full': sN['end_q_full'],
            'goal_q_full': rest_q_full,
            'env_step': len(stage1_results) - 1,
            'env_endpoints': list(endpoints_after_all),  # everything parked
            'env_include_move_in_assembly': False,       # last part has been removed
            'arm_pos': sN['arm_pos'], 'arm_euler': sN['arm_euler'],
        })

    if not transitions_specs:
        return []

    worker_args = [
        (asset_folder, assembly_dir, spec, step_infos,
         gripper_type, gripper_scale, max_time)
        for spec in transitions_specs
    ]
    if num_proc > 1 and len(worker_args) > 1:
        results = list(parallel_execute(
            _stage2_worker, worker_args, num_proc=num_proc,
            show_progress=True, desc='arm_pipeline stage 2',
        ))
    else:
        results = [_stage2_worker(*wa) for wa in worker_args]

    return [r for r in results if r is not None]


def _stage2_worker(asset_folder, assembly_dir, spec, step_infos,
                   gripper_type, gripper_scale, max_time=None):
    f_label = 'rest' if spec['from_step'] == -1 else f'step {spec["from_step"]}'
    to_label = 'rest' if spec['to_step'] == -1 else f'step {spec["to_step"]}'
    kind = spec.get('kind', '?')
    tag = f'{kind} {f_label} → {to_label}'
    t0 = time()
    print(f'[arm_pipeline:stage2:worker] {tag}: start'
          f'{f" (timeout={max_time:.1f}s)" if max_time is not None else ""}', flush=True)
    try:
        from plan_robot.motion_plan_arm import ArmMotionPlanner

        arm_pos = np.array(spec['arm_pos'], dtype=float)
        arm_euler = np.array(spec['arm_euler'], dtype=float)
        start_q_full = np.array(spec['start_q_full'], dtype=float)
        goal_q_full = np.array(spec['goal_q_full'], dtype=float)
        start_active = start_q_full[1:].tolist()
        goal_active = goal_q_full[1:].tolist()

        # Static collision env: assembly at THIS step's pose. For reach, the
        # moving part is still in the assembly; for retreat, the move part
        # has just been removed and lives in env_endpoints at its disassembly
        # endpoint (caller already set this up).
        env_step = spec['env_step']
        env_pose = step_infos[env_step]['pose']
        env_pose = np.array(env_pose) if env_pose is not None else None
        still_meshes = _build_env_meshes(
            assembly_dir, step_infos, env_step, spec['env_endpoints'], env_pose,
            include_move_in_assembly=spec.get('env_include_move_in_assembly', True),
        )

        joint_delta = float(np.linalg.norm(np.array(goal_active) - np.array(start_active)))
        print(f'[arm_pipeline:stage2:worker] {tag}: env_step={env_step}  '
              f'still_meshes={len(still_meshes)}  parked={len(spec["env_endpoints"])}  '
              f'joint_delta={joint_delta:.2f} rad', flush=True)

        motion_planner = ArmMotionPlanner(
            base_pos=arm_pos, base_euler=arm_euler,
            scale=gripper_scale, gripper_type=gripper_type,
        )
        path_full = motion_planner.plan_with_grasp(
            start=start_active, goal=goal_active,
            move_mesh=None, move_transform=None,
            still_meshes=still_meshes, open_ratio=1.0,
            verbose=False, max_time=max_time,
        )
        if path_full is None:
            print(f'[arm_pipeline:stage2:worker] {tag}: FAIL — rrt_no_path '
                  f'({time()-t0:.1f}s)', flush=True)
            return {
                'from_step': spec['from_step'], 'to_step': spec['to_step'],
                'kind': spec.get('kind'),
                'feasible': False, 'fail_reason': 'rrt_no_path',
                'arm_path_full': None,
            }
        print(f'[arm_pipeline:stage2:worker] {tag}: OK — '
              f'arm_path_len={len(path_full)} ({time()-t0:.1f}s)', flush=True)
        return {
            'from_step': spec['from_step'], 'to_step': spec['to_step'],
            'kind': spec.get('kind'),
            'feasible': True, 'fail_reason': None,
            'arm_path_full': [list(map(float, q)) for q in path_full],
        }
    except Exception as e:
        print(f'[arm_pipeline:stage2:worker] {tag}: exception after {time()-t0:.1f}s: {e}', flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            'from_step': spec['from_step'], 'to_step': spec['to_step'],
            'kind': spec.get('kind'),
            'feasible': False, 'fail_reason': f'exception: {e}',
            'arm_path_full': None,
        }


def _build_env_meshes(assembly_dir, step_infos, env_step, env_endpoints, env_pose,
                      include_move_in_assembly=True):
    """Static collision meshes for stage 2 transition collision avoidance:
       - parts present in the assembly at env_step posed by env_pose,
       - plus previously-removed parts placed at their endpoint qm.

    include_move_in_assembly: when True (reach phase), the env_step's
    part_move is placed in the assembly. When False (retreat phase), the
    part_move is excluded — the caller has already added it to env_endpoints
    at its disassembly endpoint."""
    pose_arr = np.eye(4) if env_pose is None else np.asarray(env_pose, dtype=float)
    assembly = load_assembly_all_transformed(assembly_dir)
    meshes = []

    si = step_infos[env_step] if 0 <= env_step < len(step_infos) else step_infos[-1]
    in_assembly = list(si['parts_rest'])
    if include_move_in_assembly:
        in_assembly.append(si['part_move'])
    for pid in in_assembly:
        if pid not in assembly:
            continue
        m = assembly[pid].get('mesh_final')
        if m is None:
            continue
        m = m.copy()
        m.apply_transform(pose_arr)
        meshes.append(m)

    for pid, endpoint_qm in env_endpoints or []:
        if pid not in assembly or endpoint_qm is None:
            continue
        m = assembly[pid].get('mesh_final')
        if m is None:
            continue
        m = m.copy()
        # Build the world-frame transform from the qm (free3d-exp state).
        T = get_transform_matrix_euler(endpoint_qm[:3], endpoint_qm[3:])
        # qm describes the part's world pose; mesh_final is already in the
        # assembled frame, so we apply pose_arr first then the parking T.
        # Subtle but matches what create_gripper_arm_with_assembly_posed_xml
        # does: removed parts go to pos_dict_initial. We instead use the
        # endpoint, which is what the disassembly actually produced.
        m.apply_transform(T)
        meshes.append(m)

    return meshes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_plan(gripper_type, gripper_scale):
    return {
        'gripper_type': gripper_type,
        'gripper_scale': gripper_scale,
        'arm_pos': [0.0, 0.0, 0.0],
        'arm_euler': [0.0, 0.0, 0.0],
        'rest_q_active': list(get_default_arm_rest_q()),
        'sequence': [],
        'steps': [],
        'transitions': [],
        'time_estimate': {
            'velocity_rad_s': 1.0,
            'steps_total_s': 0.0,
            'transitions_total_s': 0.0,
            'total_s': 0.0,
        },
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f'{type(obj).__name__} is not JSON serializable')
