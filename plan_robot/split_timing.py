"""Time a recursive subassembly plan the way it is carried out.

The flat timing (arm_pipeline.plan_arm_sequence) walks stats['sequence'], a
path of the planning tree on which every part comes off the whole remaining
assembly. A subassembly plan (plan_sequence/optimizer/split_plan.py) is carried
out differently: its prefix parts come off the whole, then R is lifted off S as
one rigid body (a 'join', which is no tree edge), then S is taken apart on its
own, then R on its own, each recursively. Most of those states never occur in
the tree -- removing single parts never reaches "S present, R gone" -- so the
tree has nothing to time them with. This module resolves every step in the
context it actually happens in and times the step list with the same
simplified model (arm_pipeline.time_steps_simplified):

  remove  the planner's own candidate check (_simulate_standalone) on the body
          present at that moment only, over the body's candidate poses (as
          the DFA planner generates them: its stable poses, closest to its
          current orientation first, the current orientation itself tried
          first), taking the first feasible pose the way the planner picks
          one. The checks go through the candidate-check cache when one is
          set, so states the planner already checked are replayed. A step
          with no feasible pose makes the plan infeasible as told (e.g. a
          half that does not stand on its own, which the plan's construction
          never checked), and the timing says so instead of pricing it.
  join    R pulled straight off S along the verified separation direction
          until it clears S by the separation margin, priced like a part
          extraction with R's volume. The body keeps its orientation.

Reorientation is measured from the orientation the body had before the step;
for the first step of an R block that is R's orientation at the join (R was
set aside, not turned along with S). The last part of every block is not a
step: nothing is left to take it from.

Two totals come out of the same steps:
  sequential  one worker does everything in plan order (the 'totals', the
              same model as the flat timing);
  parallel    S and R of every split are taken apart at the same time by
              separate workers: a split block takes prefix + join +
              max(S, R), recursively (up to 4 concurrent workers at the
              default depth 2). Step times are the sequential ones, except
              that the first step of an R block travels from the join (where
              R's worker picks it up) rather than from wherever S ended.
"""
import json
from pathlib import Path

import numpy as np


def _plan_walk(plan, root_order):
    """Yield the plan's steps in execution order: ('remove', part, body, path)
    with the body present before the removal, or ('join', block, body, path)
    for a split -- `path` being the block the step belongs to -- plus
    ('enter', side) markers so the caller knows when an R block starts.
    Bodies are lists in `root_order` (the planner's part order, so candidate
    checks key the same way as the planner's)."""
    rank = {p: i for i, p in enumerate(root_order)}

    def ordered(parts):
        return sorted(parts, key=lambda p: rank[p])

    def walk(block, body):
        body = ordered(body)
        path = list(block['path'])
        if block['kind'] == 'split':
            for p in block['prefix']:
                yield ('remove', p, list(body), path)
                body.remove(p)
            yield ('join', block, list(body), path)
            yield ('enter', 'S')
            yield from walk(block['S'], block['split']['S'])
            yield ('enter', 'R')
            yield from walk(block['R'], block['split']['R'])
        else:
            for p in block['sequence']:
                if len(body) <= 1:
                    break
                yield ('remove', p, list(body), path)
                body.remove(p)

    yield from walk(plan, plan['parts'])


def resolve_split_steps(asset_folder, assembly_dir, plan, split_steps, *, max_poses,
                        max_grippers, get_dof, skip_stability, allow_gap,
                        ignore_unstable, optimizer='L-BFGS-B', num_proc=1,
                        sim_cache_dir=None, save_sdf=True):
    """Resolve every step of `plan` in its own context (see module docstring).

    Returns (steps, failure): `steps` in the format
    arm_pipeline.time_steps_simplified takes; `failure` is None, or a dict
    naming the first removal that is infeasible in its context (then `steps`
    holds what was resolved before it)."""
    from assets.load import load_part_ids
    from plan_sequence.planner.base import SequencePlanner, _simulate_standalone_tagged
    from plan_sequence.planner.dfa import candidate_poses
    from plan_sequence.stable_pose import get_combined_mesh
    from utils.parallel import parallel_execute

    root_order = sorted(load_part_ids(assembly_dir))
    ignore_unstable = tuple(sorted(str(p) for p in (ignore_unstable or ())))
    cache = None
    if sim_cache_dir is not None:
        from plan_sequence.planner.sim_cache import SimCache
        # Same fields as DFASequencePlanner.plan, so the planner's records hit.
        cache = SimCache(sim_cache_dir, assembly_dir, base_part=None,
                         max_grippers=max_grippers, optimizer=optimizer,
                         allow_gap=bool(allow_gap), get_dof=bool(get_dof),
                         skip_stability=bool(skip_stability),
                         ignore_unstable=ignore_unstable)

    local_centroid = {}

    def centroid_world(parts, pose):
        key = tuple(parts)
        if key not in local_centroid:
            local_centroid[key] = np.asarray(
                get_combined_mesh(assembly_dir, list(parts)).vertices, dtype=float).mean(axis=0)
        pose = np.asarray(pose, dtype=float) if pose is not None else np.eye(4)
        c = pose[:3, :3] @ local_centroid[key] + pose[:3, 3]
        c[2] = 0.0
        return c.tolist()

    joins = iter([e for e in split_steps if e['kind'] == 'join'])
    steps = []
    pose = None             # orientation of the body currently being taken apart
    join_poses = []         # orientation at each open join, innermost last
    join_centroids = []     # where R's worker picks R up, innermost last
    starts_r_block = None   # set on entering an R block, consumed by its first step
    failure = None
    try:
        for item in _plan_walk(plan, root_order):
            if item[0] == 'enter':
                # S continues from the join's orientation; R starts from it too
                # (it was set aside at the join, not turned along with S).
                if item[1] == 'R':
                    pose = join_poses.pop()
                    starts_r_block = join_centroids.pop()
                continue
            if item[0] == 'join':
                block, body, path = item[1], item[2], item[3]
                entry = next(joins)
                if pose is None:
                    # A join first: the whole assembly in its most probable
                    # resting orientation.
                    pose = candidate_poses(assembly_dir, body, None, max_poses)[0]
                join_poses.append(pose)
                S, R = list(block['split']['S']), list(block['split']['R'])
                step = {
                    'kind': 'join', 'label': f'join({",".join(R)} | {",".join(S)})',
                    'S': S, 'R': R, 'direction': entry.get('direction'),
                    'block': path, 'depth': entry.get('depth'),
                    'pose': None if pose is None else np.asarray(pose).tolist(),
                    'reorient_from_pose': None if pose is None else np.asarray(pose).tolist(),
                    'parts_fix': [],
                    'part_centroid_world': centroid_world(R, pose),
                }
                if starts_r_block is not None:
                    step['parallel_from_centroid'] = starts_r_block
                    starts_r_block = None
                join_centroids.append(step['part_centroid_world'])
                steps.append(step)
                continue

            _, part, body, path = item
            rest = [p for p in body if p != part]
            removed = [p for p in root_order if p not in body]
            candidates = candidate_poses(assembly_dir, body, pose, max_poses)
            tasks = [(part, rest, removed, cp) for cp in candidates]
            keys = [cache.key(part, rest, cp) if cache is not None else None for cp in candidates]
            results, pending = {}, []
            for k, (pm, pr, rm, cp) in enumerate(tasks):
                hit = cache.get(keys[k]) if cache is not None else None
                if hit is not None:
                    results[k] = hit
                    continue
                pending.append((
                    asset_folder, assembly_dir, save_sdf, None, pm, pr, rm, cp,
                    max_grippers, None, optimizer, 0, False, allow_gap, get_dof,
                    None, skip_stability, ignore_unstable, (0, k),
                ))
            for sim_info, arg in parallel_execute(
                _simulate_standalone_tagged, pending, num_proc,
                show_progress=False, return_args=True,
            ):
                k = arg[-1][1]
                sim_info = {key: v for key, v in sim_info.items() if not key.startswith('_')}
                if cache is not None:
                    cache.put(keys[k], sim_info)
                results[k] = sim_info
            # The DFA planner's pick: feasible first, then closest to the
            # current orientation, then candidate order.
            ranked = sorted(range(len(tasks)), key=lambda k: (
                not results[k].get('feasible', False),
                SequencePlanner._rotation_angle_between(pose, results[k].get('pose')),
                k))
            best = results[ranked[0]] if ranked else None
            if best is None or not best.get('feasible'):
                reasons = sorted({results[k].get('fail_reason') for k in results})
                failure = {'part': part, 'body': body, 'fail_reasons': reasons,
                           'step': len(steps)}
                print(f'[split_timing] removing {part} from {body} is infeasible '
                      f'in every candidate pose ({reasons}); the plan cannot be carried '
                      f'out as told')
                break
            chosen = best.get('pose')
            step = {
                'kind': 'remove', 'label': part, 'part_move': part, 'block': path,
                'parts_rest': rest, 'parts_removed': removed,
                'action': np.asarray(best['action']).tolist(),
                'pose': None if chosen is None else np.asarray(chosen).tolist(),
                'reorient_from_pose': None if pose is None else np.asarray(pose).tolist(),
                'parts_fix': list(best['parts_fix']) if best.get('parts_fix') is not None else None,
                'part_centroid_world': centroid_world([part], chosen),
            }
            if starts_r_block is not None:
                step['parallel_from_centroid'] = starts_r_block
                starts_r_block = None
            steps.append(step)
            pose = chosen
    finally:
        if cache is not None:
            cache.close()
    # The first step turns nothing; 'reorient_from_pose' None would read as
    # "unknown" and make the angle 0 anyway, but say so explicitly.
    if steps:
        steps[0].pop('reorient_from_pose', None)
    return steps, failure


def parallel_makespan(plan, steps, overview, workers=None):
    """Time of the plan when S and R of a split are taken apart at once by
    separate workers: a split block takes its own steps (prefix + join) plus
    max(S, R). Step times are the sequential ones from `overview`, except that
    the first step of an R block taken apart in parallel travels from the join
    (its 'parallel_from_centroid') instead of from where S ended.

    `workers`: how many there are. None = as many as the plan can use (every
    split, at every depth, in parallel). With k, a split block's workers are
    shared out between its halves (ceil(k/2) to S, the rest to R), and a half
    left with one worker is taken apart sequentially, nested splits included:
    2 workers take S and R of the outermost split at once and nothing else.
    Returns {'total_s', 'per_block_s', 'workers'}."""
    from plan_robot.arm_pipeline import _arc_travel

    per_step = overview['per_step']
    center = overview.get('assembly_center') or [0.0, 0.0, 0.0]
    velocity = max(float(overview.get('base_travel_velocity', 1.0)), 1e-9)
    seq_time, par_time = [], []
    for st, entry in zip(steps, per_step):
        t = float(entry['total_s'])
        seq_time.append(t)
        if 'parallel_from_centroid' in st and st.get('part_centroid_world') is not None:
            arc, _ = _arc_travel(st['parallel_from_centroid'], st['part_centroid_world'], center)
            t += arc / velocity - float(entry['base_travel_s'])
        par_time.append(t)

    per_block = {}

    def own_steps(block, parallel_start):
        # A block's own steps (prefix, join); its first step travels from the
        # join only when it is an R half started by a worker of its own.
        times = par_time if parallel_start else seq_time
        return sum(t for st, t in zip(steps, times) if st.get('block') == list(block['path']))

    def block_time(block, k, parallel_start=False):
        own = own_steps(block, parallel_start)
        if block['kind'] == 'split':
            if k is None or k >= 2:
                k_s = None if k is None else (k + 1) // 2
                k_r = None if k is None else k // 2
                own += max(block_time(block['S'], k_s), block_time(block['R'], k_r, True))
            else:
                own += block_time(block['S'], 1) + block_time(block['R'], 1)
        per_block['.'.join(block['path']) or 'root'] = own
        return own

    return {'total_s': block_time(plan, workers), 'per_block_s': per_block, 'workers': workers}


def split_time_for(split_overview, workers):
    """The subassembly plan's time with `workers` (1 = sequential totals,
    2 = 'parallel_2', None = 'parallel', every split at once); None when the
    overview has no usable time for that."""
    if not split_overview or split_overview.get('status') != 'ok':
        return None
    if workers == 1:
        return (split_overview.get('totals') or {}).get('total_s')
    key = 'parallel_2' if workers == 2 else 'parallel'
    return (split_overview.get(key) or {}).get('total_s')


def choose_split(flat_overview, split_overview, workers=1, only_if_faster=True):
    """Whether to carry out the subassembly plan or the flat sequence: the
    plan when it is timed and (with `only_if_faster`) predicted faster with
    `workers` than the flat sequence. Returns {'adopt', 'workers', 'flat_s',
    'split_s', 'reason'}."""
    flat_s = ((flat_overview or {}).get('totals') or {}).get('total_s')
    split_s = split_time_for(split_overview, workers)
    if not split_overview:
        reason, adopt = 'no subassembly plan', False
    elif split_overview.get('status') != 'ok':
        reason, adopt = 'plan cannot be carried out as told', False
    elif split_s is None:
        reason, adopt = f'no time for {workers} worker(s)', False
    elif not only_if_faster:
        reason, adopt = 'adopted without comparison', True
    elif flat_s is None:
        reason, adopt = 'flat sequence untimed', True
    elif split_s < flat_s:
        reason, adopt = 'faster than the flat sequence', True
    else:
        reason, adopt = 'not faster than the flat sequence', False
    return {'adopt': adopt, 'workers': workers, 'flat_s': flat_s, 'split_s': split_s,
            'reason': reason}


def time_split_plan(planner_asset_folder, arm_asset_folder, assembly_dir, stats, setup,
                    log_dir=None, num_proc=1, sim_cache_dir=None, allow_gap=False,
                    gripper_type='rod', gripper_scale=0.4):
    """Resolve and time `stats['split_plan']`; writes
    `<log_dir>/timing_overview_split.json` and returns it. Its 'status' is
    'ok' (sequential totals as in timing_overview.json, per_step entries
    tagged 'remove' or 'join', and 'parallel': {'total_s', 'per_block_s'}; see
    the module docstring) or 'infeasible' (no totals; 'failure' names the
    step). None when the stats hold no subassembly plan."""
    import settings
    from plan_robot.arm_pipeline import time_steps_simplified

    plan = stats.get('split_plan')
    if not plan:
        return None
    steps, failure = resolve_split_steps(
        planner_asset_folder, assembly_dir, plan, stats.get('split_steps') or [],
        max_poses=int(setup.get('max_poses', 3)),
        max_grippers=setup.get('max_grippers', 3),
        get_dof=bool(getattr(settings, 'get_dof', False)),
        skip_stability=bool(getattr(settings, 'skip_stability', False)),
        allow_gap=allow_gap,
        ignore_unstable=stats.get('ignored_unstable_parts') or (),
        optimizer=setup.get('optimizer', 'L-BFGS-B'),
        num_proc=num_proc, sim_cache_dir=sim_cache_dir,
    )
    n_joins = sum(1 for s in steps if s['kind'] == 'join')
    if failure is not None:
        overview = {'status': 'infeasible', 'failure': failure, 'n_steps_resolved': len(steps)}
    else:
        overview = time_steps_simplified(arm_asset_folder, assembly_dir, steps,
                                         gripper_type, gripper_scale, num_proc=num_proc)
        overview['parallel'] = parallel_makespan(plan, steps, overview)
        overview['parallel_2'] = parallel_makespan(plan, steps, overview, workers=2)
        # Each removal's pull (world frame), for the plan's pull-direction
        # metric; a join's direction is recorded in another frame, so not here.
        for st, entry in zip(steps, overview.get('per_step') or []):
            if st.get('kind') == 'remove' and st.get('action') is not None:
                entry['action'] = [float(x) for x in st['action']]
        overview['status'] = 'ok'
    overview['n_joins'] = n_joins
    overview['split_sequence_source'] = stats.get('split_sequence_source')
    if log_dir is not None:
        out = Path(log_dir) / 'timing_overview_split.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(overview, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print(f'[split_timing] {overview["status"]}: '
              + (f'total={overview["totals"]["total_s"]:.2f}s sequential, '
                 f'{overview["parallel"]["total_s"]:.2f}s parallel, over {len(steps)} steps '
                 f'({n_joins} join(s))' if failure is None else f'failure {failure}')
              + f'  -> {out}', flush=True)
    return overview
