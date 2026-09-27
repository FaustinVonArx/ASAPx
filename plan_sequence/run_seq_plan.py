import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import json
import traceback
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) # for trimesh stable pose computation

from plan_sequence.generator import generators
from plan_sequence.planner import planners
from assets.save import clear_saved_sdfs


def _dump_failure_evidence(tree, asset_folder, assembly_dir, base_part, tools, save_sdf, log_dir, render=True):
    '''Walk the deepest reached node(s) of `tree`, dispatch on each failed
    outgoing edge's `fail_reason`, and (when `render` is True) dump GIF/PNG
    evidence into `<log_dir>/failures/`. Always writes `<log_dir>/failures.json`
    summarising deepest_nodes, depth and per-failure entries with extra IDs
    and image paths.

    render: when False, skip the actual GIF/PNG rendering for each failed
    child but still produce the metadata file (entry['evidence_image']
    remains None). Controlled by `settings.render_sequence` at the call
    site so a global render-off run doesn't also render the "remaining"
    parts' failure evidence.

    Returns the failures payload (dict) or None when there is no feasible
    progress past the root.'''
    from plan_sequence.planner.base import SequencePlanner
    from plan_sequence.feasibility_check import check_stable, check_tool
    from plan_sequence.physics_planner import MultiPartPathPlanner

    deepest = SequencePlanner.find_deepest_reached_nodes(tree)
    if not deepest:
        return None

    failures_dir = os.path.join(log_dir, 'failures')
    if render:
        os.makedirs(failures_dir, exist_ok=True)

    failures = []
    seen = set()
    for node in deepest:
        for child in tree.successors(node):
            sim_info = tree.edges[node, child]['sim_info']
            if sim_info.get('feasible'):
                continue
            reason = sim_info.get('fail_reason')
            evidence = sim_info.get('fail_evidence') or {}
            part_move = sim_info.get('part_move')
            if part_move is None or part_move in seen:
                continue
            seen.add(part_move)
            pose = sim_info.get('pose')
            parts_rest = [p for p in node if p != part_move]

            entry = {
                'child_part': part_move,
                'fail_reason': reason,
                'evidence_image': None,
            }

            try:
                if reason == 'stability':
                    entry['unstable_parts'] = evidence.get('unstable_parts', [])
                    if render:
                        rec = os.path.join(failures_dir, f'{part_move}_stability.gif')
                        parts_fix_init = [base_part] if base_part is not None else []
                        parts_move_init = [p for p in parts_rest if p not in parts_fix_init]
                        try:
                            check_stable(
                                asset_folder, assembly_dir, parts_fix_init, parts_move_init,
                                pose=pose, save_sdf=save_sdf, record_path=rec,
                            )
                        except Exception as _e:
                            print(f'[failure_dump] stability render failed for {part_move}: {_e}')
                        if os.path.exists(rec):
                            entry['evidence_image'] = os.path.relpath(rec, log_dir)

                elif reason == 'assembly':
                    entry['directions'] = evidence.get('directions', [])
                    if render:
                        best_dir = None
                        best_len = -1
                        for d in evidence.get('directions', []):
                            if d.get('path_len', 0) > best_len:
                                best_len = d['path_len']
                                best_dir = d.get('action')
                        if best_dir is not None:
                            try:
                                planner_obj = MultiPartPathPlanner(
                                    asset_folder, assembly_dir, parts_rest, part_move,
                                    pose=pose, save_sdf=save_sdf,
                                )
                                action = np.array(best_dir)
                                _, path = planner_obj.check_success(action, return_path=True)
                                rec = os.path.join(failures_dir, f'{part_move}_assembly.gif')
                                planner_obj.render(path=path, record_path=rec)
                                if os.path.exists(rec):
                                    entry['evidence_image'] = os.path.relpath(rec, log_dir)
                            except Exception as _e:
                                print(f'[failure_dump] assembly render failed for {part_move}: {_e}')

                elif reason == 'tool' and tools:
                    entry['subkind'] = evidence.get('subkind')
                    entry['colliding_parts'] = evidence.get('colliding_parts', [])
                    entry['tried_tools'] = evidence.get('tried_tools', [])
                    if render:
                        # Resolve per-part decided tools (if tools is a dict).
                        if isinstance(tools, dict):
                            _tools_for_part = tools.get(str(part_move), [])
                        else:
                            _tools_for_part = tools
                        if not _tools_for_part:
                            continue
                        fail_dir = os.path.join(failures_dir, f'{part_move}_tool')
                        diag2 = {}
                        try:
                            check_tool(
                                asset_folder=asset_folder, assembly_dir=assembly_dir,
                                parts_fix=parts_rest, part_move=part_move, tools=_tools_for_part,
                                diagnostics=diag2, failure_record_dir=fail_dir,
                            )
                        except Exception as _e:
                            print(f'[failure_dump] tool render failed for {part_move}: {_e}')
                        ev_paths = diag2.get('evidence_paths', [])
                        if ev_paths:
                            entry['evidence_image'] = os.path.relpath(ev_paths[0], log_dir)
                            entry['all_evidence_images'] = [os.path.relpath(p, log_dir) for p in ev_paths]
            except Exception as _e:
                print(f'[failure_dump] error for {part_move} ({reason}): {_e}')
                print(traceback.format_exc())

            failures.append(entry)

    payload = {
        'deepest_nodes': [list(n) for n in deepest],
        'depth': max(len(n) for n in deepest),
        'failures': failures,
    }
    with open(os.path.join(log_dir, 'failures.json'), 'w') as fp:
        json.dump(payload, fp, indent=2)
    return payload


def _build_subassembly_plan(stats, tree, opt, div, asset_folder, assembly_dir,
                            cost_fn=None, threshold=0.1, num_proc=1, debug=0):
    """Build the recursive prefix -> unified split -> S -> R plan and, when one
    exists, re-pick the disassembly sequence so its ORDER respects it.

    Writes into `stats`:
        split_plan       nested block dict (see optimizer/split_plan.py)
        split_steps      flattened step list, including the non-tree join steps
        split_sequence   the plan's disassembly order over every part
        split_sequence_source
                         'tree' when the order is a real root-to-leaf path of
                         this tree, 'derived' when it was assembled from the
                         plan because no path respected the block order
        flat_sequence    the unconstrained sequence, when `sequence` replaced it

    `stats['sequence']` — what the renderer and the arm pipeline consume — is
    only overwritten in the 'tree' case, where the new order is still a valid
    path. In the 'derived' case it is left exactly as it was, so everything
    downstream of it is untouched and only the manual reads the split order.
    Never raises: a subassembly plan is an enhancement, not a precondition for a
    usable sequence."""
    try:
        import settings as _user_settings
    except ImportError:
        _user_settings = None

    if not bool(getattr(_user_settings, 'subassembly_plan', True)):
        return
    if div is None or not getattr(div, 'verified_locally_free', None):
        return
    if not stats.get('sequence'):
        # optimize_scored found nothing to adopt; there is no order to respect.
        return

    from plan_sequence.optimizer.split_plan import (build_split_plan,
                                                    derive_split_sequence,
                                                    describe_split_plan,
                                                    flatten_split_plan,
                                                    retarget_split_plan,
                                                    split_direction,
                                                    split_order_constraint)

    try:
        plan = build_split_plan(
            tree, asset_folder, assembly_dir,
            sequence=stats['sequence'],
            divide_optimizer=div,
            root_split=stats.get('divide_split'),
            threshold=threshold,
            max_depth=int(getattr(_user_settings, 'subassembly_max_depth', 2)),
            min_parts=int(getattr(_user_settings, 'subassembly_min_parts', 4)),
            timeout=float(getattr(_user_settings, 'subassembly_search_timeout', 100)),
            top_k=int(getattr(_user_settings, 'subassembly_verify_top_k', 10)),
            num_proc=num_proc,
            sweep_states=bool(
                getattr(_user_settings, 'subassembly_sweep_states', True)),
            sweep_max_states=getattr(
                _user_settings, 'subassembly_sweep_max_states', None),
            debug=debug,
        )
    except Exception as e:
        print(f'[seq_plan] subassembly plan error: {e}')
        return

    if plan is None:
        if debug > 0:
            print('[seq_plan] no recursive subassembly plan found; keeping flat sequence')
        return

    try:
        constrained, cost = opt.optimize_constrained(
            split_order_constraint(plan), cost_fn=cost_fn,
        )
    except Exception as e:
        print(f'[seq_plan] constrained sequence search error: {e}')
        return

    if constrained is not None:
        # Best case: the block order is itself a valid path, so the renderer,
        # the per-step poses and the arm pipeline all follow the same order the
        # manual tells.
        source = 'tree'
        order = list(constrained)
        stats['flat_sequence'] = list(stats['sequence'])
        stats['sequence'] = list(constrained)
    else:
        # A block ordering is a narrow slice of all orderings, and the search is
        # budget-limited, so the tree usually holds no path that respects it.
        # Derive the order from the plan instead and leave stats['sequence']
        # alone: the renderer keeps following a real tree path while the manual
        # follows the subassembly narrative.
        source = 'derived'
        order = derive_split_sequence(plan, stats['sequence'])

    plan = retarget_split_plan(plan, order)
    steps = flatten_split_plan(plan, order)

    # Attach the verified separation axis to each join step. This is plan
    # metadata describing the join, not something the manual draws: its join
    # page deliberately shows only the seated result (a pre-assembly ghost of a
    # whole subassembly read as a third body rather than as "R, before it goes
    # on"). Kept because "which way do these two come apart" is the defining
    # fact about a join, and stats.json is the interchange format.
    #
    # Also record how many contact edges cross the cut. The cut score REWARDS
    # few crossings, so a split whose sides never touch scores well and is
    # common -- and for such a split there is no mating at all: the two sides
    # are independent sub-builds that later parts bridge. The manual needs to
    # say that rather than instruct the reader to seat one onto the other.
    # obstruction_graph is built from get_contact_graph, so its edges are
    # contacts and this costs nothing.
    contact_graph = getattr(div, 'obstruction_graph', None)
    for entry in steps:
        if entry['kind'] != 'join':
            continue
        entry['direction'] = split_direction(
            asset_folder, assembly_dir, entry['S'], entry['R'],
        )
        if contact_graph is None:
            entry['contact_edges'] = None
            continue
        S, R = set(entry['S']), set(entry['R'])
        entry['contact_edges'] = sum(
            1 for u, v in contact_graph.edges()
            if (u in S and v in R) or (u in R and v in S)
        )

    stats['split_plan'] = plan
    stats['split_steps'] = steps
    stats['split_sequence'] = [e['part'] for e in steps if e['kind'] == 'remove']
    stats['split_sequence_source'] = source

    n_joins = sum(1 for e in steps if e['kind'] == 'join')
    print(f'[seq_plan] subassembly plan adopted: {n_joins} split(s), '
          f'order from {source}'
          + (f', cost {cost:.4f}' if cost is not None else ''))
    if debug > 0:
        print(describe_split_plan(plan))


def seq_plan(asset_folder, assembly_dir, generator_name, planner_name, num_proc, seed, budget, max_gripper, max_pose, pose_reuse, early_term, timeout, base_part,
    save_sdf, clear_sdf, plan_grasp, plan_arm, gripper_type, gripper_scale, optimizer, debug, render, record_dir, log_dir, allow_gap=False, n_success_term=1, connect_path=False, get_dof=False, tools=None, skip_stability=False, max_frontier=4, seq_optimizer=None):

    try:
        generator_cls = generators[generator_name]
        generator = generator_cls(asset_folder, assembly_dir, base_part=base_part, save_sdf=save_sdf)
        planner_cls = planners[planner_name]
        planner = planner_cls(generator, num_proc, save_sdf=save_sdf, allow_gap=allow_gap, get_dof=get_dof, tools=tools, skip_stability=skip_stability)
        planner.seed(seed)

        setup = {
            'budget': budget, 'max_grippers': max_gripper, 'max_poses': max_pose, 'pose_reuse': pose_reuse, 'early_term': early_term, 'timeout': timeout,
            'plan_grasp': plan_grasp, 'plan_arm': plan_arm, 'gripper_type': gripper_type, 'gripper_scale': gripper_scale, 'optimizer': optimizer, 'n_success_term': n_success_term,
            'connect_path': connect_path, 'max_frontier': max_frontier,
        }
        tree = planner.plan(**setup, debug=debug - 1, render=False, log_dir=log_dir)

        stats = planner.get_stats(tree)

        # Integrated sequence selection + divide-optimizer split probe.
        # The chosen sequence replaces stats['sequence']; the split is purely
        # diagnostic (printed when debug > 0). See BaseSequenceOptimizer.
        if stats.get('success') and seq_optimizer == 'divide':
            from plan_sequence.optimizer.base import BaseSequenceOptimizer
            from plan_sequence.optimizer.divide import DivideOptimizer

            opt = BaseSequenceOptimizer(tree)

            # Best-effort divide optimizer on the full assembly graph.
            div = None
            try:
                div = DivideOptimizer(tree, asset_folder=asset_folder, assembly_dir=assembly_dir)
                if div.build_obstruction_graph() is None:
                    div = None
                else:
                    div.find_locally_free_subassemblies(timeout=100)
                    div.verify_locally_free(top_k=10, num_proc=num_proc)
            except Exception as _e:
                print(f'[seq_plan] divide optimizer error: {_e}')
                div = None

            # Per-edge cost closure if the planner exposes one.
            _cost_fn = None
            if hasattr(planner, '_cost_child') and hasattr(planner, '_load_weights'):
                _weights = planner._load_weights()
                _planner = planner
                # Capture `tree` so the heuristic's pose_change feature can
                # look up the parent step's pose via _parent_pose_for. Without
                # this, optimizer cost evaluations would silently miss the
                # pose-change penalty (parent_pose would default to None).
                _has_parent_lookup = hasattr(planner, '_parent_pose_for')
                def _cost_fn(G_prime, sim_info, parent_G,
                             _p=_planner, _w=_weights, _t=tree,
                             _has_lookup=_has_parent_lookup):
                    parent_pose = (_p._parent_pose_for(_t, parent_G)
                                   if _has_lookup else None)
                    return _p._cost_child(_w, G_prime, sim_info, parent_G,
                                          parent_pose=parent_pose)

            try:
                import settings as _user_settings
                _threshold = float(getattr(_user_settings, 'divide_split_threshold', 0.1))
            except ImportError:
                _threshold = 0.1

            chosen_sequence = opt.optimize_scored(
                cost_fn=_cost_fn, divide_optimizer=div,
                threshold=_threshold, debug=debug,
            )
            if chosen_sequence is not None:
                if debug > 0 and div is not None:
                    print(f'[seq_plan] divide optimizer found {len(div.locally_free)} locally free subassemblies:')
                    for idx, entry in enumerate(div.locally_free):
                        S, R, score = entry[0], entry[1], entry[2]
                        print(f'  {idx+1}. S={sorted(S)}  R={sorted(R)}  score={score:.4f}  '
                              f'{"ACCEPTED" if score >= _threshold else "rejected"}')
                stats['sequence'] = list(chosen_sequence)

            # Persist the top physically-verified subassembly split so the
            # renderer can visualise it (see play_subassembly_split). Skipped
            # entirely when nothing verified.
            verified = getattr(div, 'verified_locally_free', None) if div is not None else None
            if verified:
                S, R, score = verified[0][0], verified[0][1], verified[0][2]
                stats['divide_split'] = {
                    'S': sorted(S), 'R': sorted(R), 'score': float(score),
                }
                if debug > 0:
                    print(f'[seq_plan] persisted divide_split: S={sorted(S)}  R={sorted(R)}  score={score:.4f}')

            # Recursive subassembly plan. Runs alongside everything above: the
            # flat sequence has already been chosen and divide_split already
            # persisted, so a failure here leaves the run exactly as it was.
            _build_subassembly_plan(
                stats, tree, opt, div, asset_folder, assembly_dir,
                cost_fn=_cost_fn, threshold=_threshold, num_proc=num_proc,
                debug=debug,
            )

        if log_dir is not None:
            planner.log(tree, stats, log_dir)
            with open(os.path.join(log_dir, 'setup.json'), 'w') as fp:
                json.dump(setup, fp)
            if not stats.get('success'):
                # render_sequence gates both the successful-sequence GIFs and
                # the failed-children evidence renders; metadata (failures.json)
                # is cheap and always emitted regardless.
                try:
                    import settings as _user_settings
                    _render_failures = bool(getattr(_user_settings, 'render_sequence', True))
                except ImportError:
                    _render_failures = True
                # When the planner aborted at the precheck stage
                # (no_stable_pose_action='exit') it has already written a
                # failures.json populated from observed_fallen + the
                # precheck_unstable_XX.png renders. Skip the regular dump so
                # we don't overwrite it with an empty placeholder.
                _stop_msg = getattr(planner, 'stop_msg', None)
                _precheck_abort = (_stop_msg == 'no self-stable initial pose')
                if not _precheck_abort:
                    try:
                        _dump_failure_evidence(
                            tree, asset_folder, assembly_dir, base_part, tools, save_sdf, log_dir,
                            render=_render_failures,
                        )
                    except Exception as _e:
                        print(f'[seq_plan] failure-evidence dump aborted: {_e}')
                        print(traceback.format_exc())
                else:
                    print(f'[seq_plan] precheck abort detected '
                          f'(stop_msg={_stop_msg!r}); keeping the '
                          f'precheck-stability failures.json the planner '
                          f'already wrote.')

        if debug:
            print(f'[seq_plan] stats: {stats}')

        # Render whenever there is a sequence to show — full on success, or the
        # deepest feasible prefix when the planner got stuck (stats['partial']).
        if render and stats['sequence']:
            planner.render(stats['sequence'], tree, record_dir)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[seq_plan] interrupt')
        else:
            print('[seq_plan] exception:', e, f'from {assembly_dir}')
            print(traceback.format_exc())
        if clear_sdf:
            clear_saved_sdfs(assembly_dir)
        raise e

    if clear_sdf:
        clear_saved_sdfs(assembly_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--planner', type=str, required=True, choices=list(planners.keys()), help='name of planner (node selection algorithm)')
    parser.add_argument('--generator', type=str, required=True, choices=list(generators.keys()), help='name of part generator (part selection algorithm)')
    parser.add_argument('--num-proc', type=int, default=1, help='number of processes for parallel planning')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=400, help='maximum evaluation (feasibility check) budget')
    parser.add_argument('--max-gripper', type=int, default=3)
    parser.add_argument('--max-pose', type=int, default=3)
    parser.add_argument('--pose-reuse', type=int, default=0)
    parser.add_argument('--early-term', default=False, action='store_true')
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--base-part', type=str, default=None)
    parser.add_argument('--debug', type=int, default=3, help='depth of debug message')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--record-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    parser.add_argument('--clear-sdf', default=False, action='store_true')
    parser.add_argument('--plan-grasp', default=False, action='store_true')
    parser.add_argument('--plan-arm', default=False, action='store_true')
    parser.add_argument('--gripper', type=str, default='robotiq-140', choices=['panda', 'robotiq-85', 'robotiq-140'])
    parser.add_argument('--scale', type=float, default=0.4)
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B')
    parser.add_argument('--get-dof', default=False, action='store_true', help='compute per-direction DoF probe during each assemblability check')

    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)
    exp_name = f'{args.planner}-{args.generator}'

    if args.record_dir is None:
        record_dir = None
    else:
        record_dir = os.path.join(args.record_dir, exp_name, f's{args.seed}', args.id)

    if args.log_dir is None:
        log_dir = None
    else:
        log_dir = os.path.join(args.log_dir, exp_name, f's{args.seed}', f'{args.id}')

    seq_plan(asset_folder, assembly_dir, args.generator, args.planner, args.num_proc, args.seed, args.budget, args.max_gripper, args.max_pose, args.pose_reuse, args.early_term, args.timeout, args.base_part,
        not args.disable_save_sdf, args.clear_sdf, args.plan_grasp, args.plan_arm, args.gripper, args.scale, args.optimizer, args.debug, args.render, record_dir, log_dir, get_dof=args.get_dof)
