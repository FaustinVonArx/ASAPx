"""Optuna-based black-box optimisation of HeuristicDFASequencePlanner
weights, with total arm-pipeline assembly time as the objective.

Objective: the mean over the training assemblies of
``log(total_s / baseline_total_s)``, i.e. the log of the geometric-mean time
ratio against a baseline. 0 is "as fast as the baseline", -0.1 is ~10%
faster. Every assembly counts equally, however long it takes to disassemble;
a mean of raw seconds would be dominated by the largest ones.

Baseline: one plan per assembly with the reference weights (DEFAULT_WEIGHTS
overridden by ``settings.heuristic_weights``) rescaled to the pinned weight
-- the same ranking, hence the same plan. Computed before the first trial,
stored per assembly under ``<output_root>/baselines/`` and reused while the
planning and timing settings they were made with are unchanged. An assembly
whose baseline does not plan completely is left out of training.

Search space: the cost only ranks candidates, so scaling all weights by one
factor changes nothing. One weight is pinned: ``hold_count`` =
``settings.time_per_held_part_s``, so every weight reads as predicted seconds
per unit of its feature, or 1.0 while that penalty is off (the default), the
others then being relative to it. The others are sampled log-uniformly. More weights
can be fixed via ``settings.heuristic_training['fixed_weights']``. The study
starts with the reference weights (which must score exactly 0 -- a built-in
determinism check) and a prior derived from the time model.

Failures: an assembly counts only when it planned completely
(``stats['success']``, a full-length sequence, timing for every step): a
failed plan still renders and times its partial sequence, and those fewer
steps would otherwise score as a fast run. A trial that fails an assembly
records ``n_failed`` as an Optuna constraint and scores +inf, so TPE ranks it
behind every trial that planned them all; by default it also stops there,
since it can no longer be chosen.

Pruning: each assembly's log-ratio is reported as it finishes, and
WilcoxonPruner stops a trial once it is significantly worse than the best one
on the assemblies seen so far. Assemblies are evaluated in a per-trial
shuffled order (seeded by the trial number) so that pruning decisions are not
always made on the same few. A pruned trial returns its partial mean, as the
Optuna docs recommend, so TPE still learns from it; only fully evaluated
trials can become the trained weights. With a time budget, a trial that
would run past it ends the same way (outcome 'deadline') and the study
stops.

Several objectives (``settings.heuristic_training['objectives']``, e.g.
``('time', 'held_parts', 'non_upward')``): the time model does not charge
held parts or the pull direction, and a penalty in seconds for them would
only fix their exchange rate in advance. Each is instead an objective of its
own (see OBJECTIVES; read off the chosen sequence by _plan_metrics), the study
finds the Pareto front, and the weights are picked from it by
``pareto_pick`` -- by default the fastest trial no worse than the reference
weights on the others. The front is written next to the weights file
(``<weights>_pareto.json``) so another point can be chosen by hand. Optuna
does not prune multi-objective studies, so every trial is a full pass.

Storage layout (paths relative to repo root, configurable via settings):
- ``assets/heuristic_weights_optuna.json`` — the trained weights (all five,
  fixed ones included): the best fully evaluated trial that planned every
  assembly, rewritten after every trial so a job killed at its time limit
  still leaves its result; left as-is while none qualifies. This is the file
  `HeuristicDFASequencePlanner._load_weights` reads when
  ``settings.heuristic_weights_source == 'optuna'``. Candidates under
  evaluation are never written here.
- ``assets/optuna_store/`` (DEFAULT_STORE; ``--optuna-store``) — every
  planned run, shared by all runs: ``runs/<geometry key>/<fingerprint
  key>.json`` (the record), ``..._run/`` (planning output) and
  ``....weights.json`` (the weights the planner read, via
  ``settings.heuristic_weights_optuna_path``, so parallel trials never read
  each other's), plus ``sim_cache/``, the candidate-check cache
  (plan_sequence/planner/sim_cache.py). A later run -- more or other
  assemblies, a new study -- reuses every run made under the same
  fingerprint (see _ensure_runs); import_run_into_store brings in runs from
  before the store.
- ``assets/optuna_training/reference_check/`` — the reference trial's plans,
  made afresh as the determinism / stale-store check.
- ``assets/heuristic_weights_optuna_history.json`` — per-trial log rebuilt
  from the study after every trial, so parallel workers never drop each
  other's entries. A fresh study moves an existing file aside first.
- ``assets/optuna_training/study.journal`` (when ``persist_study=True``) —
  Optuna journal storage. Safe on a shared filesystem, so several processes
  (e.g. cluster array jobs) can work on one study; ``n_trials`` is the
  study-wide total.
- ``assets/optuna_training/eval_<label>/summary.{json,txt}`` —
  `evaluate_heuristic_weights`: the trained weights against the reference on
  held-out assemblies.

`main.py --optuna-dir DIR` moves the run's own files (weights, history,
journal, summary) into DIR; planned runs stay in the store.

Training / inference switch:
- Training: ``python main.py train_heuristic_weights --id <range>``. The
  trainer forces ``settings.heuristic_weights_source = 'optuna'`` for the
  duration of the study (so the planner reads each trial's weights file)
  and restores it on exit.
- Inference: set ``settings.heuristic_weights_source = 'optuna'`` and run
  normal commands — the planner just reads the file; nothing writes, so the
  weights are frozen.
"""
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import socket
import threading
import time
import traceback
import warnings
from pathlib import Path


WEIGHT_KEYS = ('contact_distance', 'free_dof', 'z_alignment', 'pose_change',
               'hold_count')

# Defaults for settings.heuristic_training (see settings.py).
_TRAINING_DEFAULTS = {
    'fixed_weights': {'hold_count': None},
    'search_bounds': (0.01, 100.0),
    'enqueue_reference': True,
    'enqueue_time_model_prior': True,
    'stop_on_failure': True,
    'pruner_p_threshold': 0.1,
    'render_gifs': False,
    'objectives': ('time',),
    'pareto_pick': 'no_worse_than_reference',
    'warm_start_top': 10,
}

# What a trial can be scored on (settings.heuristic_training['objectives']),
# all minimised. 'time' is the mean log time ratio against the baseline; the
# others are plan properties the time model does not charge, read off the
# chosen sequence's tree edges (_plan_metrics), as a per-step mean per
# assembly averaged over the assemblies.
OBJECTIVES = {
    'time': 'mean log time ratio vs baseline',
    'held_parts': 'mean extra parts held per step',
    'non_upward': 'mean 1 - cos(angle of the pull to world up) per step (0 up, 1 sideways, 2 down)',
}
PARETO_PICKS = ('no_worse_than_reference', 'fastest')

# stats['stop_msg'] of a plan aborted by the initial stable-pose precheck
# (settings.no_stable_pose_action = 'exit').
_PRECHECK_ABORT = 'no self-stable initial pose'

# Seconds between checks while another process computes a stored run.
_RUN_POLL_S = 30

# A live holder refreshes its lock's mtime every _LOCK_HEARTBEAT_S; a lock
# not refreshed for _LOCK_STALE_S is stale (its holder was killed, e.g. out of
# memory or at a job's time limit, possibly on another node) and is taken over.
_LOCK_HEARTBEAT_S = 60
_LOCK_STALE_S = 15 * 60


def _write_json(path, data, indent=2):
    # Write to a per-process temp file, then rename, so a reader never sees
    # a half-written file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    os.replace(tmp, path)


def _archived_history_path(path):
    """Free sibling path for moving an old history file aside, stamped with
    its last-modified time."""
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(path.stat().st_mtime))
    candidate = path.with_name(f'{path.stem}.{stamp}{path.suffix}')
    n = 1
    while candidate.exists():
        candidate = path.with_name(f'{path.stem}.{stamp}_{n}{path.suffix}')
        n += 1
    return candidate


# ----------------------------------------------------------------------------
# Weights: pinning, reference, prior
# ----------------------------------------------------------------------------

def _training_config():
    import settings
    cfg = dict(_TRAINING_DEFAULTS)
    cfg.update(getattr(settings, 'heuristic_training', None) or {})
    fixed = {}
    for k, v in (cfg.get('fixed_weights') or {}).items():
        if k not in WEIGHT_KEYS:
            raise ValueError(f'heuristic_training.fixed_weights: unknown weight {k!r}')
        if v is None:
            if k != 'hold_count':
                raise ValueError(f'heuristic_training.fixed_weights: only hold_count '
                                 f'may be None (pinned to time_per_held_part_s), not {k!r}')
            v = float(getattr(settings, 'time_per_held_part_s', 0.0))
            if v <= 0:
                # The objective does not charge held parts, so there is no
                # time unit to pin to; any positive value fixes the scale.
                v = 1.0
        fixed[k] = float(v)
    cfg['fixed_weights'] = fixed
    objectives = tuple(cfg.get('objectives') or ('time',))
    unknown = [o for o in objectives if o not in OBJECTIVES]
    if unknown or 'time' not in objectives or len(set(objectives)) != len(objectives):
        raise ValueError(f'heuristic_training.objectives: {objectives!r}; distinct names from '
                         f'{list(OBJECTIVES)}, including time')
    cfg['objectives'] = objectives
    if cfg.get('pareto_pick') not in PARETO_PICKS:
        raise ValueError(f'heuristic_training.pareto_pick: {cfg.get("pareto_pick")!r}, '
                         f'not one of {PARETO_PICKS}')
    return cfg


def _reference_weights():
    """The weights a plain `heuristic` run uses: DEFAULT_WEIGHTS overridden by
    settings.heuristic_weights (the 'default' branch of _load_weights)."""
    import settings
    from plan_sequence.planner.heuristic import HeuristicDFASequencePlanner

    weights = dict(HeuristicDFASequencePlanner.DEFAULT_WEIGHTS)
    overrides = getattr(settings, 'heuristic_weights', None)
    if isinstance(overrides, dict):
        weights.update({k: float(v) for k, v in overrides.items() if k in weights})
    return weights


def _pin(weights, fixed):
    """Rescale `weights` so the pinned weight takes its fixed value (the
    ranking is unchanged), then apply the other fixed values. hold_count is
    the pin when fixed, otherwise the first fixed weight."""
    weights = dict(weights)
    if fixed:
        pin = 'hold_count' if 'hold_count' in fixed else next(iter(fixed))
        if weights.get(pin, 0.0) > 0:
            scale = fixed[pin] / weights[pin]
            weights = {k: v * scale for k, v in weights.items()}
    weights.update(fixed)
    return {k: float(weights[k]) for k in WEIGHT_KEYS}


def _time_model_prior(fixed):
    """Weights read off the arm pipeline's time model, in the pinned units
    (seconds per unit of feature). Rough by design: a starting point for the
    search, not an answer."""
    import settings

    prior = {
        'hold_count': float(getattr(settings, 'time_per_held_part_s', 2.0)) or 1.0,
        # pose_change is (1 - cos θ) / 2, i.e. 1 for a 180° flip, which the
        # time model charges π / assembly_reorientation_velocity_rad_s.
        'pose_change': math.pi / float(getattr(settings, 'assembly_reorientation_velocity_rad_s', 0.15)),
        # contact_distance is ln(hops + 1). Guess: a jump to a part ~3 hops
        # away (ln 4) walks about a quarter turn around an assembly of radius
        # ~6 units (asap bounding-box diagonals are ~7-16) at the base-travel
        # velocity.
        'contact_distance': (6.0 * math.pi / 2)
        / float(getattr(settings, 'arm_base_travel_velocity', 2.0)) / math.log(4),
        # Not charged by the time model.
        'z_alignment': 0.5,
        'free_dof': 0.5,
    }
    prior.update(fixed)
    return {k: float(prior[k]) for k in WEIGHT_KEYS}


# ----------------------------------------------------------------------------
# Running one assembly
# ----------------------------------------------------------------------------

def _assess_run(storage_dir, ass, split=False):
    """Classify one assembly's pipeline output. Returns (status, total_s,
    components, extra): total_s and the timing components (timing_overview
    totals) are None unless status == 'ok'; `extra` is {} except with
    `split`.

    `split` (a --seq-optimizer divide run): score the subassembly plan's own
    timing (timing_overview_split.json) when the run found a plan that can be
    carried out, else the run's flat sequence; `extra` records which
    ('used' / 'none' / 'infeasible'), the flat total and the number of joins.

    'excluded' marks a precheck abort, which the weights cannot influence.
    Every other status except 'ok' is a failure.
    """
    log_dir = Path(storage_dir) / 'log'
    try:
        with open(log_dir / 'stats.json') as _f:
            stats = json.load(_f)
    except (OSError, json.JSONDecodeError):
        return 'no_stats', None, None, {}

    stop_msg = stats.get('stop_msg')
    if stop_msg == 'interrupt':
        # The planner swallows Ctrl+C and returns what it had. Re-raise so the
        # study stops and writes its best-so-far, instead of charging the
        # interrupted plan to the weights.
        raise KeyboardInterrupt
    if stop_msg == _PRECHECK_ABORT:
        return 'excluded', None, None, {}
    if not stats.get('success'):
        return f'plan_failed: {stop_msg}', None, None, {}
    # A complete plan removes every part but the last, which stays put.
    sequence = stats.get('sequence') or []
    if len(set(ass.objects) - set(sequence)) > 1:
        return 'incomplete_sequence', None, None, {}

    try:
        with open(log_dir / 'timing_overview.json') as _f:
            timing = json.load(_f)
    except (OSError, json.JSONDecodeError):
        return 'no_timing', None, None, {}
    if len(timing.get('per_step') or []) != len(sequence):
        return 'timing_mismatch', None, None, {}
    total = (timing.get('totals') or {}).get('total_s')
    if total is None or float(total) <= 0:
        # A complete plan always takes time; a ratio against zero is undefined.
        return 'no_timing', None, None, {}
    if not split:
        return 'ok', float(total), timing['totals'], {}

    extra = {'split': 'none', 'flat_total_s': float(total), 'flat_components': timing['totals'],
             'n_joins': 0, 'source': stats.get('split_sequence_source')}
    if not stats.get('split_plan'):
        return 'ok', float(total), timing['totals'], extra
    try:
        with open(log_dir / 'timing_overview_split.json') as _f:
            split_timing = json.load(_f)
    except (OSError, json.JSONDecodeError):
        extra['split'] = 'untimed'
        return 'ok', float(total), timing['totals'], extra
    extra['n_joins'] = split_timing.get('n_joins', 0)
    if split_timing.get('status') != 'ok':
        extra['split'] = 'infeasible'
        extra['failure'] = split_timing.get('failure')
        return 'ok', float(total), timing['totals'], extra
    extra['split'] = 'used'
    extra['parallel_total_s'] = (split_timing.get('parallel') or {}).get('total_s')
    extra['parallel2_total_s'] = (split_timing.get('parallel_2') or {}).get('total_s')
    return 'ok', float(split_timing['totals']['total_s']), split_timing['totals'], extra


def _plan_metrics(run_dir):
    """The non-time objectives of one completed plan: per-step means, over
    the chosen sequence (stats['sequence']), of the extra parts held
    (len(parts_fix)) and of the pull's non-upwardness (1 - z of the unit
    action, world frame), both read off the tree edges the sequence walks and
    defined like the heuristic's hold_count and z_alignment features. None
    when the tree does not contain the sequence."""
    import pickle
    log_dir = Path(run_dir) / 'log'
    try:
        with open(log_dir / 'stats.json') as _f:
            sequence = json.load(_f).get('sequence') or []
        with open(log_dir / 'tree.pkl', 'rb') as _f:
            tree = pickle.load(_f)
    except (OSError, json.JSONDecodeError, pickle.UnpicklingError, EOFError):
        return None
    if not sequence or not tree.number_of_nodes():
        return None
    by_set = {frozenset(n): n for n in tree.nodes}
    node = max(tree.nodes, key=len)
    held, non_upward = [], []
    for part in sequence:
        child = by_set.get(frozenset(node) - {part})
        if child is None or not tree.has_edge(node, child):
            return None
        sim_info = tree.edges[node, child].get('sim_info') or {}
        held.append(len(sim_info.get('parts_fix') or []))
        action = sim_info.get('action')
        if action is None:
            non_upward.append(1.0)
        else:
            a = [float(x) for x in action]
            n = math.sqrt(sum(x * x for x in a))
            non_upward.append(1.0 - a[2] / n if n > 1e-9 else 1.0)
        node = child
    return {'held_parts': sum(held) / len(held), 'non_upward': sum(non_upward) / len(non_upward)}


def _use_weights(weights, path):
    """Write `weights` to `path` and point the planner at it."""
    import settings
    _write_json(path, weights)
    settings.heuristic_weights_optuna_path = str(path)


def _run_assembly(ass, args, run_dir, split=False):
    """Plan + arm-time one assembly into `run_dir`, which is wiped first:
    get_assembly_plans_ASAP returns a leftover sequence.json without planning.
    Returns (status, total_s, wall_s, components, extra); see _assess_run."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(str(run_dir))
    run_dir.mkdir(parents=True)
    _saved_storage = ass.storage_dir
    t0 = time.time()
    try:
        ass.storage_dir = run_dir
        ass.planner.get_assembly_plans(args)
        status, total, components, extra = _assess_run(run_dir, ass, split=split)
    except Exception as _e:
        traceback.print_exc()
        status, total, components, extra = f'error: {_e}', None, None, {}
    finally:
        ass.storage_dir = _saved_storage
    return status, total, time.time() - t0, components, extra


# ----------------------------------------------------------------------------
# Stored runs (baselines, evaluation)
# ----------------------------------------------------------------------------

def _run_fingerprint(args, weights, planner='heuristic', generator='rand', seq_optimizer=None):
    """What a stored run depends on besides the assembly: its planner and
    weights and the planning / timing configuration. A stored run is reused
    only while this matches."""
    import settings

    setting_keys = (
        'max_frontier', 'n_success_term', 'get_dof', 'skip_stability',
        'no_stable_pose_action', 'max_initial_held_parts', 'filter_below_ground',
        'sim_param_preset', 'arm_continuous', 'arm_simplified_mode',
        'arm_simplified_k_dist', 'arm_simplified_k_vol', 'arm_simplified_check_grasp',
        'arm_simplified_replan_paths', 'arm_simplified_clip_path',
        'contact_model', 'arm_joint_velocity_rad_s', 'arm_base_travel_velocity',
        'assembly_reorientation_velocity_rad_s', 'time_per_held_part_s',
        'failed_step_time_multiplier', 'subassembly_plan',
    )
    # Not num_proc: results no longer depend on it (DFA submission order).
    arg_keys = ('budget', 'max_pose', 'max_gripper', 'pose_reuse', 'early_term',
                'seq_optimizer', 'seed', 'gripper_type', 'gripper_scale')
    if seq_optimizer == 'divide':
        # What the subassembly plan and its timing depend on.
        setting_keys += (
            'divide_weights', 'divide_split_threshold', 'subassembly_max_depth',
            'subassembly_min_parts', 'subassembly_search_timeout',
            'subassembly_verify_top_k', 'subassembly_sweep_states',
            'subassembly_sweep_max_states',
        )
    fp = {
        'planner': planner,
        'generator': generator,
        'seq_optimizer': seq_optimizer,
        'weights': weights,
        'settings': {k: getattr(settings, k, None) for k in setting_keys},
        'args': {k: getattr(args, k, None) for k in arg_keys if k != 'seq_optimizer'},
    }
    # Only the heuristic planner selects its sequence (run_seq_plan). Runs
    # stored before selection existed kept the first sequence found, so
    # 'first' leaves the fingerprint as it was and they stay reusable; the
    # divide optimizer used to pick its own sequence, so a divide run always
    # records the mode and never matches one of those.
    selection = getattr(settings, 'sequence_selection', 'min_cost')
    if planner == 'heuristic' and (selection != 'first' or seq_optimizer == 'divide'):
        fp['sequence_selection'] = selection
    # JSON round trip so a fresh fingerprint compares equal to a stored one
    # (tuples become lists).
    return json.loads(json.dumps(fp, default=str))


def _try_lock(path):
    """Create `path` exclusively; None if another live process holds it. A
    lock left by a dead process on this host, or not refreshed for
    _LOCK_STALE_S (a holder on another node that was killed), is taken over.
    Hold it through _LockHeartbeat, which keeps it fresh."""
    me = {'host': socket.gethostname(), 'pid': os.getpid(), 'since': time.time()}
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(path) as _f:
                    holder = json.load(_f)
            except (OSError, json.JSONDecodeError):
                return None
            try:
                age = time.time() - os.path.getmtime(path)
            except OSError:
                continue  # released meanwhile
            if age > _LOCK_STALE_S:
                os.remove(str(path))
                continue
            if holder.get('host') != me['host']:
                return None
            try:
                os.kill(int(holder.get('pid')), 0)
                return None  # alive
            except (ProcessLookupError, TypeError, ValueError):
                os.remove(str(path))  # stale: its process is gone
                continue
            except PermissionError:
                return None
        with os.fdopen(fd, 'w') as _f:
            json.dump(me, _f)
        return path
    return None


class _LockHeartbeat:
    """Refresh a held lock's mtime from a background thread until closed, so
    other processes (on any node) can tell a live holder from a dead one."""

    def __init__(self, path):
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self):
        while not self._stop.wait(_LOCK_HEARTBEAT_S):
            try:
                os.utime(self.path)
            except OSError:
                return

    def close(self):
        self._stop.set()
        self._thread.join()


# Every planned run (baselines, trials, evaluation) lives in a store shared
# by all runs, keyed by the assembly's geometry and the run's fingerprint, so
# a later run -- a larger or wider sample, a new study -- reuses whatever an
# earlier one already planned under identical conditions.
DEFAULT_STORE = 'assets/optuna_store'
_geometry_keys = {}


def _assembly_key(assembly_dir):
    """Store key of an assembly: the hash of its OBJ files (the key the
    candidate-check cache uses), so ids of different datasets never collide
    and re-meshed parts never reuse old plans."""
    from plan_sequence.planner.sim_cache import _geometry_hash
    path = str(Path(assembly_dir).resolve())
    if path not in _geometry_keys:
        _geometry_keys[path] = _geometry_hash(assembly_dir)[:16]
    return _geometry_keys[path]


def _fingerprint_key(fingerprint):
    return hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]


def _store_paths(store, assembly_dir, fingerprint):
    """Where one stored run lives:
    <store>/runs/<geometry key>/<fingerprint key>{.json,_run/,.lock,.weights.json}."""
    d = Path(store) / 'runs' / _assembly_key(assembly_dir)
    k = _fingerprint_key(fingerprint)
    return {'dir': d, 'record': d / f'{k}.json', 'run': d / f'{k}_run',
            'lock': d / f'{k}.lock', 'weights': d / f'{k}.weights.json'}


def _load_record(paths, fingerprint):
    """The stored record at `paths`, or None. Records made before plan
    metrics were stored get them added from their run directory."""
    try:
        with open(paths['record']) as _f:
            record = json.load(_f)
    except (OSError, json.JSONDecodeError):
        return None
    if record.get('fingerprint') != fingerprint:
        return None  # a key collision; recompute
    if record.get('status') == 'ok' and 'metrics' not in record and not record.get('split'):
        record['metrics'] = _plan_metrics(paths['run'])
        _write_json(paths['record'], record)
    return record


def _note_assembly(paths, ass):
    """A readable note of which assembly a store directory belongs to."""
    note = paths['dir'] / 'assembly.json'
    if not note.exists():
        _write_json(note, {'id': str(ass.id), 'assembly_dir': str(ass.assembly_dir),
                           'n_parts': len(ass.objects)})


def _ensure_runs(ass_list, args, store, weights, label, clear_sdf=False,
                 deadline=None, planner='heuristic', generator='rand', seq_optimizer=None,
                 wait=True, arg_overrides=None, setting_overrides=None):
    """See _ensure_runs_. `arg_overrides` / `setting_overrides`: args fields
    and settings values the run set is planned under (e.g. another --seed,
    sequence_selection='first'); they enter the fingerprint like any other."""
    saved_args = {k: getattr(args, k, None) for k in (arg_overrides or {})}
    for k, v in (arg_overrides or {}).items():
        setattr(args, k, v)
    try:
        with _settings_as(setting_overrides or {}):
            return _ensure_runs_(ass_list, args, store, weights, label, clear_sdf, deadline,
                                 planner, generator, seq_optimizer, wait)
    finally:
        for k, v in saved_args.items():
            setattr(args, k, v)


def _ensure_runs_(ass_list, args, store, weights, label, clear_sdf=False,
                  deadline=None, planner='heuristic', generator='rand', seq_optimizer=None,
                  wait=True):
    """One stored run per assembly with fixed `weights`, computing the missing
    ones into `store` (see _store_paths) and returning {id: record}. Several
    processes can run this at once (e.g. the workers of a job array): each
    run is computed by whichever process takes its lock, and the others pick
    up the result. A record is reused by any later call, from any run, whose
    fingerprint (_run_fingerprint: planner, weights, planning and timing
    configuration) and assembly geometry match.

    `clear_sdf`: clear the assembly's SDF caches before planning, like a normal
    run does, so they are regenerated from the current meshes. Used for the
    first plan of an assembly (its baseline); every later plan reuses them
    (args.use_previous_sdf), which is what makes concurrent workers on one
    assembly safe.

    `deadline` (epoch seconds): past it, no new run starts and no waiting on
    other processes continues; the assemblies not done by then are missing
    from the result.

    `planner` / `generator`: what plans (default: the heuristic planner, which
    reads `weights`); e.g. 'gen-adapter' / 'heur-out' for the heur-out
    baseline, which ignores the weights. `seq_optimizer='divide'` adds the
    subassembly plan, and the run is scored by its split timing (see
    _assess_run).

    `wait=False`: compute what no other process holds and return without
    waiting for the rest, so a caller with several run sets can do all the
    work it can take before it waits on anyone."""
    fingerprint = _run_fingerprint(args, weights, planner, generator, seq_optimizer)

    records = {}
    pending = list(ass_list)
    announced = None
    while pending:
        waiting = []
        for ass in pending:
            aid = str(ass.id)
            paths = _store_paths(store, ass.assembly_dir, fingerprint)
            record = _load_record(paths, fingerprint)
            if record is not None:
                records[aid] = record
                continue
            if deadline is not None and time.time() > deadline:
                print(f'[optuna] {label} {aid}: skipped, time budget used up')
                continue
            paths['dir'].mkdir(parents=True, exist_ok=True)
            lock = _try_lock(paths['lock'])
            if lock is None:
                waiting.append(ass)
                continue
            heartbeat = _LockHeartbeat(lock)
            try:
                # Another process may have finished it between the check and the lock.
                record = _load_record(paths, fingerprint)
                if record is not None:
                    records[aid] = record
                    continue
                _note_assembly(paths, ass)
                _use_weights(weights, paths['weights'])
                _saved = (getattr(args, 'use_previous_sdf', False), args.planner,
                          args.generator, getattr(args, 'seq_optimizer', None))
                if clear_sdf:
                    args.use_previous_sdf = False
                args.planner, args.generator, args.seq_optimizer = planner, generator, seq_optimizer
                try:
                    status, total, wall, components, extra = _run_assembly(
                        ass, args, paths['run'], split=(seq_optimizer == 'divide'))
                finally:
                    (args.use_previous_sdf, args.planner, args.generator,
                     args.seq_optimizer) = _saved
                record = {'id': aid, 'n_parts': len(ass.objects), 'status': status,
                          'total_s': total, 'components': components,
                          'wall_s': wall, 'label': label, 'created': time.time(),
                          'assembly_dir': str(ass.assembly_dir),
                          'fingerprint': fingerprint, **extra}
                # Held parts / pull direction of the timed sequence; not for a
                # split run, whose timed order is not stats['sequence'].
                if status == 'ok' and seq_optimizer is None:
                    record['metrics'] = _plan_metrics(paths['run'])
                _write_json(paths['record'], record)
                records[aid] = record
                print(f'[optuna] {label} {aid}: {status}'
                      + (f'  total={total:.2f}s' if status == 'ok' else '')
                      + (f'  split={extra["split"]}' if extra else '')
                      + f'  ({wall:.0f}s)')
            finally:
                heartbeat.close()
                os.remove(str(lock))
        if waiting and not wait:
            break
        if waiting and len(waiting) == len(pending):
            ids = [str(a.id) for a in waiting]
            if ids != announced:
                print(f'[optuna] waiting for {label} run(s) another process is computing: '
                      f'{ids} (locks under {Path(store) / "runs"})')
                announced = ids
            if deadline is not None and time.time() > deadline:
                print(f'[optuna] {label}: stopped waiting for {ids}, time budget used up')
                break
            time.sleep(_RUN_POLL_S)
        pending = waiting
    return records


def import_run_into_store(run_dir, dataset_dir, store=None):
    """Copy the planned runs of a run directory from before the shared store
    (``<run>/baselines``, ``heur-out``, ``eval_*/runs``, ``eval_*/split_runs``
    and the trials' ``trial_NNNN/<id>``) into `store`, so later runs reuse
    them. `dataset_dir` holds the assemblies (``<dataset_dir>/<id>/*.obj``),
    whose geometry keys the store. Records keep their fingerprints, so an
    imported run is reused only under the conditions it was made in. A trial
    run has no record of its own: its fingerprint is its assembly's baseline
    fingerprint with the trial's weights, and a run without stats (a trial
    cut short) is left out. Existing store entries are never overwritten.
    Returns {label: number imported}."""
    import types
    run_dir, dataset_dir = Path(run_dir), Path(dataset_dir)
    store = Path(store or DEFAULT_STORE)
    counts = {}

    def put(aid, label, fingerprint, record, src_run):
        assembly_dir = dataset_dir / aid
        if not assembly_dir.is_dir():
            print(f'[import] {label} {aid}: no {assembly_dir}; skipped')
            return
        paths = _store_paths(store, assembly_dir, fingerprint)
        if paths['record'].exists():
            return
        paths['dir'].mkdir(parents=True, exist_ok=True)
        _note_assembly(paths, types.SimpleNamespace(
            id=aid, assembly_dir=assembly_dir,
            objects={p.stem: None for p in assembly_dir.glob('*.obj')}))
        if Path(src_run).is_dir():
            shutil.copytree(str(src_run), str(paths['run']), dirs_exist_ok=True)
        _write_json(paths['weights'], fingerprint.get('weights'))
        record = dict(record, label=label, fingerprint=fingerprint,
                      assembly_dir=str(assembly_dir), imported_from=str(run_dir))
        if (record.get('status') == 'ok' and 'metrics' not in record
                and fingerprint.get('seq_optimizer') is None):
            record['metrics'] = _plan_metrics(paths['run'])
        _write_json(paths['record'], record)
        counts[label] = counts.get(label, 0) + 1

    record_dirs = [('baseline', run_dir / 'baselines'), ('heur-out', run_dir / 'heur-out')]
    for eval_dir in sorted(run_dir.glob('eval_*')):
        record_dirs += [('trained', eval_dir / 'runs'), ('trained+split', eval_dir / 'split_runs')]
    baseline_fp = {}
    for label, d in record_dirs:
        for rec_path in sorted(d.glob('*.json')):
            if rec_path.name == 'weights.json':
                continue
            try:
                with open(rec_path) as _f:
                    record = json.load(_f)
            except (OSError, json.JSONDecodeError):
                continue
            if 'fingerprint' not in record:
                continue
            aid = rec_path.stem
            if label == 'baseline':
                baseline_fp[aid] = record['fingerprint']
            put(aid, label, record['fingerprint'], record, d / f'{aid}_run')

    for trial_dir in sorted(run_dir.glob('trial_*')):
        try:
            with open(trial_dir / 'weights.json') as _f:
                weights = json.load(_f)
        except (OSError, json.JSONDecodeError):
            continue
        for sub in sorted(p for p in trial_dir.iterdir() if p.is_dir()):
            aid = sub.name
            if aid not in baseline_fp:
                continue
            parts = {p.stem: None for p in (dataset_dir / aid).glob('*.obj')}
            try:
                status, total, components, _extra = _assess_run(
                    sub, types.SimpleNamespace(id=aid, objects=parts))
            except KeyboardInterrupt:
                continue  # the plan was interrupted, not finished
            if status == 'no_stats':
                continue
            fingerprint = json.loads(json.dumps(dict(baseline_fp[aid], weights=weights)))
            put(aid, 'trial', fingerprint,
                {'id': aid, 'n_parts': len(parts), 'status': status, 'total_s': total,
                 'components': components, 'wall_s': None, 'trial': trial_dir.name}, sub)
    # The candidate-check cache: its directories are keyed by geometry and
    # configuration already, and shards are per process, so they merge by copy.
    if (run_dir / 'sim_cache').is_dir():
        shutil.copytree(str(run_dir / 'sim_cache'), str(store / 'sim_cache'), dirs_exist_ok=True)
    print(f'[import] {run_dir} -> {store}: {counts}')
    return counts


# ----------------------------------------------------------------------------
# Store maintenance: sequence selection for runs stored before it existed
# ----------------------------------------------------------------------------

def _store_index(store):
    """{(assembly id, fingerprint key): record path} over the whole store."""
    index = {}
    for d in (Path(store) / 'runs').glob('*'):
        try:
            with open(d / 'assembly.json') as _f:
                aid = str(json.load(_f)['id'])
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        for rec in d.glob('*.json'):
            if rec.name != 'assembly.json' and not rec.name.endswith('.weights.json'):
                index[(aid, rec.stem)] = rec
    return index


@contextlib.contextmanager
def _settings_as(values):
    """Temporarily set the settings a stored run was made with (its
    fingerprint's), so re-timing it reproduces its conditions."""
    import settings
    saved = {k: getattr(settings, k) for k in values if hasattr(settings, k)}
    missing = [k for k in values if not hasattr(settings, k)]
    for k, v in values.items():
        setattr(settings, k, tuple(v) if isinstance(v, list) and isinstance(saved.get(k), tuple) else v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)
        for k in missing:
            delattr(settings, k)


def _derive_selected(rec_path, record, store, mode, num_proc):
    """The run `record` would have been with sequence selection `mode`: the
    planning is identical up to the selection, so the stored tree is reused;
    the selected sequence is re-timed only when it differs. Returns the new
    record, written next to the old one under the new fingerprint."""
    import pickle
    import types
    from plan_sequence.optimizer.base import select_min_cost_sequence
    from plan_sequence.planner.heuristic import HeuristicDFASequencePlanner

    fingerprint = json.loads(json.dumps(dict(record['fingerprint'], sequence_selection=mode)))
    d = Path(rec_path).parent
    key = _fingerprint_key(fingerprint)
    new_rec, new_run = d / f'{key}.json', d / f'{key}_run'
    old_run = d / f'{Path(rec_path).stem}_run'
    base = {k: v for k, v in record.items() if k not in ('fingerprint', 'created')}
    base.update(fingerprint=fingerprint, created=time.time(), derived_from=Path(rec_path).stem)

    def alias():
        # Same plan as the old run: point at its output instead of copying.
        if not new_run.exists() and old_run.exists():
            os.symlink(old_run.name, new_run)

    if record.get('status') != 'ok':
        # A failed plan returns before any selection, in the pipeline too.
        alias()
        base['sequence_selection'] = {'mode': mode, 'changed': False, 'reason': 'plan not complete'}
        return base, new_rec
    with open(old_run / 'log' / 'stats.json') as _f:
        stats = json.load(_f)
    with open(old_run / 'log' / 'tree.pkl', 'rb') as _f:
        tree = pickle.load(_f)
    assembly_dir = record['assembly_dir']
    parts = sorted(max(tree.nodes, key=len))
    edge_cost = HeuristicDFASequencePlanner.edge_scorer(
        os.path.abspath('assets'), assembly_dir, parts, fingerprint['weights'])
    first = list(stats['sequence'])
    chosen, cost, first_cost = select_min_cost_sequence(tree, edge_cost, prefer=first)
    info = {'mode': mode, 'changed': chosen is not None and list(chosen) != first,
            'cost': cost, 'first_cost': first_cost, 'first_sequence': first}
    base['sequence_selection'] = info
    if not info['changed']:
        alias()
        return base, new_rec

    # Re-time the selected sequence exactly as _render_plan does.
    if new_run.is_symlink() or new_run.is_file():
        new_run.unlink()
    elif new_run.exists():
        shutil.rmtree(str(new_run))  # a half-written earlier attempt
    shutil.copytree(str(old_run), str(new_run), symlinks=True)
    log = new_run / 'log'
    for name in ('timing_overview.json', 'arm_plans.json', 'timing_overview_split.json'):
        if (log / name).exists():
            (log / name).unlink()
    stats['sequence'] = list(chosen)
    stats['sequence_selection'] = info
    _write_json(log / 'stats.json', stats)
    from plan_robot.arm_pipeline import plan_arm_sequence
    fp_settings = fingerprint.get('settings') or {}
    gripper_type = ('rod' if fp_settings.get('contact_model', 'rod') == 'rod'
                    else (fingerprint.get('args') or {}).get('gripper_type') or 'rod')
    gripper_scale = (fingerprint.get('args') or {}).get('gripper_scale') or 0.4
    t0 = time.time()
    with _settings_as({k: v for k, v in fp_settings.items() if v is not None}):
        plan_arm_sequence(os.path.abspath(os.path.join('ASAPx', 'assets')), assembly_dir,
                          list(chosen), tree, gripper_type=gripper_type,
                          gripper_scale=gripper_scale, log_dir=str(log), num_proc=num_proc)
    objects = {p.stem: None for p in Path(assembly_dir).glob('*.obj')}
    status, total, components, _extra = _assess_run(
        new_run, types.SimpleNamespace(id=record.get('id'), objects=objects))
    base.update(status=status, total_s=total, components=components,
                metrics=_plan_metrics(new_run) if status == 'ok' else None,
                retime_s=time.time() - t0)
    return base, new_rec


def derive_selected_runs(store=None, mode='min_cost', deadline=None, num_proc=10):
    """For every stored heuristic run made before sequence selection existed
    (no 'sequence_selection' in its fingerprint, i.e. the first sequence
    found), store the run selection `mode` would have made, under the
    fingerprint a current run gets -- so every later training, evaluation or
    warm start reuses it. Locked per run, so an array of workers shares the
    work; resumable (done runs are skipped). Run from the repository root.
    Returns {'derived': n, 'changed': n, 'skipped': n}."""
    store = Path(store or DEFAULT_STORE)
    todo = []
    for rec_path in sorted((store / 'runs').glob('*/*.json')):
        if rec_path.name == 'assembly.json' or rec_path.name.endswith('.weights.json'):
            continue
        try:
            with open(rec_path) as _f:
                record = json.load(_f)
        except (OSError, json.JSONDecodeError):
            continue
        fp = record.get('fingerprint') or {}
        if fp.get('planner') == 'heuristic' and not fp.get('seq_optimizer') \
                and 'sequence_selection' not in fp and record.get('assembly_dir'):
            todo.append((rec_path, record))
    # Every worker walks the list in its own order, so they rarely meet on a lock.
    random.Random(os.getpid()).shuffle(todo)
    counts = {'derived': 0, 'changed': 0, 'skipped': 0, 'failed': 0}
    print(f'[select] {len(todo)} stored heuristic runs without sequence selection')
    for rec_path, record in todo:
        if deadline is not None and time.time() > deadline:
            print('[select] time budget used up; the rest is left for the next run')
            break
        fingerprint = json.loads(json.dumps(dict(record['fingerprint'], sequence_selection=mode)))
        new_rec = rec_path.parent / f'{_fingerprint_key(fingerprint)}.json'
        if new_rec.exists():
            counts['skipped'] += 1
            continue
        lock = _try_lock(new_rec.with_suffix('.lock'))
        if lock is None:
            continue
        heartbeat = _LockHeartbeat(lock)
        try:
            if new_rec.exists():
                continue
            try:
                derived, path = _derive_selected(rec_path, record, store, mode, num_proc)
            except Exception as _e:
                traceback.print_exc()
                print(f'[select] {record.get("id")} {rec_path.name}: failed ({_e}); left for the next run')
                counts['failed'] += 1
                continue
            _write_json(path, derived)
            counts['derived'] += 1
            sel = derived['sequence_selection']
            if sel.get('changed'):
                counts['changed'] += 1
                print(f'[select] {record.get("id")} {record.get("label")}: '
                      f'{record.get("total_s"):.2f}s -> '
                      + (f'{derived["total_s"]:.2f}s' if derived.get('status') == 'ok'
                         else derived.get('status'))
                      + ('  (the first sequence was not a feasible path of the tree)'
                         if sel['first_cost'] is None
                         else f'  (cost {sel["first_cost"]:.3f} -> {sel["cost"]:.3f})'))
        finally:
            heartbeat.close()
            os.remove(str(lock))
    print(f'[select] done: {counts}')
    return counts


def selection_report(run_dir, store=None, mode='min_cost'):
    """How sequence selection `mode` changes a run: per stored run, the time
    of the selected sequence against the first one found; and the training
    history re-scored with selected sequences (every trial's objective, the
    best trial, its weights -> <run>/heuristic_weights_<mode>.json). Needs
    derive_selected_runs first. Writes <run>/sequence_selection_report.{txt,json}."""
    run_dir = Path(run_dir)
    store = Path(store or DEFAULT_STORE)
    index = _store_index(store)

    def lookup(aid, fingerprint):
        path = index.get((aid, _fingerprint_key(json.loads(json.dumps(fingerprint)))))
        if path is None:
            return None
        with open(path) as _f:
            return json.load(_f)

    def pair(aid, fingerprint):
        old = lookup(aid, fingerprint)
        new = lookup(aid, dict(fingerprint, sequence_selection=mode)) if old else None
        return old, new

    # Baseline fingerprints: the run's own (legacy baselines/ dir) or its
    # history's per-assembly records.
    base_fp = {}
    for rec in sorted((run_dir / 'baselines').glob('*.json')):
        if rec.name != 'weights.json':
            with open(rec) as _f:
                r = json.load(_f)
            if 'fingerprint' in r:
                base_fp[rec.stem] = r['fingerprint']
    try:
        with open(run_dir / 'history.json') as _f:
            history = json.load(_f)
    except (OSError, json.JSONDecodeError):
        history = []
    for e in history:
        for aid, path in (e.get('per_assembly_record') or {}).items():
            if aid not in base_fp and Path(path).exists():
                with open(path) as _f:
                    fp = json.load(_f)['fingerprint']
                fp = {k: v for k, v in fp.items() if k != 'sequence_selection'}
                base_fp[aid] = dict(fp, weights=json.loads(json.dumps(
                    _pin(_reference_weights(), _training_config()['fixed_weights']))))

    lines = [f'Sequence selection "{mode}" vs the first sequence found: {run_dir}', '=' * 78]
    out = {'run': str(run_dir), 'mode': mode}

    # History, re-scored.
    rescored = []
    missing = 0
    for e in history:
        if e.get('outcome') != 'complete' or e.get('n_failed') or not e.get('per_assembly_total_s'):
            continue
        logs_old, logs_new, ok = [], [], True
        for aid in e['per_assembly_total_s']:
            if aid not in base_fp:
                ok = False
                break
            b_old, b_new = pair(aid, base_fp[aid])
            t_old, t_new = pair(aid, dict(base_fp[aid], weights=e['weights']))
            if not all(r and r.get('status') == 'ok' for r in (b_old, b_new, t_old, t_new)):
                ok = False
                break
            logs_old.append(math.log(t_old['total_s'] / b_old['total_s']))
            logs_new.append(math.log(t_new['total_s'] / b_new['total_s']))
        if not ok:
            missing += 1
            continue
        rescored.append({'trial': e['trial'], 'weights': e['weights'],
                         'objective_first': sum(logs_old) / len(logs_old),
                         'objective_selected': sum(logs_new) / len(logs_new)})
    out['history'] = rescored
    if not rescored:
        lines += [f'training history: no trial re-scored yet ({missing} complete trials lack '
                  f'selected runs for some assembly; run select to completion)'
                  if history else 'training history: none in this run directory', '']
    if rescored:
        b1 = min(rescored, key=lambda r: r['objective_first'])
        b2 = min(rescored, key=lambda r: r['objective_selected'])
        out['best_first'], out['best_selected'] = b1, b2
        _write_json(run_dir / f'heuristic_weights_{mode}.json', b2['weights'])
        lines += [f'training history: {len(rescored)} complete trials re-scored'
                  + (f' ({missing} could not be: runs missing from the store)' if missing else ''),
                  f'  best trial, first sequence:     {b1["trial"]:4d}  x{math.exp(b1["objective_first"]):.4f} of its baseline',
                  f'  best trial, selected sequence:  {b2["trial"]:4d}  x{math.exp(b2["objective_selected"]):.4f} of its baseline'
                  f' (weights -> heuristic_weights_{mode}.json)', '']

    # Every derived run in the store: selected vs first.
    def label_of(r):
        label = r.get('label') or '?'
        return 'trial' if label.startswith('trial') else label

    by_label = {}
    for (aid, _key), path in index.items():
        with open(path) as _f:
            r = json.load(_f)
        sel = r.get('sequence_selection') or {}
        if sel.get('mode') != mode or not r.get('derived_from'):
            continue
        old_path = path.parent / f'{r["derived_from"]}.json'
        if not old_path.exists():
            continue
        with open(old_path) as _f:
            old = json.load(_f)
        if old.get('status') != 'ok' or r.get('status') != 'ok':
            by_label.setdefault(label_of(r), []).append(None)
            continue
        by_label.setdefault(label_of(r), []).append(math.log(r['total_s'] / old['total_s']))
    lines.append('whole store (all runs), selected / first sequence time per stored run:')
    out['store'] = {}
    for label in sorted(by_label, key=lambda x: (x == 'trial', x)):
        vals = [v for v in by_label[label] if v is not None]
        if not vals:
            continue
        s = {'n': len(vals), 'changed': sum(abs(v) > _TIE_LOG_RATIO for v in vals),
             'faster': sum(v < -_TIE_LOG_RATIO for v in vals),
             'slower': sum(v > _TIE_LOG_RATIO for v in vals),
             'geomean_ratio': _geomean(vals),
             'geomean_ratio_changed': _geomean([v for v in vals if abs(v) > _TIE_LOG_RATIO])}
        out['store'][label] = s
        lines.append(f'  {label:<13} n={s["n"]:4d}  changed {s["changed"]:4d}  faster/slower {s["faster"]}/{s["slower"]}'
                     f'  geomean x{s["geomean_ratio"]:.4f}'
                     + (f'  (changed only x{s["geomean_ratio_changed"]:.4f})' if s['changed'] else ''))
    text = '\n'.join(lines) + '\n'
    _write_json(run_dir / 'sequence_selection_report.json', out)
    (run_dir / 'sequence_selection_report.txt').write_text(text)
    print(text)
    return out


@contextlib.contextmanager
def _pipeline_environment(args, store, cfg):
    """Configuration shared by training and evaluation, restored on exit: the
    planner reads its weights from the file each run points
    heuristic_weights_optuna_path at; runs render no media (the arm pipeline
    still runs inside _render_plan, so render_sequence stays on), share the
    candidate-check cache under `store` and reuse SDFs generated by an
    assembly's first (baseline) plan."""
    import settings

    saved_settings = {k: getattr(settings, k, None) for k in (
        'heuristic_weights_source', 'heuristic_weights_optuna_path',
        'render_sequence', 'render_gifs')}
    saved_args = {k: getattr(args, k, None) for k in (
        'planner', 'generator', 'plan_arm', 'sim_cache_dir', 'use_previous_sdf')}
    settings.heuristic_weights_source = 'optuna'
    settings.render_sequence = True
    settings.render_gifs = bool(cfg['render_gifs'])
    args.planner = 'heuristic'
    args.generator = 'rand'
    args.plan_arm = True
    args.sim_cache_dir = (str(Path(store) / 'sim_cache')
                          if getattr(settings, 'sim_cache', False) else None)
    args.use_previous_sdf = True
    try:
        yield
    finally:
        for k, v in saved_settings.items():
            if v is None and hasattr(settings, k):
                delattr(settings, k)
            elif v is not None:
                setattr(settings, k, v)
        for k, v in saved_args.items():
            setattr(args, k, v)


# ----------------------------------------------------------------------------
# Study
# ----------------------------------------------------------------------------

def _warm_start_weights(sources, top):
    """Weights worth re-evaluating from earlier runs: from each history
    (a run directory's history.json, or the file itself), the `top` fully
    evaluated trials without failures by time, plus that history's Pareto
    front when it was a multi-objective study. Their values are not reused
    -- a new study may train on other assemblies -- only the weights, and a
    weight set's stored runs make it free on the assemblies it has seen."""
    out, seen = [], set()
    for src in sources:
        path = Path(src)
        path = path / 'history.json' if path.is_dir() else path
        try:
            with open(path) as _f:
                history = json.load(_f)
        except (OSError, json.JSONDecodeError) as _e:
            print(f'[optuna] WARN warm start: cannot read {path} ({_e})')
            continue
        done = [e for e in history if e.get('outcome') == 'complete' and not e.get('n_failed')
                and e.get('objective') is not None]
        picked = sorted(done, key=lambda e: e['objective'])[:top]
        with_obj = [e for e in done if e.get('objectives')]
        if with_obj:
            names = list(with_obj[0]['objectives'])
            vals = [[e['objectives'][o] for o in names] for e in with_obj]
            picked += [e for e, v in zip(with_obj, vals)
                       if not any(all(x <= y for x, y in zip(w, v)) and w != v for w in vals)]
        for e in picked:
            key = json.dumps(e['weights'], sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append({k: float(e['weights'][k]) for k in WEIGHT_KEYS})
    return out


def _trial_owner():
    """Who runs a trial: host, pid and the Slurm job (the array's, so all
    tasks of one submission share it)."""
    return {'host': socket.gethostname(), 'pid': os.getpid(),
            'slurm_job': os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID')}


def _fail_orphaned_trials(study):
    """Mark as failed the trials a killed worker left RUNNING (scancel, time
    limit, out of memory): they would otherwise stay in the journal for good,
    and constant-liar TPE keeps treating their points as bad. A trial is
    orphaned when its owner was another Slurm submission (the submit script
    refuses to run two on one run directory), or a process on this host that
    is gone."""
    import optuna
    me = _trial_owner()
    n = 0
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        owner = t.user_attrs.get('owner') or {}
        dead = bool(me['slurm_job'] and owner.get('slurm_job')
                    and owner['slurm_job'] != me['slurm_job'])
        if not dead and owner.get('host') == me['host'] and owner.get('pid') != me['pid']:
            try:
                os.kill(int(owner['pid']), 0)
            except ProcessLookupError:
                dead = True
            except (PermissionError, TypeError, ValueError):
                pass
        if dead:
            try:
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                n += 1
            except Exception:
                pass  # another worker got there first
    if n:
        print(f'[optuna] marked {n} trial(s) left running by a killed worker as failed')


def _constraints_func(trial):
    # Optuna constraint: <= 0 is feasible. A trial without the attribute (e.g.
    # from a study written before failures were counted) reads as infeasible.
    return (float(trial.user_attrs.get('n_failed', 1)),)


def _write_history(study, history_path):
    """Rebuild the history file from the study's trials. The study is the
    record (shared by every worker with journal storage), so no worker can
    overwrite another's entries."""
    entries = [t.user_attrs['entry'] for t in study.get_trials(deepcopy=False)
               if 'entry' in t.user_attrs]
    entries.sort(key=lambda e: e['trial'])
    try:
        _write_json(history_path, entries)
    except Exception as _e:
        print(f'[optuna] WARN history write failed: {_e}')


def train_heuristic_weights(test_eval, args,
                            n_trials=50, seed=42,
                            search_space=None,
                            output_root=None,
                            weights_path=None,
                            history_path=None,
                            persist_study=False,
                            time_budget_s=None,
                            store=None,
                            warm_start=None):
    """Run an Optuna study minimising the mean log time ratio against the
    baseline across `test_eval.assemblies`. `n_trials` is the study-wide
    number of trials, so a resumed study or several workers stop once it is
    reached; 0 only computes the baselines. See module docstring.

    `time_budget_s`: wall-clock budget for this process, counted from the
    call (e.g. a cluster job's time limit minus a margin). No trial starts
    after it, and a trial ends before an assembly whose planning would run
    past it (estimated from that assembly's baseline wall time), with
    outcome 'deadline'. The weights file always holds the best trial so far,
    so the budget only decides how many trials run.

    `search_space`: {weight: (low, high)} for the weights that are not fixed,
    sampled log-uniformly; defaults to settings.heuristic_training's bounds.

    `store`: where every planned run is kept (default DEFAULT_STORE), shared
    with other runs: a trial whose weights were already evaluated on an
    assembly under the same conditions reuses that result.

    `warm_start`: earlier run directories (or history files) whose best
    weights are queued as the first trials (see _warm_start_weights); on the
    assemblies they were evaluated on, they come from the store for free.

    Returns the `optuna.Study` so the caller can introspect.
    """
    try:
        import optuna
    except ImportError as _e:
        raise ImportError(
            "Optuna is required for train_heuristic_weights but is not installed. "
            "pip install optuna"
        ) from _e
    deadline = None if time_budget_s is None else time.time() + float(time_budget_s)
    cfg = _training_config()
    fixed = cfg['fixed_weights']
    free_keys = [k for k in WEIGHT_KEYS if k not in fixed]
    lo, hi = cfg['search_bounds']
    search_space = dict(search_space or {k: (lo, hi) for k in free_keys})
    if not fixed:
        print('[optuna] WARN no weight is fixed: the cost is scale-invariant, so one '
              'search dimension is redundant')

    weights_path = Path(weights_path or 'assets/heuristic_weights_optuna.json')
    history_path = Path(history_path or 'assets/heuristic_weights_optuna_history.json')
    output_root = Path(output_root or 'assets/optuna_training')
    output_root.mkdir(parents=True, exist_ok=True)
    store = Path(store or DEFAULT_STORE)
    print(f'[optuna] store: {store}')

    study = None
    env = _pipeline_environment(args, store, cfg)
    env.__enter__()
    try:
        # --- baselines (stage 0) ---
        reference = _pin(_reference_weights(), fixed)
        print(f'[optuna] reference weights (pinned): {reference}')
        baselines = _ensure_runs(test_eval.assemblies, args, store,
                                 reference, 'baseline', clear_sdf=True, deadline=deadline)
        train_ass = []
        for ass in test_eval.assemblies:
            rec = baselines.get(str(ass.id))
            if rec is None:
                print(f'[optuna] {ass.id}: left out of training (no baseline within the time budget)')
            elif rec['status'] == 'ok':
                train_ass.append(ass)
            else:
                print(f'[optuna] {ass.id}: left out of training (baseline {rec["status"]})')
        if not train_ass:
            print('[optuna] no assembly has a complete baseline plan; nothing to train on')
            return None
        # Pruner step ids: stable per assembly, independent of evaluation order.
        step_of = {aid: i for i, aid in enumerate(sorted(str(a.id) for a in train_ass))}
        print(f'[optuna] training on {len(train_ass)} assemblies: {sorted(step_of)}')

        objectives = cfg['objectives']
        multi = len(objectives) > 1
        extra_objectives = [o for o in objectives if o != 'time']
        # The baseline's values of the other objectives: what the reference
        # weights achieve, the bar of pareto_pick 'no_worse_than_reference'.
        baseline_metrics = {}
        if extra_objectives:
            for ass in list(train_ass):
                aid = str(ass.id)
                m = baselines[aid].get('metrics')
                if m is None:
                    print(f'[optuna] {aid}: left out of training (baseline plan metrics unreadable)')
                    train_ass.remove(ass)
                else:
                    baseline_metrics[aid] = m
            if not train_ass:
                print('[optuna] no assembly left to train on')
                return None
        reference_values = {'time': 0.0}
        for o in extra_objectives:
            reference_values[o] = sum(m[o] for m in baseline_metrics.values()) / len(baseline_metrics)
        if multi:
            print(f'[optuna] multi-objective: {list(objectives)} (all minimised; no pruning); '
                  f'reference weights score {reference_values}')

        # --- study ---
        storage = None
        if persist_study:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            storage = JournalStorage(JournalFileBackend(str(output_root / 'study.journal')))
        # Workers sharing a study (or a resumed one) must not replay the same
        # random start-up samples, so their seed also depends on the process.
        sampler_seed = seed if not persist_study else (seed * 1_000_003 + os.getpid()) % 2**32
        # Optuna prunes single-objective studies only.
        p_threshold = None if multi else cfg['pruner_p_threshold']
        with warnings.catch_warnings():
            # multivariate / constant_liar / constraints_func and WilcoxonPruner
            # are flagged experimental; the warnings are noise here.
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=sampler_seed,
                # The weights act through their ratios, so model them jointly.
                multivariate=True,
                # Parallel workers see running trials as provisional results
                # instead of all sampling the same point.
                constant_liar=True,
                constraints_func=_constraints_func,
            )
            pruner = (optuna.pruners.WilcoxonPruner(p_threshold=p_threshold)
                      if p_threshold is not None else optuna.pruners.NopPruner())
        study = optuna.create_study(
            study_name='heuristic_weights',
            directions=['minimize'] * len(objectives),
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=bool(storage),
        )
        _fail_orphaned_trials(study)
        n_done_initial = len(study.trials)
        if history_path.exists() and not n_done_initial:
            # Fresh study: trial numbers restart at 0, so the old history
            # belongs to another study. Move it aside.
            archived = _archived_history_path(history_path)
            try:
                os.replace(history_path, archived)
                print(f'[optuna] previous history moved to {archived}')
            except FileNotFoundError:
                pass  # another worker moved it first

        def _free(weights):
            return {k: min(max(weights[k], search_space[k][0]), search_space[k][1])
                    for k in free_keys}

        reference_params = _free(reference)
        if cfg['enqueue_reference']:
            if any(reference_params[k] != reference[k] for k in free_keys):
                print('[optuna] WARN reference weights fall outside the search bounds; the '
                      'queued reference trial is clipped and will not reproduce the baseline')
            study.enqueue_trial(reference_params, skip_if_exists=True)
        if cfg['enqueue_time_model_prior']:
            study.enqueue_trial(_free(_time_model_prior(fixed)), skip_if_exists=True)
        if warm_start:
            queued = _warm_start_weights(warm_start, cfg['warm_start_top'])
            for w in queued:
                study.enqueue_trial(_free(_pin(w, fixed)), skip_if_exists=True)
            print(f'[optuna] warm start: {len(queued)} weight sets queued from {list(warm_start)}')

        stop_on_failure = bool(cfg['stop_on_failure'])

        def objective(trial):
            trial.set_user_attr('owner', _trial_owner())
            params = {k: trial.suggest_float(k, *search_space[k], log=True) for k in free_keys}
            weights = {k: float(params[k]) if k in params else fixed[k] for k in WEIGHT_KEYS}
            is_reference = params == reference_params
            trial_label = f'trial_{trial.number:04d}'
            # The reference trial plans afresh instead of reading the stored
            # baselines it would match: it is the check that the pipeline is
            # deterministic, and that the stored runs still hold for the
            # current code (the fingerprint does not cover code changes).
            check_dir = output_root / 'reference_check'
            if is_reference:
                if check_dir.exists():
                    shutil.rmtree(str(check_dir))
                _use_weights(weights, check_dir / 'weights.json')

            order = list(train_ass)
            random.Random(trial.number).shuffle(order)
            t0 = time.time()
            per_total, per_ratio, per_status, per_metrics, per_run = {}, {}, {}, {}, {}
            log_ratios = []
            n_failed = 0
            outcome = 'complete'
            for ass in order:
                aid = str(ass.id)
                # Planning with the cache is usually faster than the cold
                # baseline, so 1.5x its wall time is a safe upper bound.
                fingerprint = _run_fingerprint(args, weights)
                paths = _store_paths(store, ass.assembly_dir, fingerprint)
                stored = None if is_reference else _load_record(paths, fingerprint)
                if (stored is None and deadline is not None
                        and time.time() + 1.5 * (baselines[aid]['wall_s'] or 0.0) > deadline):
                    outcome = 'deadline'
                    study.stop()
                    break
                if is_reference:
                    status, total, _wall, _components, _extra = _run_assembly(ass, args, check_dir / aid)
                    metrics = _plan_metrics(check_dir / aid) if status == 'ok' else None
                else:
                    rec = stored or _ensure_runs([ass], args, store, weights, trial_label).get(aid)
                    status, total = (rec['status'], rec['total_s']) if rec else ('not_run', None)
                    metrics = rec.get('metrics') if rec else None
                    per_run[aid] = str(paths['record'])
                if status == 'ok' and extra_objectives:
                    per_metrics[aid] = metrics
                    if metrics is None:
                        status = 'metrics_unreadable'
                per_status[aid] = status
                per_total[aid] = total
                if status != 'ok':
                    n_failed += 1
                    print(f'[optuna] {trial_label} / {aid}: {status}')
                    if stop_on_failure:
                        outcome = 'stopped_on_failure'
                        break
                    continue
                ratio = total / baselines[aid]['total_s']
                per_ratio[aid] = ratio
                log_ratios.append(math.log(ratio))
                print(f'[optuna] {trial_label} / {aid}: total={total:.2f}s  '
                      f'x{ratio:.3f} of baseline' + ('  (stored)' if stored else '')
                      + ''.join(f'  {o}={per_metrics[aid][o]:.3f} (baseline {baseline_metrics[aid][o]:.3f})'
                                for o in extra_objectives))
                if p_threshold is None:
                    continue
                trial.report(math.log(ratio), step_of[aid])
                if trial.should_prune():
                    outcome = 'pruned'
                    break
            elapsed = time.time() - t0

            mean_log = sum(log_ratios) / len(log_ratios) if log_ratios else None
            value = float('inf') if n_failed or mean_log is None else mean_log
            values = {'time': value}
            for o in extra_objectives:
                values[o] = (float('inf') if value == float('inf')
                             else sum(m[o] for m in per_metrics.values()) / len(per_metrics))
            trial.set_user_attr('n_failed', n_failed)  # read by _constraints_func
            trial.set_user_attr('outcome', outcome)
            entry = {
                'trial': trial.number,
                'outcome': outcome,
                'weights': weights,
                'is_reference': is_reference,
                # Mean log ratio over the assemblies evaluated; a partial
                # estimate for pruned / deadline trials, None once an
                # assembly failed.
                'objective': None if value == float('inf') else value,
                'geomean_ratio': None if value == float('inf') else math.exp(value),
                'n_failed': n_failed,
                'n_evaluated': len(per_status),
                'per_assembly_total_s': per_total,
                'per_assembly_ratio': per_ratio,
                'per_assembly_status': per_status,
                'per_assembly_record': per_run,
                'elapsed_s': elapsed,
            }
            if multi:
                entry['objectives'] = {o: (None if v == float('inf') else v)
                                       for o, v in values.items()}
                entry['per_assembly_metrics'] = per_metrics
            trial.set_user_attr('entry', entry)
            _write_history(study, history_path)
            # Best so far after every trial, so a job killed at its time limit
            # still leaves its result behind.
            _write_best(study, weights_path, final=False, objectives=objectives,
                        reference_values=reference_values, pick=cfg['pareto_pick'])

            if is_reference and outcome == 'complete':
                drift = max((abs(math.log(r)) for r in per_ratio.values()), default=0.0)
                if drift > 1e-9:
                    print(f'[optuna] WARN the reference trial differs from the baseline '
                          f'(max |log ratio| {drift:.3g}) although it plans with the same '
                          f'weights: the pipeline is not deterministic, so part of every '
                          f'difference between trials is noise')
            score = (f'geomean x{math.exp(value):.4f}' if value != float('inf')
                     else f'failed {n_failed}')
            if value != float('inf'):
                score += ''.join(f'  {o}={values[o]:.3f}' for o in extra_objectives)
            print(f'[optuna] trial {trial.number} {outcome}: {score}  '
                  f'elapsed={elapsed:.0f}s  weights={weights}')
            return tuple(values[o] for o in objectives) if multi else value

        n_complete = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = n_trials - n_complete
        print(f'[optuna] study has {n_complete} completed trials; target {n_trials}')
        print(f'[optuna] weights file:  {weights_path} (best trial so far)')
        print(f'[optuna] history file:  {history_path}')
        print(f'[optuna] output root:   {output_root}')
        print(f'[optuna] store:         {store}')
        timeout = None if deadline is None else max(0.0, deadline - time.time())
        if timeout is not None:
            print(f'[optuna] time budget: {timeout / 3600:.2f} h left for trials')
        if remaining > 0 and timeout != 0.0:
            try:
                study.optimize(
                    objective, n_trials=remaining, gc_after_trial=True, timeout=timeout,
                    # Stops every worker once the study as a whole is done.
                    callbacks=[optuna.study.MaxTrialsCallback(
                        n_trials, states=(optuna.trial.TrialState.COMPLETE,))],
                )
            except KeyboardInterrupt:
                print(f'[optuna] interrupted after {len(study.trials)} trials; '
                      f'writing best-so-far before exit')
    finally:
        if study is not None and study.trials:
            _write_history(study, history_path)
            _write_best(study, weights_path, objectives=objectives,
                        reference_values=reference_values, pick=cfg['pareto_pick'])
        env.__exit__(None, None, None)

    return study


def pareto_path_for(weights_path):
    """Where a multi-objective study writes its Pareto front."""
    weights_path = Path(weights_path)
    return weights_path.with_name(f'{weights_path.stem}_pareto.json')


def _pareto_front(trials, objectives):
    """The trials no other trial dominates (<= on every objective, < on one)."""
    vals = [[t.user_attrs['entry']['objectives'][o] for o in objectives] for t in trials]
    return [t for t, v in zip(trials, vals)
            if not any(all(x <= y for x, y in zip(w, v)) and w != v for w in vals)]


def _write_best(study, weights_path, final=True, objectives=('time',),
                reference_values=None, pick='no_worse_than_reference'):
    """The trained weights are the best trial that evaluated every assembly
    and failed none; pruned, deadline and failing trials never qualify.
    `final=False` (after each trial) writes quietly.

    With several objectives, 'best' is picked from the Pareto front of the
    qualifying trials by `pick`: 'no_worse_than_reference' is the fastest
    trial that is at most as bad as the reference weights (`reference_values`)
    on every other objective (the queued reference trial always is, so this
    falls back to 'fastest' only when that trial never completed); 'fastest'
    ignores the other objectives. The front itself goes to
    pareto_path_for(weights_path), so another point can be chosen by hand."""
    import optuna

    def time_of(t):
        return t.values[0] if t.values else None

    qualified = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.user_attrs.get('outcome') == 'complete'
                 and t.user_attrs.get('n_failed') == 0
                 and t.values is not None and all(math.isfinite(v) for v in t.values)]
    if not qualified:
        if final:
            print('[optuna] WARN no trial completed every assembly; weights file left as-is')
        return
    multi = len(objectives) > 1
    if multi:
        front = sorted(_pareto_front(qualified, objectives), key=time_of)
        _write_json(pareto_path_for(weights_path), {
            'objectives': {o: OBJECTIVES[o] for o in objectives},
            'reference_values': reference_values,
            'pick': pick,
            'front': [{'trial': t.number,
                       'objectives': t.user_attrs['entry']['objectives'],
                       'geomean_ratio': math.exp(time_of(t)),
                       'weights': t.user_attrs['entry']['weights']} for t in front],
        })
        candidates = front
        if pick == 'no_worse_than_reference' and reference_values:
            bar = [t for t in front
                   if all(t.user_attrs['entry']['objectives'][o] <= reference_values[o] + 1e-12
                          for o in objectives if o != 'time')]
            if bar:
                candidates = bar
            elif final:
                print('[optuna] WARN no Pareto trial is as good as the reference on the other '
                      'objectives; taking the fastest')
        best = min(candidates, key=time_of)
    else:
        best = min(qualified, key=time_of)
    best_weights = best.user_attrs['entry']['weights']
    _write_json(weights_path, best_weights)
    if not final:
        return
    print(f'[optuna] best trial: {best.number}  geomean time ratio vs baseline: '
          f'x{math.exp(time_of(best)):.4f}')
    if multi:
        print(f'[optuna] its objectives: {best.user_attrs["entry"]["objectives"]}; '
              f'reference: {reference_values}; picked by {pick!r} from a Pareto front of '
              f'{len(front)} trials, written to {pareto_path_for(weights_path)}')
    print(f'[optuna] best weights: {best_weights}')
    print(f'[optuna] best weights written to {weights_path}')


# ----------------------------------------------------------------------------
# Evaluation on held-out assemblies
# ----------------------------------------------------------------------------

_SIZE_BANDS = ((5, 9), (10, 14), (15, 19), (20, 26), (27, None))

# |log ratio| below this is a tie: the pipeline is deterministic, so equal
# plans give equal times to the last digit.
_TIE_LOG_RATIO = 1e-9

# The planners compared on the test split:
# (label, planner, generator, seq_optimizer, weights, what). 'reference' is the
# heuristic planner with the reference weights (its runs are the training
# baselines), 'trained' the same planner with the trained weights, 'heur-out'
# the gen:heur-out baseline of data_assembly_time, 'trained+split' the trained
# weights with the recursive subassembly plan (--seq-optimizer divide), scored
# by the plan's own timing (plan_robot/split_timing.py) -- only with `split`.
_EVAL_RUNS = (
    ('reference', 'heuristic', 'rand', None, 'reference', 'heuristic DFA planner, reference weights'),
    ('trained', 'heuristic', 'rand', None, 'trained', 'heuristic DFA planner, trained weights'),
    ('heur-out', 'gen-adapter', 'heur-out', None, 'reference', 'gen:heur-out baseline (heur-out generator)'),
    ('trained+split', 'heuristic', 'rand', 'divide', 'trained',
     'trained weights + subassembly plan, timed as carried out'),
)
_EVAL_COMPARISONS = (('trained', 'reference'), ('trained', 'heur-out'), ('heur-out', 'reference'),
                     ('reference', 'random'), ('reference-first', 'random'),
                     ('trained', 'random'), ('heur-out', 'random'),
                     ('reference', 'reference-first'))
# With random_seeds: the reference weights without sequence selection, so
# random (which cannot select) is also compared with the ranking alone.
_REFERENCE_FIRST_RUN = ('reference-first', 'heuristic', 'rand', None, 'reference',
                        'heuristic DFA planner, reference weights, first sequence found')


def _aggregate_random(ass_list, runsets, seeds):
    """One record per assembly from its random seeds: complete if any seed
    planned completely, the time the geometric mean of those, the plan
    metrics their mean; 'seeds' lists every seed's outcome."""
    out = {}
    for ass in ass_list:
        aid = str(ass.id)
        per = [(k, runsets[f'random#{k}'].get(aid)) for k in seeds]
        done = [r for _k, r in per if r]
        if not done:
            continue
        ok = [r for r in done if r.get('status') == 'ok' and r.get('total_s')]
        metrics = [r['metrics'] for r in ok if r.get('metrics')]
        out[aid] = {
            'status': 'ok' if ok else done[0].get('status'),
            'total_s': _geomean([math.log(r['total_s']) for r in ok]) if ok else None,
            'components': None,
            'metrics': ({m: sum(x[m] for x in metrics) / len(metrics)
                         for m in ('held_parts', 'non_upward')} if metrics else None),
            'seeds': [{'seed': k, 'status': r.get('status') if r else None,
                       'total_s': r.get('total_s') if r else None} for k, r in per],
        }
    return out
# 'trained+divide', 'trained+split-2w' and 'trained+split-par' are no runs of
# their own but other totals of the 'trained+split' run: its flat timing (its
# sequence without carrying out the split), its time with two workers (S and R
# of the outermost split at once) and with as many as the plan can use (every
# split, at every depth; plan_robot/split_timing.parallel_makespan).
# trained+split vs trained+divide isolates the split.
_SPLIT_COMPARISONS = (('trained+split-2w', 'trained'), ('trained+split-2w', 'reference'),
                      ('trained+split-2w', 'heur-out'), ('trained+split-2w', 'trained+split'),
                      ('trained+split', 'random'), ('trained+split-2w', 'random'),
                      ('trained+split', 'trained'), ('trained+split', 'trained+divide'),
                      ('trained+divide', 'trained'), ('trained+split', 'reference'),
                      ('trained+split', 'heur-out'),
                      ('trained+split-par', 'trained+split'),
                      ('trained+split-par', 'trained+divide'),
                      ('trained+split-par', 'trained'), ('trained+split-par', 'reference'),
                      ('trained+split-par', 'heur-out'))


def evaluate_heuristic_weights(test_eval, args, weights_path, output_root=None,
                               label='trained', time_budget_s=None, reference_only=False,
                               split=False, store=None, random_seeds=0,
                               reference_first=False, wait_for_others=True):
    """Test the weights in `weights_path` on `test_eval.assemblies` (held out
    from training) against two baselines: the heuristic planner with the
    reference weights, and gen:heur-out. Every run is planned and arm-timed
    exactly like a training trial: no media, the candidate-check cache shared,
    one stored run per assembly guarded by a lock, so the workers of a job
    array split the set between them and a re-run only computes what is
    missing. Writes `<output_root>/eval_<label>/summary.{json,txt}` and
    returns the summary.
    `time_budget_s`: no run starts, and no waiting continues, after it; the
    summary then lists what was left out.

    `reference_only`: plan only the two baselines (reference weights and
    heur-out), which do not depend on training, and stop. A cluster run does
    this on wide workers while training is still going, so the evaluation
    proper only has the trained-weight runs left. Returns None.

    Every run goes through the shared `store` (default DEFAULT_STORE): a run
    any earlier training or evaluation made under the same conditions is
    reused, so widening the test set plans only the new assemblies.

    `split`: also plan every assembly with the trained weights plus the
    recursive subassembly plan ('trained+split', in `eval_<label>/split_runs/`)
    and compare it with the other three. Its search replays the 'trained'
    run's physics from the cache; the divide optimizer, the plan's
    verification and the per-context checks of its timing are what it adds.

    `random_seeds` (k > 0): also plan every assembly k times with random
    decisions ('dfa-random': the same search, the next frontier drawn at
    random; seeds 0..k-1) and compare: is a planner better than chance. The
    random run of an assembly is the geometric mean of its complete seeds.
    `reference_first`: also the reference weights without sequence selection
    ('reference-first'): how much is the ranking vs the selection.

    Work goes assembly by assembly in the given order, so a run cut short by
    its time limit leaves whole assemblies, not a planner's column. Within an
    assembly, the reference run comes first (it regenerates the SDFs the
    others read), trained before trained+split (each replays the one before
    from the candidate-check cache); a run whose predecessor is still being
    planned elsewhere is left for later instead of planned cold beside it.
    `wait_for_others=False`: return once nothing is left that this process
    can claim, instead of waiting for the runs other processes hold (a large
    job array, followed by one summary pass).
    """
    deadline = None if time_budget_s is None else time.time() + float(time_budget_s)
    cfg = _training_config()
    output_root = Path(output_root or 'assets/optuna_training')
    store = Path(store or DEFAULT_STORE)
    reference = _pin(_reference_weights(), cfg['fixed_weights'])
    if reference_only:
        print(f'[eval] {len(test_eval.assemblies)} assemblies: reference and heur-out runs only')
        with _pipeline_environment(args, store, cfg):
            # First everything no other worker holds, then wait for the rest:
            # a worker must not sit on another's reference run while heur-out
            # runs are still unclaimed.
            for wait in (False, True):
                for name, planner, generator, seq_opt, which, _what in _EVAL_RUNS:
                    if which != 'reference':
                        continue
                    _ensure_runs(test_eval.assemblies, args, store, reference, name,
                                 clear_sdf=(name == 'reference'), deadline=deadline,
                                 planner=planner, generator=generator, seq_optimizer=seq_opt,
                                 wait=wait)
        return None
    with open(weights_path) as _f:
        loaded = json.load(_f)
    missing = [k for k in WEIGHT_KEYS if k not in loaded]
    if missing:
        raise ValueError(f'{weights_path} lacks weights {missing}')
    weights = {k: float(loaded[k]) for k in WEIGHT_KEYS}
    eval_dir = output_root / f'eval_{label}'
    print(f'[eval] {len(test_eval.assemblies)} assemblies; reference {reference}; '
          f'{label} {weights} (from {weights_path})')

    eval_runs = [r for r in _EVAL_RUNS if split or r[3] is None]
    overrides = {}
    seeds = list(range(int(random_seeds or 0)))
    if reference_first:
        eval_runs.append(_REFERENCE_FIRST_RUN)
        overrides['reference-first'] = ({}, {'sequence_selection': 'first'})
    if seeds:
        for k in seeds:
            eval_runs.append((f'random#{k}', 'dfa-random', 'rand', None, 'reference',
                              f'random decisions, seed {k}'))
            overrides[f'random#{k}'] = ({'seed': k}, {})
    # Reference first (an assembly's first plan regenerates its SDFs), then
    # the runs only this phase makes, then heur-out, which the reference-only
    # phase normally leaves done; so a shortfall there cannot starve the
    # trained runs. Two passes: first what no other worker holds, then wait.
    priority = {'reference': 0, 'reference-first': 1, 'trained': 2, 'trained+split': 3,
                'heur-out': 99}
    ordered = sorted(eval_runs, key=lambda r: priority.get(r[0], 50))
    runsets = {r[0]: {} for r in ordered}

    def prerequisite(name):
        if name == 'reference':
            return None
        return {'trained+split': 'trained'}.get(name, 'reference')

    def claim(run, ass):
        name, planner, generator, seq_opt, which, _what = run
        arg_over, set_over = overrides.get(name, ({}, {}))
        got = _ensure_runs(
            [ass], args, store, reference if which == 'reference' else weights, name,
            clear_sdf=(name == 'reference'), deadline=deadline,
            planner=planner, generator=generator, seq_optimizer=seq_opt,
            wait=False, arg_overrides=arg_over, setting_overrides=set_over)
        runsets[name].update(got)
        return bool(got)

    with _pipeline_environment(args, store, cfg):
        while True:
            progress, pending = False, 0
            for ass in test_eval.assemblies:
                aid = str(ass.id)
                for run in ordered:
                    if aid in runsets[run[0]]:
                        continue
                    pre = prerequisite(run[0])
                    if (pre is not None and aid not in runsets[pre]) or \
                            (deadline is not None and time.time() > deadline):
                        pending += 1
                        continue
                    if claim(run, ass):
                        progress = True
                    else:
                        pending += 1
            if not pending or (deadline is not None and time.time() > deadline):
                break
            if not progress:
                if not wait_for_others:
                    print(f'[eval] nothing left to claim; {pending} runs are held by other '
                          f'processes or wait for one; leaving them to those')
                    break
                time.sleep(_RUN_POLL_S)
    if seeds:
        runsets['random'] = _aggregate_random(test_eval.assemblies, runsets, seeds)
        eval_runs = [r for r in eval_runs if not r[0].startswith('random#')] + [
            ('random', None, None, None, None,
             f'random decisions (dfa-random), geometric mean over {len(seeds)} seeds')]

    summary = _summarize_evaluation(test_eval.assemblies, runsets, reference, weights,
                                    str(weights_path), eval_runs)
    _write_json(eval_dir / 'summary.json', summary)
    text = _format_evaluation(summary)
    (eval_dir / 'summary.txt').write_text(text)
    print(text)
    print(f'[eval] summary written to {eval_dir}/summary.{{json,txt}}')
    return summary


def _geomean(logs):
    return math.exp(sum(logs) / len(logs)) if logs else None


def _compare(rows, a, b):
    """Paired comparison of run set `a` against `b` over the assemblies both
    planned completely: time ratio a/b, and which assemblies only one of them
    planned (the success side, kept out of the ratio)."""
    paired = [r for r in rows
              if all(r['runs'][x]['status'] == 'ok' and r['runs'][x]['total_s'] is not None
                     for x in (a, b))]
    ratios = [r['runs'][a]['total_s'] / r['runs'][b]['total_s'] for r in paired]
    logs = [math.log(x) for x in ratios]
    out = {
        'n_paired': len(paired),
        f'failed_only_{a}': [r['id'] for r in rows
                             if r['runs'][b]['status'] == 'ok' and r['runs'][a]['status'] not in ('ok', None)],
        f'failed_only_{b}': [r['id'] for r in rows
                             if r['runs'][a]['status'] == 'ok' and r['runs'][b]['status'] not in ('ok', None)],
        'geomean_ratio': _geomean(logs),
        'median_ratio': sorted(ratios)[len(ratios) // 2] if ratios else None,
        'wins': sum(l < -_TIE_LOG_RATIO for l in logs),
        'ties': sum(abs(l) <= _TIE_LOG_RATIO for l in logs),
        'losses': sum(l > _TIE_LOG_RATIO for l in logs),
        'geomean_ratio_ci95': None,
        'wilcoxon_p': None,
    }
    # Bootstrap 95% interval of the geometric-mean ratio over assemblies.
    if len(logs) >= 2:
        rng = random.Random(0)
        means = sorted(sum(rng.choice(logs) for _ in logs) / len(logs) for _ in range(10000))
        out['geomean_ratio_ci95'] = [math.exp(means[249]), math.exp(means[9749])]
    # Wilcoxon signed-rank test on the paired log ratios (two-sided).
    if out['wins'] + out['losses'] >= 2:
        try:
            from scipy.stats import wilcoxon
            out['wilcoxon_p'] = float(wilcoxon(logs, zero_method='zsplit').pvalue)
        except Exception as _e:  # scipy missing, or a degenerate sample
            print(f'[eval] Wilcoxon test skipped ({a} vs {b}): {_e}')
    out['by_size'] = []
    for lo, hi in _SIZE_BANDS:
        band = [l for r, l in zip(paired, logs)
                if r['n_parts'] >= lo and (hi is None or r['n_parts'] <= hi)]
        if band:
            out['by_size'].append({'parts': f'{lo}-{hi}' if hi else f'{lo}+', 'n': len(band),
                                   'geomean_ratio': _geomean(band)})
    out['metric_mean'] = {}
    with_metrics = [r for r in paired if all(r['runs'][x].get('metrics') for x in (a, b))]
    if with_metrics:
        for m in ('held_parts', 'non_upward'):
            out['metric_mean'][m] = {
                side: sum(r['runs'][side]['metrics'][m] for r in with_metrics) / len(with_metrics)
                for side in (a, b)}
        out['metric_n'] = len(with_metrics)
    out['component_mean_s'] = {}
    # Only where both sides have a breakdown (the parallel total has none: it
    # is not a sum of the components).
    if paired and all(r['runs'][x]['components'] for r in paired for x in (a, b)):
        for comp in ('step_disassembly_s', 'transitions_s', 'base_travel_s',
                     'reorientation_s', 'hold_s'):
            out['component_mean_s'][comp] = {
                side: sum((r['runs'][side]['components'] or {}).get(comp, 0.0) for r in paired) / len(paired)
                for side in (a, b)}
    return out


def _summarize_evaluation(ass_list, runsets, reference, weights, weights_path, eval_runs):
    rows = []
    for ass in ass_list:
        aid = str(ass.id)
        runs = {}
        for name, _planner, _generator, seq_opt, *_ in eval_runs:
            rec = runsets[name].get(aid) or {}
            runs[name] = {'status': rec.get('status'), 'total_s': rec.get('total_s'),
                          'components': rec.get('components')}
            if seq_opt is None:
                runs[name]['metrics'] = rec.get('metrics')
            if 'seeds' in rec:
                runs[name]['seeds'] = rec['seeds']
            if 'split' in rec:
                runs[name].update({k: rec.get(k) for k in
                                   ('split', 'n_joins', 'flat_total_s', 'failure',
                                    'parallel_total_s', 'parallel2_total_s')})
                used = rec.get('split') == 'used'
                runs['trained+split-2w'] = {
                    'status': (rec.get('status') if not used or rec.get('parallel2_total_s')
                               else 'no_2_worker_timing'),
                    'total_s': rec.get('parallel2_total_s') if used else rec.get('total_s'),
                    'components': None, 'split': rec.get('split')}
                runs['trained+divide'] = {
                    'status': rec.get('status'), 'total_s': rec.get('flat_total_s'),
                    'components': rec.get('flat_components')}
                # Without a usable plan there is nothing to parallelise: the
                # flat time, as for trained+split.
                runs['trained+split-par'] = {
                    'status': rec.get('status'),
                    'total_s': (rec.get('parallel_total_s') if rec.get('split') == 'used'
                                else rec.get('total_s')),
                    'components': None, 'split': rec.get('split')}
            elif name == 'trained+split':
                runs['trained+divide'] = {'status': rec.get('status'), 'total_s': None,
                                          'components': None}
                runs['trained+split-par'] = dict(runs['trained+divide'])
                runs['trained+split-2w'] = dict(runs['trained+divide'])
        rows.append({'id': aid, 'n_parts': len(ass.objects), 'runs': runs})
    if any(r[0] == 'trained+split' for r in eval_runs):
        eval_runs = list(eval_runs) + [
            ('trained+split-2w', None, None, None, None,
             'the trained+split plan taken apart by two workers (S and R of the outermost split at once)'),
            ('trained+split-par', None, None, None, None,
             'the trained+split plan with S and R of every split taken apart in parallel'),
            ('trained+divide', None, None, None, None,
             'trained weights, divide optimizer\'s sequence timed flat (the trained+split run without the split)')]
    success = {}
    for name, *_ in eval_runs:
        statuses = [r['runs'][name]['status'] for r in rows]
        success[name] = {
            'ok': sum(s == 'ok' for s in statuses),
            'excluded': sum(s == 'excluded' for s in statuses),
            'failed': [r['id'] for r, s in zip(rows, statuses) if s not in ('ok', 'excluded', None)],
            'not_run': [r['id'] for r, s in zip(rows, statuses) if s is None],
        }
    names = [r[0] for r in eval_runs]
    comparisons = [c for c in _EVAL_COMPARISONS + _SPLIT_COMPARISONS
                   if c[0] in names and c[1] in names]
    summary = {
        'runs': {r[0]: r[-1] for r in eval_runs},
        'weights_path': weights_path,
        'reference_weights': reference,
        'weights': weights,
        'success': success,
        'comparisons': {f'{a} vs {b}': _compare(rows, a, b) for a, b in comparisons},
        'per_assembly': rows,
    }
    if 'trained+split' in names:
        # How often the subassembly plan was actually what got timed.
        kinds = [r['runs']['trained+split'].get('split') for r in rows
                 if r['runs']['trained+split']['status'] == 'ok']
        summary['split_usage'] = {
            'used': sum(k == 'used' for k in kinds),
            'no_plan': sum(k == 'none' for k in kinds),
            'infeasible': [r['id'] for r in rows
                           if r['runs']['trained+split'].get('split') == 'infeasible'],
            'untimed': [r['id'] for r in rows
                        if r['runs']['trained+split'].get('split') == 'untimed'],
        }
    return summary


def _format_evaluation(summary):
    names = list(summary['runs'])
    lines = [
        'Held-out evaluation of the trained heuristic weights',
        '=' * 78,
    ]
    lines += [f'{name:<10} {what}' for name, what in summary['runs'].items()]
    lines += [f'reference weights: {summary["reference_weights"]}',
              f'trained weights:   {summary["weights"]}', '',
              f'{len(summary["per_assembly"])} assemblies; complete plans per planner:']
    for name, s in summary['success'].items():
        lines.append(f'  {name:<13} ok {s["ok"]:>3}   precheck-excluded {s["excluded"]:>3}   '
                     f'failed {s["failed"]}' + (f'   not run (time budget) {s["not_run"]}' if s['not_run'] else ''))
    seed_runs = [x for r in summary['per_assembly'] for x in (r['runs'].get('random') or {}).get('seeds', [])]
    if seed_runs:
        n_ok = sum(x['status'] == 'ok' for x in seed_runs)
        lines.append(f'  random seeds planned completely: {n_ok} of {len(seed_runs)} '
                     f'(an assembly counts as ok above when any seed did)')
    if 'split_usage' in summary:
        u = summary['split_usage']
        lines.append(f'  trained+split timed as a subassembly plan on {u["used"]}; no plan found '
                     f'on {u["no_plan"]}; plan not executable as told (flat time used) on '
                     f'{len(u["infeasible"])} {u["infeasible"]}'
                     + (f'; split timing missing on {u["untimed"]}' if u['untimed'] else ''))
    for key, c in summary['comparisons'].items():
        a, b = key.split(' vs ')
        lines += ['', f'{key}: time ratio {a}/{b} over {c["n_paired"]} assemblies both planned',
                  f'  failed only with {a}: {c[f"failed_only_{a}"]}   only with {b}: {c[f"failed_only_{b}"]}']
        if c['geomean_ratio'] is None:
            continue
        ci = c['geomean_ratio_ci95']
        lines += [
            f'  geometric mean x{c["geomean_ratio"]:.4f}'
            + (f'   95% CI [x{ci[0]:.4f}, x{ci[1]:.4f}]' if ci else '')
            + f'   median x{c["median_ratio"]:.4f}',
            f'  {a} faster / equal / slower: {c["wins"]} / {c["ties"]} / {c["losses"]}'
            + (f'   Wilcoxon p = {c["wilcoxon_p"]:.3g}' if c['wilcoxon_p'] is not None else ''),
            '  by part count: ' + ',  '.join(f'{x["parts"]} (n={x["n"]}) x{x["geomean_ratio"]:.3f}'
                                            for x in c['by_size']),
        ]
        if c['metric_mean']:
            lines.append(f'  mean per step over {c["metric_n"]} assemblies: ' + ',  '.join(
                f'{m} {v[a]:.3f} vs {v[b]:.3f}' for m, v in c['metric_mean'].items()))
        if c['component_mean_s']:
            lines.append('  mean component time (s): ' + ',  '.join(
                f'{comp[:-2]} {v[a]:.1f} vs {v[b]:.1f}' for comp, v in c['component_mean_s'].items()
                if v[a] or v[b]))
    ratio_cols = [('trained', 'reference', 'trained/ref'), ('trained', 'heur-out', 'trained/h-o')]
    if 'random' in names:
        ratio_cols += [('reference', 'random', 'ref/random'), ('heur-out', 'random', 'h-o/random')]
        if 'reference-first' in names:
            ratio_cols.append(('reference-first', 'random', 'rfirst/random'))
    if 'trained+split' in names:
        ratio_cols += [('trained+split', 'trained', 'split/trained'),
                       ('trained+split', 'trained+divide', 'split/divide'),
                       ('trained+split-2w', 'trained', '2w/trained'),
                       ('trained+split-par', 'trained+divide', 'par/divide')]
    width = {n: max(13, len(n) + 3) + 2 for n in names}
    head = f'{"id":<7}{"parts":>6}' + ''.join(f'{n + " s":>{width[n]}}' for n in names)
    head += ''.join(f'{label:>14}' for _a, _b, label in ratio_cols)
    lines += ['', head]
    for r in summary['per_assembly']:
        cells = []
        for n in names:
            run = r['runs'][n]
            mark = {'used': ' S', 'none': ' -', 'infeasible': ' !', 'untimed': ' ?'}.get(run.get('split'), '')
            cells.append(f'{run["total_s"]:{width[n] - 2}.2f}{mark:<2}' if run['status'] == 'ok'
                         and run['total_s'] is not None
                         else f'{(run["status"] or "not run")[:width[n] - 1]:>{width[n]}}')
        ratios = []
        for a, b, _label in ratio_cols:
            ra, rb = r['runs'][a], r['runs'][b]
            ratios.append(f'{"x%.3f" % (ra["total_s"] / rb["total_s"]):>14}'
                          if ra['status'] == 'ok' and rb['status'] == 'ok' else f'{"-":>14}')
        lines.append(f'{r["id"]:<7}{r["n_parts"]:>6}' + ''.join(cells) + ''.join(ratios))
    if 'trained+split' in names:
        lines.append('(trained+split: S = timed as a subassembly plan, - = no plan found, '
                     '! = plan not executable as told, flat time used)')
    if seed_runs:
        lines += ['', 'random, per seed (s):']
        for r in summary['per_assembly']:
            seeds = (r['runs'].get('random') or {}).get('seeds') or []
            lines.append(f'  {r["id"]:<7}' + '  '.join(
                f'{x["total_s"]:8.2f}' if x['status'] == 'ok' and x['total_s'] else f'{(x["status"] or "not run")[:8]:>8}'
                for x in seeds))
    return '\n'.join(lines) + '\n'
