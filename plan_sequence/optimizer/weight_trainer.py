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

Storage layout (paths relative to repo root, configurable via settings):
- ``assets/heuristic_weights_optuna.json`` — the trained weights (all five,
  fixed ones included): the best fully evaluated trial that planned every
  assembly, rewritten after every trial so a job killed at its time limit
  still leaves its result; left as-is while none qualifies. This is the file
  `HeuristicDFASequencePlanner._load_weights` reads when
  ``settings.heuristic_weights_source == 'optuna'``. Candidates under
  evaluation are never written here.
- ``assets/optuna_training/trial_<NNNN>/`` — one trial: ``weights.json``
  (the candidate; ``settings.heuristic_weights_optuna_path`` points here
  while the trial runs, so parallel trials never read each other's weights)
  and one directory per assembly. Wiped when the trial starts.
- ``assets/optuna_training/baselines/<id>.json`` — baseline records; the
  planning output sits in ``<id>_run/``.
- ``assets/optuna_training/sim_cache/`` — the planner's candidate-check
  cache (plan_sequence/planner/sim_cache.py), shared by every trial and
  worker when ``settings.sim_cache`` is on.
- ``assets/heuristic_weights_optuna_history.json`` — per-trial log rebuilt
  from the study after every trial, so parallel workers never drop each
  other's entries. A fresh study moves an existing file aside first.
- ``assets/optuna_training/study.journal`` (when ``persist_study=True``) —
  Optuna journal storage. Safe on a shared filesystem, so several processes
  (e.g. cluster array jobs) can work on one study; ``n_trials`` is the
  study-wide total.
- ``assets/optuna_training/eval_<label>/`` — `evaluate_heuristic_weights`:
  the trained weights against the reference on held-out assemblies
  (``runs/`` per assembly, ``summary.{json,txt}``).

`main.py --optuna-dir DIR` moves all of it (weights and history included)
into DIR, so a run on a cluster is self-contained.

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
import json
import math
import os
import random
import shutil
import socket
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
}

# stats['stop_msg'] of a plan aborted by the initial stable-pose precheck
# (settings.no_stable_pose_action = 'exit').
_PRECHECK_ABORT = 'no self-stable initial pose'

# Seconds between checks while another process computes a stored run.
_RUN_POLL_S = 30

# A lock older than this is stale (its holder was killed, e.g. at a cluster
# job's time limit) and is taken over. Far above any single plan's duration.
_LOCK_STALE_S = 6 * 3600


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

def _assess_run(storage_dir, ass):
    """Classify one assembly's pipeline output. Returns (status, total_s,
    components): total_s and the timing components (timing_overview totals)
    are None unless status == 'ok'.

    'excluded' marks a precheck abort, which the weights cannot influence.
    Every other status except 'ok' is a failure.
    """
    log_dir = Path(storage_dir) / 'log'
    try:
        with open(log_dir / 'stats.json') as _f:
            stats = json.load(_f)
    except (OSError, json.JSONDecodeError):
        return 'no_stats', None, None

    stop_msg = stats.get('stop_msg')
    if stop_msg == 'interrupt':
        # The planner swallows Ctrl+C and returns what it had. Re-raise so the
        # study stops and writes its best-so-far, instead of charging the
        # interrupted plan to the weights.
        raise KeyboardInterrupt
    if stop_msg == _PRECHECK_ABORT:
        return 'excluded', None, None
    if not stats.get('success'):
        return f'plan_failed: {stop_msg}', None, None
    # A complete plan removes every part but the last, which stays put.
    sequence = stats.get('sequence') or []
    if len(set(ass.objects) - set(sequence)) > 1:
        return 'incomplete_sequence', None, None

    try:
        with open(log_dir / 'timing_overview.json') as _f:
            timing = json.load(_f)
    except (OSError, json.JSONDecodeError):
        return 'no_timing', None, None
    if len(timing.get('per_step') or []) != len(sequence):
        return 'timing_mismatch', None, None
    total = (timing.get('totals') or {}).get('total_s')
    if total is None or float(total) <= 0:
        # A complete plan always takes time; a ratio against zero is undefined.
        return 'no_timing', None, None
    return 'ok', float(total), timing['totals']


def _use_weights(weights, path):
    """Write `weights` to `path` and point the planner at it."""
    import settings
    _write_json(path, weights)
    settings.heuristic_weights_optuna_path = str(path)


def _run_assembly(ass, args, run_dir):
    """Plan + arm-time one assembly into `run_dir`, which is wiped first:
    get_assembly_plans_ASAP returns a leftover sequence.json without planning.
    Returns (status, total_s, wall_s, components)."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(str(run_dir))
    run_dir.mkdir(parents=True)
    _saved_storage = ass.storage_dir
    t0 = time.time()
    try:
        ass.storage_dir = run_dir
        ass.planner.get_assembly_plans(args)
        status, total, components = _assess_run(run_dir, ass)
    except Exception as _e:
        traceback.print_exc()
        status, total, components = f'error: {_e}', None, None
    finally:
        ass.storage_dir = _saved_storage
    return status, total, time.time() - t0, components


# ----------------------------------------------------------------------------
# Stored runs (baselines, evaluation)
# ----------------------------------------------------------------------------

def _run_fingerprint(args, weights, planner='heuristic', generator='rand'):
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
    fp = {
        'planner': planner,
        'generator': generator,
        'weights': weights,
        'settings': {k: getattr(settings, k, None) for k in setting_keys},
        'args': {k: getattr(args, k, None) for k in arg_keys},
    }
    # JSON round trip so a fresh fingerprint compares equal to a stored one
    # (tuples become lists).
    return json.loads(json.dumps(fp, default=str))


def _try_lock(path):
    """Create `path` exclusively; None if another live process holds it. A
    lock left by a dead process on this host, or older than _LOCK_STALE_S
    (a holder on another node that was killed), is taken over."""
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
            if time.time() - float(holder.get('since', time.time())) > _LOCK_STALE_S:
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


def _ensure_runs(ass_list, args, run_root, weights, label, clear_sdf=False,
                 deadline=None, planner='heuristic', generator='rand'):
    """One stored run per assembly with fixed `weights`, computing the missing
    ones: records in `run_root/<id>.json`, planning output in
    `run_root/<id>_run/`. Several processes can run this at once (e.g. the
    workers of a job array): each assembly is computed by whichever process
    takes its lock, and the others pick up the result.

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
    baseline, which ignores the weights."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint(args, weights, planner, generator)
    _use_weights(weights, run_root / 'weights.json')

    records = {}
    pending = list(ass_list)
    announced = None
    while pending:
        waiting = []
        for ass in pending:
            aid = str(ass.id)
            record_path = run_root / f'{aid}.json'
            try:
                with open(record_path) as _f:
                    record = json.load(_f)
                if record.get('fingerprint') == fingerprint:
                    records[aid] = record
                    continue
                print(f'[optuna] {label} {aid}: settings changed since it was computed; recomputing')
            except (OSError, json.JSONDecodeError):
                pass
            if deadline is not None and time.time() > deadline:
                print(f'[optuna] {label} {aid}: skipped, time budget used up')
                continue
            lock = _try_lock(run_root / f'{aid}.lock')
            if lock is None:
                waiting.append(ass)
                continue
            try:
                _saved = (getattr(args, 'use_previous_sdf', False), args.planner, args.generator)
                if clear_sdf:
                    args.use_previous_sdf = False
                args.planner, args.generator = planner, generator
                try:
                    status, total, wall, components = _run_assembly(
                        ass, args, run_root / f'{aid}_run')
                finally:
                    args.use_previous_sdf, args.planner, args.generator = _saved
                record = {'id': aid, 'n_parts': len(ass.objects), 'status': status,
                          'total_s': total, 'components': components,
                          'wall_s': wall, 'fingerprint': fingerprint}
                _write_json(record_path, record)
                records[aid] = record
                print(f'[optuna] {label} {aid}: {status}'
                      + (f'  total={total:.2f}s' if status == 'ok' else '')
                      + f'  ({wall:.0f}s)')
            finally:
                os.remove(str(lock))
        if waiting and len(waiting) == len(pending):
            ids = [str(a.id) for a in waiting]
            if ids != announced:
                print(f'[optuna] waiting for {label} run(s) another process is computing: '
                      f'{ids} (locks in {run_root}; delete a lock whose process is gone)')
                announced = ids
            if deadline is not None and time.time() > deadline:
                print(f'[optuna] {label}: stopped waiting for {ids}, time budget used up')
                break
            time.sleep(_RUN_POLL_S)
        pending = waiting
    return records


@contextlib.contextmanager
def _pipeline_environment(args, output_root, cfg):
    """Configuration shared by training and evaluation, restored on exit: the
    planner reads its weights from the file each run points
    heuristic_weights_optuna_path at; runs render no media (the arm pipeline
    still runs inside _render_plan, so render_sequence stays on), share the
    candidate-check cache under `output_root` and reuse SDFs generated by an
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
    args.sim_cache_dir = (str(Path(output_root) / 'sim_cache')
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
                            time_budget_s=None):
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

    study = None
    env = _pipeline_environment(args, output_root, cfg)
    env.__enter__()
    try:
        # --- baselines (stage 0) ---
        reference = _pin(_reference_weights(), fixed)
        print(f'[optuna] reference weights (pinned): {reference}')
        baselines = _ensure_runs(test_eval.assemblies, args, output_root / 'baselines',
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

        # --- study ---
        storage = None
        if persist_study:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            storage = JournalStorage(JournalFileBackend(str(output_root / 'study.journal')))
        # Workers sharing a study (or a resumed one) must not replay the same
        # random start-up samples, so their seed also depends on the process.
        sampler_seed = seed if not persist_study else (seed * 1_000_003 + os.getpid()) % 2**32
        p_threshold = cfg['pruner_p_threshold']
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
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=bool(storage),
        )
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

        stop_on_failure = bool(cfg['stop_on_failure'])

        def objective(trial):
            params = {k: trial.suggest_float(k, *search_space[k], log=True) for k in free_keys}
            weights = {k: float(params[k]) if k in params else fixed[k] for k in WEIGHT_KEYS}
            is_reference = params == reference_params
            trial_label = f'trial_{trial.number:04d}'
            trial_dir = output_root / trial_label
            if trial_dir.exists():
                # Trial numbers restart at 0 in every study that isn't resumed.
                shutil.rmtree(str(trial_dir))
            _use_weights(weights, trial_dir / 'weights.json')

            order = list(train_ass)
            random.Random(trial.number).shuffle(order)
            t0 = time.time()
            per_total, per_ratio, per_status = {}, {}, {}
            log_ratios = []
            n_failed = 0
            outcome = 'complete'
            for ass in order:
                aid = str(ass.id)
                # Planning with the cache is usually faster than the cold
                # baseline, so 1.5x its wall time is a safe upper bound.
                if deadline is not None and time.time() + 1.5 * baselines[aid]['wall_s'] > deadline:
                    outcome = 'deadline'
                    study.stop()
                    break
                status, total, _wall, _components = _run_assembly(ass, args, trial_dir / aid)
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
                      f'x{ratio:.3f} of baseline')
                trial.report(math.log(ratio), step_of[aid])
                if p_threshold is not None and trial.should_prune():
                    outcome = 'pruned'
                    break
            elapsed = time.time() - t0

            mean_log = sum(log_ratios) / len(log_ratios) if log_ratios else None
            value = float('inf') if n_failed or mean_log is None else mean_log
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
                'elapsed_s': elapsed,
            }
            trial.set_user_attr('entry', entry)
            _write_history(study, history_path)
            # Best so far after every trial, so a job killed at its time limit
            # still leaves its result behind.
            _write_best(study, weights_path, final=False)

            if is_reference and outcome == 'complete':
                drift = max((abs(math.log(r)) for r in per_ratio.values()), default=0.0)
                if drift > 1e-9:
                    print(f'[optuna] WARN the reference trial differs from the baseline '
                          f'(max |log ratio| {drift:.3g}) although it plans with the same '
                          f'weights: the pipeline is not deterministic, so part of every '
                          f'difference between trials is noise')
            score = (f'geomean x{math.exp(value):.4f}' if value != float('inf')
                     else f'failed {n_failed}')
            print(f'[optuna] trial {trial.number} {outcome}: {score}  '
                  f'elapsed={elapsed:.0f}s  weights={weights}')
            return value

        n_complete = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = n_trials - n_complete
        print(f'[optuna] study has {n_complete} completed trials; target {n_trials}')
        print(f'[optuna] weights file:  {weights_path} (best trial so far)')
        print(f'[optuna] history file:  {history_path}')
        print(f'[optuna] output root:   {output_root}')
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
            _write_best(study, weights_path)
        env.__exit__(None, None, None)

    return study


def _write_best(study, weights_path, final=True):
    """The trained weights are the best trial that evaluated every assembly
    and failed none; pruned, deadline and failing trials never qualify.
    `final=False` (after each trial) writes quietly."""
    import optuna

    qualified = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.user_attrs.get('outcome') == 'complete'
                 and t.user_attrs.get('n_failed') == 0
                 and t.value is not None and math.isfinite(t.value)]
    if not qualified:
        if final:
            print('[optuna] WARN no trial completed every assembly; weights file left as-is')
        return
    best = min(qualified, key=lambda t: t.value)
    best_weights = best.user_attrs['entry']['weights']
    _write_json(weights_path, best_weights)
    if not final:
        return
    print(f'[optuna] best trial: {best.number}  geomean time ratio vs baseline: '
          f'x{math.exp(best.value):.4f}')
    print(f'[optuna] best weights: {best_weights}')
    print(f'[optuna] best weights written to {weights_path}')


# ----------------------------------------------------------------------------
# Evaluation on held-out assemblies
# ----------------------------------------------------------------------------

_SIZE_BANDS = ((5, 9), (10, 14), (15, 19), (20, 26), (27, None))

# |log ratio| below this is a tie: the pipeline is deterministic, so equal
# plans give equal times to the last digit.
_TIE_LOG_RATIO = 1e-9

# The planners compared on the test split: (label, planner, generator, what).
# 'reference' is the heuristic planner with the reference weights (its runs
# are the training baselines), 'trained' the same planner with the trained
# weights, 'heur-out' the gen:heur-out baseline of data_assembly_time.
_EVAL_RUNS = (
    ('reference', 'heuristic', 'rand', 'heuristic DFA planner, reference weights'),
    ('trained', 'heuristic', 'rand', 'heuristic DFA planner, trained weights'),
    ('heur-out', 'gen-adapter', 'heur-out', 'gen:heur-out baseline (heur-out generator)'),
)
_EVAL_COMPARISONS = (('trained', 'reference'), ('trained', 'heur-out'), ('heur-out', 'reference'))


def evaluate_heuristic_weights(test_eval, args, weights_path, output_root=None,
                               label='trained', time_budget_s=None):
    """Test the weights in `weights_path` on `test_eval.assemblies` (held out
    from training) against two baselines: the heuristic planner with the
    reference weights, and gen:heur-out. Every run is planned and arm-timed
    exactly like a training trial: no media, the candidate-check cache shared
    under `output_root`, one stored run per assembly guarded by a lock, so the
    workers of a job array split the set between them and a re-run only
    computes what is missing. Reference runs share `<output_root>/baselines/`
    with training (same fingerprint); heur-out runs, which do not depend on
    any weights, live in `<output_root>/heur-out/`. Writes
    `<output_root>/eval_<label>/summary.{json,txt}` and returns the summary.
    `time_budget_s`: no run starts, and no waiting continues, after it; the
    summary then lists what was left out.
    """
    deadline = None if time_budget_s is None else time.time() + float(time_budget_s)
    cfg = _training_config()
    output_root = Path(output_root or 'assets/optuna_training')
    with open(weights_path) as _f:
        loaded = json.load(_f)
    missing = [k for k in WEIGHT_KEYS if k not in loaded]
    if missing:
        raise ValueError(f'{weights_path} lacks weights {missing}')
    weights = {k: float(loaded[k]) for k in WEIGHT_KEYS}
    reference = _pin(_reference_weights(), cfg['fixed_weights'])
    eval_dir = output_root / f'eval_{label}'
    print(f'[eval] {len(test_eval.assemblies)} assemblies; reference {reference}; '
          f'{label} {weights} (from {weights_path})')

    run_dirs = {'reference': output_root / 'baselines', 'trained': eval_dir / 'runs',
                'heur-out': output_root / 'heur-out'}
    run_weights = {'reference': reference, 'trained': weights, 'heur-out': reference}
    runsets = {}
    with _pipeline_environment(args, output_root, cfg):
        for name, planner, generator, _what in _EVAL_RUNS:
            runsets[name] = _ensure_runs(
                test_eval.assemblies, args, run_dirs[name], run_weights[name], name,
                clear_sdf=(name == 'reference'), deadline=deadline,
                planner=planner, generator=generator)

    summary = _summarize_evaluation(test_eval.assemblies, runsets, reference, weights,
                                    str(weights_path))
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
    paired = [r for r in rows if r['runs'][a]['status'] == 'ok' and r['runs'][b]['status'] == 'ok']
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
    out['component_mean_s'] = {}
    if paired:
        for comp in ('step_disassembly_s', 'transitions_s', 'base_travel_s',
                     'reorientation_s', 'hold_s'):
            out['component_mean_s'][comp] = {
                side: sum((r['runs'][side]['components'] or {}).get(comp, 0.0) for r in paired) / len(paired)
                for side in (a, b)}
    return out


def _summarize_evaluation(ass_list, runsets, reference, weights, weights_path):
    rows = []
    for ass in ass_list:
        aid = str(ass.id)
        runs = {}
        for name, *_ in _EVAL_RUNS:
            rec = runsets[name].get(aid) or {}
            runs[name] = {'status': rec.get('status'), 'total_s': rec.get('total_s'),
                          'components': rec.get('components')}
        rows.append({'id': aid, 'n_parts': len(ass.objects), 'runs': runs})
    success = {}
    for name, *_ in _EVAL_RUNS:
        statuses = [r['runs'][name]['status'] for r in rows]
        success[name] = {
            'ok': sum(s == 'ok' for s in statuses),
            'excluded': sum(s == 'excluded' for s in statuses),
            'failed': [r['id'] for r, s in zip(rows, statuses) if s not in ('ok', 'excluded', None)],
            'not_run': [r['id'] for r, s in zip(rows, statuses) if s is None],
        }
    return {
        'runs': {name: what for name, _p, _g, what in _EVAL_RUNS},
        'weights_path': weights_path,
        'reference_weights': reference,
        'weights': weights,
        'success': success,
        'comparisons': {f'{a} vs {b}': _compare(rows, a, b) for a, b in _EVAL_COMPARISONS},
        'per_assembly': rows,
    }


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
        lines.append(f'  {name:<10} ok {s["ok"]:>3}   precheck-excluded {s["excluded"]:>3}   '
                     f'failed {s["failed"]}' + (f'   not run (time budget) {s["not_run"]}' if s['not_run'] else ''))
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
            '  mean component time (s): ' + ',  '.join(
                f'{comp[:-2]} {v[a]:.1f} vs {v[b]:.1f}' for comp, v in c['component_mean_s'].items()
                if v[a] or v[b]),
        ]
    head = f'{"id":<7}{"parts":>6}' + ''.join(f'{n + " s":>13}' for n in names)
    head += f'{"trained/ref":>13}{"trained/h-o":>13}'
    lines += ['', head]
    for r in summary['per_assembly']:
        cells = []
        for n in names:
            run = r['runs'][n]
            cells.append(f'{run["total_s"]:13.2f}' if run['status'] == 'ok'
                         else f'{(run["status"] or "not run")[:12]:>13}')
        t = r['runs']['trained']
        ratios = []
        for other in ('reference', 'heur-out'):
            o = r['runs'][other]
            ratios.append(f'{"x%.3f" % (t["total_s"] / o["total_s"]):>13}'
                          if t['status'] == 'ok' and o['status'] == 'ok' else f'{"-":>13}')
        lines.append(f'{r["id"]:<7}{r["n_parts"]:>6}' + ''.join(cells) + ''.join(ratios))
    return '\n'.join(lines) + '\n'
