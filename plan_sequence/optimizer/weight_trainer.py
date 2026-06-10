"""Optuna-based black-box optimisation of HeuristicDFASequencePlanner
weights, with total arm-pipeline assembly time as the objective.

Storage layout (paths relative to repo root, configurable via settings):
- ``assets/heuristic_weights_optuna.json`` — current/best weights. Written
  per-trial during a study (so the planner reads the trial's weights live)
  and overwritten with the best trial's weights at study end. This is the
  same file `HeuristicDFASequencePlanner._load_weights` reads when
  ``settings.heuristic_weights_source == 'optuna'``.
- ``assets/heuristic_weights_optuna_history.json`` — per-trial log
  ``[{trial, weights, mean_total_s, per_assembly_total_s}, ...]`` saved
  after every trial so an aborted study still has usable data.
- ``assets/heuristic_weights_optuna_study.db`` (when ``persist_study=True``)
  — Optuna SQLite store so studies can be resumed.

Training / inference switch:
- Training: ``python main.py train_heuristic_weights --id <range>``. The
  trainer forces ``settings.heuristic_weights_source = 'optuna'`` for the
  duration of the study (so each trial's freshly-written weights are read
  back by the planner) and restores it on exit.
- Inference: set ``settings.heuristic_weights_source = 'optuna'`` and run
  normal commands — the planner just reads the file; nothing writes, so the
  weights are frozen.
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


WEIGHT_KEYS = ('contact_distance', 'free_dof', 'z_alignment', 'pose_change',
               'hold_count')

# Search bounds per weight. All non-negative (cost is minimised, negatives
# would flip the sign of a feature → nonsensical preferences). Upper bounds
# loosely follow the current DEFAULT_WEIGHTS magnitudes.
DEFAULT_SEARCH_SPACE = {
    'contact_distance': (0.0, 5.0),
    'free_dof':         (0.0, 5.0),
    'z_alignment':      (0.0, 5.0),
    'pose_change':      (0.0, 5.0),
    'hold_count':       (0.0, 10.0),
}


def _write_json(path, data, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def _evaluate_weights_on_assemblies(weights, ass_list, args,
                                    output_root, trial_label,
                                    weights_path):
    """Write `weights` to disk, then run the full pipeline (plan + render +
    arm) on every assembly in `ass_list`. Returns
    (mean_total_s, per_assembly_dict). Failed assemblies contribute None,
    not the mean — so a partial batch still gives a meaningful objective.
    """
    # Write weights atomically: write to tmp file, rename. The planner
    # reads this file at plan time when source='optuna'.
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = weights_path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(weights, f, indent=2)
    os.replace(tmp, weights_path)

    per_ass = {}
    totals = []

    # Force the per-trial planner config. Each trial fully re-plans (no
    # cache reuse) so the new weights actually take effect.
    args.planner = 'heuristic'
    args.generator = 'rand'
    args.plan_arm = True
    _saved_cache = args.cache
    args.cache = 'new'

    try:
        for ass in ass_list:
            _saved_storage = ass.storage_dir
            trial_storage = Path(output_root) / trial_label / str(ass.id)
            trial_storage.mkdir(parents=True, exist_ok=True)
            try:
                ass.storage_dir = trial_storage
                # Wipe any stale render artifacts from previous trials.
                for stale_dir in ('log', 'paths'):
                    p = trial_storage / stale_dir
                    if p.exists():
                        shutil.rmtree(str(p))
                for stale_gif in trial_storage.glob('*.gif'):
                    try:
                        stale_gif.unlink()
                    except OSError:
                        pass

                try:
                    import settings  # noqa: F401 (force fresh re-read)
                    _orig_render = getattr(settings, 'render_sequence', True)
                    settings.render_sequence = True  # arm pipeline reads after rendering
                except ImportError:
                    _orig_render = True

                ass.planner.get_assembly_plans(args)

                timing_path = trial_storage / 'log' / 'timing_overview.json'
                if not timing_path.exists():
                    per_ass[str(ass.id)] = None
                    print(f'[optuna] {trial_label} / {ass.id}: no timing_overview.json')
                    continue
                with open(timing_path) as _f:
                    timing = json.load(_f)
                total = float((timing.get('totals') or {}).get('total_s', 0.0))
                per_ass[str(ass.id)] = total
                totals.append(total)
                print(f'[optuna] {trial_label} / {ass.id}: total={total:.2f}s')
            except Exception as _e:
                traceback.print_exc()
                per_ass[str(ass.id)] = None
                print(f'[optuna] {trial_label} / {ass.id}: pipeline failed: {_e}')
            finally:
                ass.storage_dir = _saved_storage
                try:
                    settings.render_sequence = _orig_render
                except Exception:
                    pass
    finally:
        args.cache = _saved_cache

    mean_total = (sum(totals) / len(totals)) if totals else float('inf')
    return mean_total, per_ass


def train_heuristic_weights(test_eval, args,
                            n_trials=50, seed=42,
                            search_space=None,
                            output_root=None,
                            weights_path=None,
                            history_path=None,
                            persist_study=False):
    """Run an Optuna study to minimise mean assembly total_s across
    `test_eval.assemblies`. See module docstring for storage layout.

    Returns the completed `optuna.Study` so the caller can introspect.
    """
    try:
        import optuna
    except ImportError as _e:
        raise ImportError(
            "Optuna is required for train_heuristic_weights but is not installed. "
            "pip install optuna"
        ) from _e

    search_space = dict(search_space or DEFAULT_SEARCH_SPACE)
    weights_path = Path(weights_path or 'assets/heuristic_weights_optuna.json')
    history_path = Path(history_path or 'assets/heuristic_weights_optuna_history.json')
    output_root = Path(output_root or 'assets/optuna_training')
    output_root.mkdir(parents=True, exist_ok=True)

    # Force the planner to read trial weights from disk for the duration of
    # the study. Restored on exit.
    import settings
    _orig_source = getattr(settings, 'heuristic_weights_source', 'default')
    _orig_optuna_path = getattr(settings, 'heuristic_weights_optuna_path', None)
    settings.heuristic_weights_source = 'optuna'
    settings.heuristic_weights_optuna_path = str(weights_path)

    history = []
    # Resume incremental history if it exists (useful when extending an
    # earlier aborted study).
    if history_path.exists():
        try:
            with open(history_path) as _f:
                history = json.load(_f) or []
        except (OSError, json.JSONDecodeError):
            history = []

    sampler = optuna.samplers.TPESampler(seed=seed)
    storage = None
    if persist_study:
        storage = f'sqlite:///{output_root / "study.db"}'

    study = optuna.create_study(
        study_name='heuristic_weights',
        direction='minimize',
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(storage),
    )

    n_done_initial = len(study.trials)
    print(f'[optuna] starting study with {n_trials} trials  '
          f'(resuming from {n_done_initial} prior trials)' if n_done_initial
          else f'[optuna] starting fresh study with {n_trials} trials')
    print(f'[optuna] weights file:  {weights_path}')
    print(f'[optuna] history file:  {history_path}')
    print(f'[optuna] output root:   {output_root}')

    def objective(trial):
        weights = {
            k: trial.suggest_float(k, *search_space[k])
            for k in WEIGHT_KEYS
        }
        t0 = time.time()
        mean_total, per_ass = _evaluate_weights_on_assemblies(
            weights, test_eval.assemblies, args,
            output_root=output_root,
            trial_label=f'trial_{trial.number:04d}',
            weights_path=weights_path,
        )
        elapsed = time.time() - t0

        entry = {
            'trial': trial.number,
            'weights': weights,
            'mean_total_s': None if mean_total == float('inf') else mean_total,
            'per_assembly_total_s': per_ass,
            'elapsed_s': elapsed,
        }
        history.append(entry)
        try:
            _write_json(history_path, history)
        except Exception as _e:
            print(f'[optuna] WARN incremental history write failed: {_e}')

        print(f'[optuna] trial {trial.number}: mean_total={mean_total:.2f}s  '
              f'elapsed={elapsed:.1f}s  weights={weights}')
        return mean_total

    try:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print(f'[optuna] interrupted after {len(study.trials)} trials; '
              f'writing best-so-far before exit')
    finally:
        # Write best trial's weights as the FINAL state of the weights file
        # (so inference picks them up cleanly).
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None and t.value != float('inf')]
        if completed:
            best = min(completed, key=lambda t: t.value)
            best_weights = {k: float(best.params[k]) for k in WEIGHT_KEYS}
            with open(weights_path, 'w') as _f:
                json.dump(best_weights, _f, indent=2)
            print(f'[optuna] best trial: {best.number}  '
                  f'best mean total_s: {best.value:.2f}')
            print(f'[optuna] best weights: {best_weights}')
            print(f'[optuna] best weights written to {weights_path}')
        else:
            print('[optuna] WARN no completed trials; weights file left as-is')

        # Restore the original source so an interactive session doesn't
        # silently switch to optuna inference after training.
        settings.heuristic_weights_source = _orig_source
        if _orig_optuna_path is not None:
            settings.heuristic_weights_optuna_path = _orig_optuna_path

    return study
