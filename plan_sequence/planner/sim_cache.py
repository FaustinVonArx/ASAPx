"""Persistent cache of the DFA planner's per-candidate physics results.

One candidate check (`_simulate_standalone`: assemblability path, DoF probe,
stability) is a function of the assembly geometry, the moving part, the parts
still present, the pose and a few planner settings. The heuristic weights
never enter it; they only decide which candidates get checked. Plans of the
same assembly under different weights (training trials) or different planners
(the data_assembly_time RUNS) therefore re-check largely the same candidates,
and those checks are ~97% of planning wall time. The cache stores each result
once and replays it.

Layout::

    <root>/<geometry hash>/<config hash>/shard_<host>_<pid>.pkl
    <root>/<geometry hash>/assembly.txt    (last assembly dir seen, for reading)

Each process appends to its own shard, a stream of pickled (key, sim_info)
records, so concurrent writers (parallel training workers on a shared
filesystem) never share a file. Every shard is read when a plan starts; what
another process adds mid-plan is picked up by the next plan. A torn record at
the end of a shard (a writer that crashed mid-write) ends that shard's read.

The config hash covers the physics module constants (plan_sequence.sim_string,
plan_sequence.physics_planner), the source of the simulation code and
settings.filter_below_ground, so changing any of them starts a fresh cache
rather than replaying stale results. Other code changes that alter physics
outcomes are not detected: delete the cache root after such an edit.

Replaying is exact only for checks that do not depend on timing. The planner
therefore leaves the cache off with a planning timeout, a per-parent success
quota, tools or rendering (see DFASequencePlanner.plan). The physics still
stops some probes on wall-clock limits (physics_planner.MAX_TIME,
DOF_MAX_TIME); a cached result replays what the first computation saw, which
keeps later plans consistent with each other.
"""
import copy
import hashlib
import inspect
import os
import pickle
import socket
from pathlib import Path

import numpy as np

# Bump when the stored record layout or the key fields change.
CACHE_VERSION = 1

# Physics code whose source is part of the config hash.
_PHYSICS_MODULES = (
    'plan_sequence.physics_planner',
    'plan_sequence.sim_string',
    'plan_sequence.feasibility_check',
    'plan_sequence.stable_pose',
)


def _geometry_hash(assembly_dir):
    h = hashlib.sha1()
    for name in sorted(os.listdir(assembly_dir)):
        if not name.endswith('.obj'):
            continue
        h.update(name.encode())
        with open(os.path.join(assembly_dir, name), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def _module_constants(module):
    return sorted(
        (k, v) for k, v in vars(module).items()
        if k.isupper() and isinstance(v, (bool, int, float, str))
    )


def _config_hash(extra):
    import importlib

    import settings
    from plan_sequence.planner.base import _simulate_standalone

    h = hashlib.sha1()
    h.update(repr(CACHE_VERSION).encode())
    for name in _PHYSICS_MODULES:
        module = importlib.import_module(name)
        h.update(repr(_module_constants(module)).encode())
        with open(module.__file__, 'rb') as f:
            h.update(f.read())
    h.update(inspect.getsource(_simulate_standalone).encode())
    h.update(repr(bool(getattr(settings, 'filter_below_ground', False))).encode())
    h.update(repr(extra).encode())
    return h.hexdigest()


class SimCache:
    """Cache for one plan of one assembly. `key()` identifies a record by its
    per-task inputs; `fields` are the inputs fixed for the whole plan (the
    planner configuration, or what kind of record it is, e.g. the initial
    precheck) and go into the directory's config hash, so records made under
    different ones never mix. Values must be hashable by repr, e.g. an
    ignore list as a sorted tuple."""

    def __init__(self, root, assembly_dir, **fields):
        assembly_dir = os.path.abspath(assembly_dir)
        planner_fields = tuple(sorted((k, repr(v)) for k, v in fields.items()))
        # Keyed by geometry, not by directory: data_assembly_time plans the same
        # subassembly out of a different temp dir in each RUN.
        geometry_dir = Path(root) / _geometry_hash(assembly_dir)[:16]
        self.dir = geometry_dir / _config_hash(planner_fields)[:16]
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            (geometry_dir / 'assembly.txt').write_text(assembly_dir + '\n')
        except OSError:
            pass
        self._entries = {}
        for shard in sorted(self.dir.glob('shard_*.pkl')):
            self._load_shard(shard)
        self._n_loaded = len(self._entries)
        self._shard = self.dir / f'shard_{socket.gethostname()}_{os.getpid()}.pkl'
        self._fh = None
        self.hits = 0
        self.misses = 0

    def _load_shard(self, path):
        try:
            with open(path, 'rb') as f:
                while True:
                    try:
                        key, sim_info = pickle.load(f)
                    except EOFError:
                        return
                    self._entries[key] = sim_info
        except (OSError, pickle.UnpicklingError, ValueError, TypeError, AttributeError):
            # Torn tail or unreadable shard: keep what was read before it.
            return

    @staticmethod
    def key(part_move, parts_rest, pose, extra=None):
        """`extra`: inputs that differ between records of one plan besides
        the part, the remaining parts and the pose (e.g. the leaf expansion's
        gripper budget)."""
        pose_key = (None if pose is None
                    else np.ascontiguousarray(pose, dtype=np.float64).tobytes())
        fields = (part_move, tuple(parts_rest), pose_key)
        if extra is not None:
            fields += (extra,)
        return hashlib.sha1(pickle.dumps(fields, protocol=4)).hexdigest()

    def get(self, key):
        sim_info = self._entries.get(key)
        if sim_info is None:
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(sim_info)

    def put(self, key, sim_info):
        # Drop the per-call timings (`_dt_*`) and tags: a replayed result has
        # not cost any simulation time.
        record = {k: v for k, v in sim_info.items() if not k.startswith('_')}
        self._entries[key] = copy.deepcopy(record)
        if self._fh is None:
            self._fh = open(self._shard, 'ab')
        pickle.dump((key, record), self._fh, protocol=4)
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def summary(self):
        return {
            'dir': str(self.dir),
            'entries_loaded': self._n_loaded,
            'hits': self.hits,
            'misses': self.misses,
        }
