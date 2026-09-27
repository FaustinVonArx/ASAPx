from .random import RandomGenerator
from .heuristic import HeuristicsVolumeGenerator, HeuristicsOutsidenessGenerator
from .dfa import DFAGenerator


class LazyLearningBasedGenerator:
    """Stands in for LearningBasedGenerator until the `learn` generator is used.

    learning.py imports torch, torch_geometric and the GNN modules under
    generator/network/, none of which any other generator needs. Importing it
    from this registry meant every run paid for torch, because run_seq_plan
    imports `generators` unconditionally. Keeping it out means a rand /
    heur-vol / heur-out / dfa run never loads torch at all.

    That matters beyond startup time: loading torch's native library into a
    process that already holds redmax_py's segfaults inside the Apptainer
    container on the cluster, with no Python-level error. The two libraries
    import fine in either order on their own, so the clash needs the rest of
    the stack too (coacd, python-fcl, matplotlib) and is not understood; not
    importing torch for runs that do not need it sidesteps it entirely. A
    `--generator learn` run still loads torch and can still hit it.

    The registry keeps its five keys, so `list(generators.keys())` (the
    --generator argparse choices) is unchanged, and every consumer calls the
    value as a constructor, which is all this supports. Constructing it returns
    a real LearningBasedGenerator: __new__ returns an object that is not an
    instance of this class, so Python skips __init__ and the real constructor
    inside has already run.
    """

    def __new__(cls, *args, **kwargs):
        from .learning import LearningBasedGenerator

        return LearningBasedGenerator(*args, **kwargs)


generators = {
    'rand': RandomGenerator,
    'heur-vol': HeuristicsVolumeGenerator,
    'heur-out': HeuristicsOutsidenessGenerator,
    'learn': LazyLearningBasedGenerator,
    'dfa': DFAGenerator,
}
