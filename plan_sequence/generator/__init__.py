from .random import RandomGenerator
from .heuristic import HeuristicsVolumeGenerator, HeuristicsOutsidenessGenerator
from .learning import LearningBasedGenerator
from .dfa import DFAGenerator


generators = {
    'rand': RandomGenerator,
    'heur-vol': HeuristicsVolumeGenerator,
    'heur-out': HeuristicsOutsidenessGenerator,
    'learn': LearningBasedGenerator,
    'dfa': DFAGenerator,
}
