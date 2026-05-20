from .dfs import DFSSequencePlanner
from .beam import BeamSearchSequencePlanner
from .random import RandomSequencePlanner
from .dfa import DFASequencePlanner
from .heuristic import HeuristicDFASequencePlanner


planners = {
    'dfs': DFSSequencePlanner,
    'beam': BeamSearchSequencePlanner,
    'randseq': RandomSequencePlanner,
    'dfa': DFASequencePlanner,
    'heuristic': HeuristicDFASequencePlanner,
}
