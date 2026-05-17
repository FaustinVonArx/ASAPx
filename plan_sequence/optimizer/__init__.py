from .base import BaseSequenceOptimizer
from .divide import DivideOptimizer


optimizers = {
    'base': BaseSequenceOptimizer,
    'divide': DivideOptimizer,
}
