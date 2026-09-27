import random
import sys

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # Seed torch only when something has already imported it, i.e. the `learn`
    # generator. plan_robot.run_grasp_plan imports this module, and
    # plan_sequence.planner.base imports that, so importing torch here would
    # put it back into every run and undo the lazy import in
    # plan_sequence.generator. A run with no torch loaded has no torch RNG to
    # seed, so skipping it is a no-op rather than a gap.
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
