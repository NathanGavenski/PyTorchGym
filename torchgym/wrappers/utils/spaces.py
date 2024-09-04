from copy import deepcopy

import numpy as np
from torchgym.spaces import Box
from gymnasium.vector.utils import batch_space


@batch_space.register(Box)
def _batch_space_box(space, n=1):
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))