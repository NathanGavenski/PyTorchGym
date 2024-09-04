from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Space
from gymnasium.vector.utils import concatenate, create_empty_array
import torch


@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(
    space: Space,
    items: tuple[torch.Tensor],
    out: torch.Tensor
) -> torch.Tensor:
    return torch.stack(items, dim=0)

@create_empty_array.register(Box)
@create_empty_array.register(Discrete)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_base(
    space: Space,
    n: int = 1,
    fn=torch.zeros
) -> torch.Tensor:
    shape = space.shape if (n is None) else (n,) + space.shape
    return fn(shape)