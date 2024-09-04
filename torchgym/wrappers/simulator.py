from __future__ import annotations

from typing import Any, SupportsFloat, Union

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType
from gymnasium.utils import RecordConstructorArgs


class Simulator(Wrapper[ObsType, ActType, ObsType, ActType], RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def set_state(
        self,
        state: ObsType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        try:
            return self.env.unwrapped.set_state(state)
        except AttributeError as e:
            gym.logger.warn(f"An exception occured ({e}) while setting a state")
            return None
