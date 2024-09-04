from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Optional

from gymnasium import Space
from gymnasium.core import Env
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.utils import concatenate, create_empty_array, batch_space
import torch
from torch import Tensor
from torch import device as Device
import numpy as np


class SyncVectorSimulator(SyncVectorEnv):
    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: None | Space = None,
        action_space: None | Space = None,
        copy: bool = False,
        device: Device = "cpu",
    ):
        SyncVectorEnv.__init__(
            self,
            env_fns,
            observation_space=observation_space,
            action_space=action_space,
            copy=copy,
        )
        self.device = device

        self.observation_space = batch_space(self.envs[0].observation_space, len(self.envs))
        self.action_space = batch_space(self.envs[0].action_space, len(self.envs))

        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs,
        ).to(self.device)
        self._rewards = torch.zeros((self.num_envs,), dtype=torch.float32)
        self._terminateds = torch.zeros((self.num_envs,), dtype=torch.bool)
        self.truncateds = torch.zeros((self.num_envs,), dtype=torch.bool)
        self._actions = None

    def reset_wait(
        self,
        seed: Optional[int | list[int]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False

        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos
    
    def step_wait(self) -> tuple[Any, Tensor[Any], Tensor[Any], Tensor[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            self._rewards,
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def set_state(self, states: Tensor) -> tuple[Tensor[Any], dict[str, Any]]:
        assert states.size(0) == len(self.envs), \
            "states size don't match the number of environments"

        observations, infos = [], {}
        for i, (env, state) in enumerate(zip(self.envs, states)):
            (observation, info) = env.set_state(state)

            observations.append(observation)
            info = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            infos,
        )
