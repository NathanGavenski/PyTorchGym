from copy import deepcopy

import unittest
from unittest.mock import MagicMock
import torch
import numpy as np

from torchgym.envs import PendulumEnv
from torchgym.spaces import Box
from torchgym.wrappers import SyncVectorSimulator


class TestSyncVectorSimulator(unittest.TestCase):

    def setUp(self):
        observation_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.mock_env = MagicMock(spec=PendulumEnv)
        self.mock_env.observation_space = observation_space
        self.mock_env.action_space = action_space
        self.mock_env.reset.return_value = (torch.tensor([0.0, 0.0, 0.0]), {"info": "reset"})
        self.mock_env.step.return_value = (
            torch.tensor([1.0, 1.0, 1.0]), 1.0, False, False, {"info": "step"}
        )

        self.env_fns = [lambda: deepcopy(self.mock_env) for _ in range(3)]
        self.simulator = SyncVectorSimulator(self.env_fns, device="cpu")

    def test_initialization(self):
        self.assertEqual(self.simulator.device, "cpu")
        self.assertIsInstance(self.simulator.observations, torch.Tensor)
        self.assertEqual(self.simulator.observations.shape[0], self.simulator.num_envs)
        self.assertTrue((self.simulator._rewards == 0).all())
        self.assertTrue((self.simulator._terminateds == False).all())
        self.assertTrue((self.simulator.truncateds == False).all())

    def test_reset_wait(self):
        seed = [42, 43, 44]
        options = {"option": "value"}

        observations, infos = self.simulator.reset_wait(seed=seed, options=options)

        for single_seed, env in zip(seed, self.simulator.envs):
            env.reset.assert_called_with(seed=single_seed, options=options)

        self.assertIsInstance(observations, torch.Tensor)
        self.assertEqual(len(infos["_info"]), self.simulator.num_envs)

    def test_reset_wait_no_seed(self):
        observations, infos = self.simulator.reset_wait()

        for env in self.simulator.envs:
            env.reset.assert_called()

        self.assertIsInstance(observations, torch.Tensor)
        self.assertEqual(len(infos["_info"]), self.simulator.num_envs)

    def test_step_wait(self):
        self.simulator._actions = [np.array([0.0]), np.array([1.0]), np.array([2.0])]

        observations, rewards, terminateds, truncateds, infos = self.simulator.step_wait()

        for i, env in enumerate(self.simulator.envs):
            env.step.assert_called_with(self.simulator._actions[i])

        self.assertIsInstance(observations, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(terminateds, np.ndarray)
        self.assertIsInstance(truncateds, np.ndarray)
        self.assertEqual(len(infos["_info"]), self.simulator.num_envs)

        self.assertTrue((rewards == 1.0).all())
        self.assertTrue((terminateds == False).all())
        self.assertTrue((truncateds == False).all())

    def test_step_wait_with_terminal_state(self):
        self.simulator._actions = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
        side_effect = [
            (torch.tensor([1.0, 1.0, 1.0]), 1.0, True, False, {"info": "step"}),
            (torch.tensor([1.0, 1.0, 1.0]), 1.0, False, True, {"info": "step"}),
            (torch.tensor([1.0, 1.0, 1.0]), 1.0, False, False, {"info": "step"}),
        ]
        for i, env in enumerate(self.simulator.envs):
            env.step.side_effect = [side_effect[i]]

        observations, rewards, terminateds, truncateds, infos = self.simulator.step_wait()

        self.assertEqual(self.simulator.envs[0].reset.call_count, 1)
        self.assertEqual(self.simulator.envs[1].reset.call_count, 1)

        self.assertIsInstance(observations, torch.Tensor)
        self.assertTrue((rewards == 1.0).all())
        self.assertEqual(terminateds[0], True)
        self.assertEqual(truncateds[1], True)

    def test_set_state(self):
        states = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])

        for i, env in enumerate(self.simulator.envs):
            env.set_state.return_value = (torch.tensor([0.0, 0.0, 0.0]), {"info": f"set_state_{i}"})

        observations, infos = self.simulator.set_state(states)

        for i, env in enumerate(self.simulator.envs):
            # Have to do like this because of pytorch ambiguous comparison
            called_state = env.set_state.call_args[0][0]
            self.assertTrue(
                torch.equal(called_state, states[i]), f"State for env {i} does not match."
            )

        self.assertIsInstance(observations, torch.Tensor)
        self.assertEqual(observations.shape[0], self.simulator.num_envs)
        self.assertEqual(len(infos["_info"]), self.simulator.num_envs)

    def test_set_state_incorrect_state_size(self):
        incorrect_states = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]  # Extra state
        ])

        with self.assertRaises(AssertionError):
            self.simulator.set_state(incorrect_states)

    def test_set_state_handles_info_merging(self):
        states = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])

        for i, env in enumerate(self.simulator.envs):
            env.set_state.return_value = (torch.tensor([i, i, i]), {"extra_info": f"extra_{i}"})

        observations, infos = self.simulator.set_state(states)

        expected_observations = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.assertTrue(torch.equal(observations, expected_observations))

    def test_set_state_with_copy(self):
        states = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])

        self.simulator.copy = True

        for env in self.simulator.envs:
            env.set_state.return_value = (torch.tensor([0.0, 0.0, 0.0]), {"info": "set_state"})

        observations, infos = self.simulator.set_state(states)

        self.assertIsInstance(observations, torch.Tensor)
        self.assertIsNot(observations, self.simulator.observations)
        self.assertTrue(torch.equal(observations, self.simulator.observations))
