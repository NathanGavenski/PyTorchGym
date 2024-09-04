import unittest
from unittest.mock import MagicMock, patch
import gymnasium as gym
from gymnasium.core import ActType, ObsType

from torchgym.wrappers import Simulator

class TestSimulator(unittest.TestCase):

    def setUp(self):
        # Create a mock environment
        self.mock_env = MagicMock(spec=gym.Env)
        self.simulator = Simulator(self.mock_env)

    def test_step(self):
        # Setup mock return value
        obs = MagicMock(spec=ObsType)
        reward = 1.0
        done = False
        truncated = False
        info = {}
        self.mock_env.step.return_value = (obs, reward, done, truncated, info)

        # Call the step method
        action = MagicMock(spec=ActType)
        result = self.simulator.step(action)

        # Verify the environment's step method was called with the correct action
        self.mock_env.step.assert_called_once_with(action)

        # Verify the result
        self.assertEqual(result, (obs, reward, done, truncated, info))

    def test_reset(self):
        # Setup mock return value
        obs = MagicMock(spec=ObsType)
        info = {}
        self.mock_env.reset.return_value = (obs, info)

        # Call the reset method
        seed = 123
        options = {'option': 'value'}
        result = self.simulator.reset(seed=seed, options=options)

        # Verify the environment's reset method was called with the correct parameters
        self.mock_env.reset.assert_called_once_with(seed=seed, options=options)

        # Verify the result
        self.assertEqual(result, (obs, info))

    def test_set_state_success(self):
        # Setup mock return value for set_state
        obs = MagicMock(spec=ObsType)
        reward = 1.0
        done = False
        truncated = False
        info = {}
        self.mock_env.unwrapped.set_state = MagicMock(return_value=(obs, reward, done, truncated, info))

        # Call the set_state method
        result = self.simulator.set_state(obs)

        # Verify the environment's set_state method was called with the correct state
        self.mock_env.unwrapped.set_state.assert_called_once_with(obs)

        # Verify the result
        self.assertEqual(result, (obs, reward, done, truncated, info))

    @patch('gymnasium.logger')
    def test_set_state_attribute_error(self, mock_logger):
        # Mock an AttributeError
        self.mock_env.unwrapped.set_state = MagicMock(side_effect=AttributeError('set_state not implemented'))

        # Call the set_state method
        result = self.simulator.set_state(MagicMock())

        # Verify the logger warning was called
        mock_logger.warn.assert_called_once_with("An exception occured (set_state not implemented) while setting a state")

        # Verify the result
        self.assertIsNone(result)
