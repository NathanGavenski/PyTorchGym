import unittest
import torch
import gymnasium as gym
import numpy as np

import torchgym.envs.pendulum as torch_pendulum


class TestPendulumEnv(unittest.TestCase):
    """Tests for the Pendulum environment."""

    def setUp(self):
        """Set up the environments for testing."""
        self.gym_env = gym.make('Pendulum-v1', render_mode='rgb_array')
        self.torch_env = torch_pendulum.PendulumEnv()

        self.gym_env.reset(seed=0)
        self.torch_env.reset(seed=0)

    def test_reset(self):
        """Test the reset method of the environment."""
        gym_obs, _ = self.gym_env.reset()
        torch_obs, _ = self.torch_env.reset()

        np.testing.assert_allclose(
            gym_obs, torch_obs, rtol=1e-5, atol=1e-5,
            err_msg="The initial observations after reset do not match."
        )
        self.assertTrue(isinstance(torch_obs, torch.Tensor))

    def test_step(self):
        """Test the step method of the environment."""
        action = self.gym_env.action_space.sample()
        torch_action = torch.from_numpy(action)

        gym_obs, gym_reward, gym_done, \
            gym_terminated, gym_info = self.gym_env.step(action)
        torch_obs, torch_reward, torch_done, \
            torch_terminated, torch_info = self.torch_env.step(torch_action)

        np.testing.assert_allclose(
            gym_obs, torch_obs, rtol=1e-5, atol=1e-5,
            err_msg="Observations do not match after step."
        )
        np.testing.assert_allclose(
            gym_reward, torch_reward, rtol=1e-5, atol=1e-5,
            err_msg="Rewards do not match after step."
        )

        self.assertEqual(gym_terminated, torch_terminated, "Terminated flags do not match after step.")
        self.assertEqual(gym_done, torch_done, "Done flags do not match after step.")
        self.assertEqual(gym_info, torch_info, "Info dictionaries do not match after step.")

        self.assertTrue(isinstance(torch_obs, torch.Tensor))
        self.assertTrue(isinstance(torch_reward, torch.Tensor))

    def test_episode(self):
        """Test the environment over an entire episode."""
        self.gym_env.reset()
        self.torch_env.reset()

        done = False
        steps = 0
        while not done:
            action = self.gym_env.action_space.sample()
            torch_action = torch.from_numpy(action)

            gym_obs, gym_reward, gym_done, \
                gym_terminated, gym_info = self.gym_env.step(action)
            torch_obs, torch_reward, torch_done, \
                torch_terminated, torch_info = self.torch_env.step(torch_action)

            np.testing.assert_allclose(
                gym_obs, torch_obs, rtol=1e-5, atol=1e-5,
                err_msg=f"Observations do not match during the episode (steps: {steps})."
            )
            np.testing.assert_allclose(
                gym_reward, torch_reward, rtol=1e-5, atol=1e-5,
                err_msg="Rewards do not match during the episode."
            )
            self.assertEqual(gym_done, torch_done, "Done flags do not match during the episode.")
            self.assertEqual(gym_info, torch_info, "Info dictionaries do not match during the episode.")

            done = gym_done or gym_terminated
            steps += 1

    def tearDown(self):
        """Close the environments after testing."""
        self.gym_env.close()
        self.torch_env.close()
