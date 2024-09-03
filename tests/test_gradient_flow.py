import unittest
import torch
import torchgym.envs.pendulum as torch_pendulum

class TestGradientFlow(unittest.TestCase):
    """Class for testing gradient flow through the environment."""

    def setUp(self):
        """Set up the environment for testing."""
        self.env = torch_pendulum.PendulumEnv()

    def test_gradient_flow(self):
        action = torch.tensor(self.env.action_space.sample(), dtype=torch.float32, requires_grad=True)

        self.env.reset(seed=42)

        torch_obs, torch_reward, *_ = self.env.step(action)

        loss = torch_reward.sum()

        loss.backward()

        self.assertTrue(action.grad is not None, "Gradients are not flowing through the action tensor.")
        self.assertTrue(torch_obs.grad is None, "Gradients are flowing through the observation tensor.")
        self.assertTrue(torch_reward.grad is None, "Gradients are flowing through the reward tensor.")
        self.assertTrue(torch.any(action.grad != 0), "Gradients are zero; something is wrong with the gradient flow.")
        self.assertTrue(torch.all(torch.isfinite(action.grad)), "Gradients are not finite; something is wrong with the gradient flow.")

    def tearDown(self):
        """Close the environment after testing."""
        self.env.close()
