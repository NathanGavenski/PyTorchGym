"""Register environments for gym."""
from gymnasium.envs.registration import register


register(
    id="TorchPendulum-v1",
    entry_point="torchgym.envs.pendulum:PendulumEnv",
    max_episode_steps=200,
)
