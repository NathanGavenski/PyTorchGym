from gymnasium.envs.registration import make, pprint_registry, register, registry, spec


register(
    id="TorchPendulum-v1",
    entry_point="torchgym.envs.pendulum:PendulumEnv",
    max_episode_steps=200,
)
