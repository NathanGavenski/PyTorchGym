# PyTorchGym

## Install

From source:

```bash
git clone https://github.com/NathanGavenski/PyTorchGym
cd PyTorchGym
pip install .
```

From git:
```bash
pip install git+https://github.com/NathanGavenski/PyTorchGym
```

## To use

```python
import gymnasium as gym
import torchgym


env = gym.create("TorchPendulum-v1")
obs, _ = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, done, terminated, info = env.step(action)
```