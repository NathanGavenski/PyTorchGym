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


env = gym.make("TorchPendulum-v1")
obs, _ = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, done, terminated, info = env.step(action)
```

We also support vectorized environments:
```python
import gymnasium as gym
import torchgym
from torchgym.wrappers import SyncVectorSimulator

env = SyncVectorSimulator([gym.make("TorchPendulum-v1"), gym.make("TorchPendulum-v1")])
obs, _ = env.reset(seed=[0, 1])
actions = env.action_space.sample()
obs, rewards, dones, terminateds, infos = env.step(actions)
```

## Simulator
This project is mainly for simulating transitions and maintaining them differentiable.
That allows us to pass actions that have gradient and receive rewards and next states in the same gradient graph.

```python
from imitation_datasets.dataset import BaselineDataset
import gymnasium as gym
import torchgym
from torchgym.wrappers import Simulator

# Just to get some data
dataset = BaselineDataset("NathanGavenski/Pendulum-v1", source="huggingface", n_episodes=1)
state, action, next_state = dataset[0]

env = gym.make("TorchPendulum-v1")
env = Simulator(env)
env.set_state(state) # Now the state will be the same as the one from the dataset

obs, reward, done, terminated, info = env.step(action)
assert (obs == next_state).all()  # will pass since the transition will be the same
```

