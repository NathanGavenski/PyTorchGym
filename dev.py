import torch
import gymnasium as gym
import torchgym


if __name__ == "__main__":
    env = gym.make("TorchPendulum-v1")
    print(env.reset(seed=0))
    action = torch.tensor([1.6685], requires_grad=True)
    result = env.step(action)
    obs, cost, done, trunc, info = result
    loss = obs.sum()
    loss.backward()

