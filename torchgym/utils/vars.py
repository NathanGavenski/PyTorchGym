import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2
