"""Provides some variables used in the environments."""
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2
