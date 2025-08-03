import torch
import torch.nn as nn

class RandomGaussianNoise3D(nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise