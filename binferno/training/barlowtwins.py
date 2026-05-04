import torch
from .base import BaseTraining


class BarlowTwinsTraining(BaseTraining):
    def __call__(self, features:torch.Tensor) -> torch.Tensor:
        pass