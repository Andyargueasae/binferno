import torch

from .barlowtwins import BarlowTwinsTraining
from .simclr import SimCLRTraining
from .base import BaseTraining


class BinfernoTraining(BaseTraining):
    simclr:SimCLRTraining
    barlowtwins:BarlowTwinsTraining

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        results = []

        if self.simclr.use:
            results.append(self.simclr(*args, **kwargs))

        if self.barlowtwins.use:
            results.append(self.barlowtwins(*args, **kwargs))

        return torch.cat(results, dim=1)        