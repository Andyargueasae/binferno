from pydantic import Field
import torch

from .barlowtwins import BarlowTwinsTraining
from .simclr import SimCLRTraining
from .base import BaseTraining


class BinfernoTraining(BaseTraining):
    use_simsclr:bool = Field(default=True, description="Whether to use SimCLR for training.")
    simclr:SimCLRTraining
    use_barlowtwins:bool = Field(default=True, description="Whether to use Barlow Twins for training.")
    barlowtwins:BarlowTwinsTraining

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        results = []

        if self.use_simsclr:
            results.append(self.simclr(*args, **kwargs))

        if self.use_barlowtwins:
            results.append(self.barlowtwins(*args, **kwargs))

        return torch.cat(results, dim=1)        