import torch

from .base import BaseClustering
from .leiden import LeidenClustering


class BinfernoClustering(BaseClustering):
    leiden: LeidenClustering

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self.leiden.use:
            return self.leiden(*args, **kwargs)

        raise NotImplementedError()