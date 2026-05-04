import torch
from pydantic import Field

from .base import BaseClustering
from .leiden import LeidenClustering


class BinfernoClustering(BaseClustering):
    use_leiden:bool = Field(True, description="Whether or not to use Leiden clustering") # This should be an enum of clustering methods, but for now we only have one.
    leiden: LeidenClustering

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self.use_leiden:
            return self.leiden(*args, **kwargs)

        raise NotImplementedError()