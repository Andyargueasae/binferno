import torch
from .base import BaseFeatures


class CorgiFeatures(BaseFeatures):
    def __call__(
        self,
        seq:str,
        coverage:int,
    ) -> torch.Tensor:
        raise NotImplementedError()