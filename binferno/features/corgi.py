import torch
from .base import BaseFeatures


class CorgiFeatures(BaseFeatures):
    use:bool = True

    def __call__(
        self,
        seq:str,
        coverage:int,
    ) -> torch.Tensor:
        raise NotImplementedError()