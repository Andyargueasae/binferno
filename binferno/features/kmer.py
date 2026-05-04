import torch
from .base import BaseFeatures


class KmerFeatures(BaseFeatures):
    use:bool = True
    
    def __call__(
        self,
        fasta,
        bam,
    ) -> torch.Tensor:
        raise NotImplementedError()