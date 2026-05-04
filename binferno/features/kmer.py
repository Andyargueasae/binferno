import torch
from .base import BaseFeatures


class KmerFeatures(BaseFeatures):    
    def __call__(
        self,
        fasta,
        bam,
    ) -> torch.Tensor:
        raise NotImplementedError()