from pathlib import Path
import torch

from .base import BaseTraining
from ..features import BaseFeatures


class SimCLRTraining(BaseTraining):
    temperature:float = 0.15
    
    def __call__(
        self,
        fasta:Path,
        bam:Path,
        feature_module:BaseFeatures,
    ) -> torch.Tensor:
        
        raise NotImplementedError()
