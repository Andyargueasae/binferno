from pathlib import Path
import torch

from ..features import BaseFeatures
from .base import BaseTraining


class BarlowTwinsTraining(BaseTraining):
    def __call__(
        self,
        fasta:Path,
        bam:Path,
        feature_module:BaseFeatures,
    ) -> torch.Tensor:
        
        # generate augmentations and get features using `feature_module`
        for augmentation in self.augmentations:
            features = feature_module(fasta, bam)

        raise NotImplementedError()