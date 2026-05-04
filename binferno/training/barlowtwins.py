from pathlib import Path
import torch

from ..features import BaseFeatures
from .base import BaseTraining


class BarlowTwinsTraining(BaseTraining):
    use:bool = False

    def __call__(
        self,
        fasta:Path,
        bam:Path,
        feature_module:BaseFeatures,
    ) -> torch.Tensor:
        
        # generate substring for augmentation and coverage for augmentation
        for augmentation in self.augmentations:
            features = feature_module(seq, coverage)

        raise NotImplementedError()