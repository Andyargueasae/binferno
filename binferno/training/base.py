from abc import ABC
from pathlib import Path
from ..features import BaseFeatures

import torch

class BaseTraining(ABC):
    def __call__(
        self,
        fasta:Path,
        bam:Path,
        feature_module:BaseFeatures,
    ) -> torch.Tensor:
        """ 
        Args:
            features (torch.Tensor): A tensor of shape (num_samples, num_features) containing the features for each sample.

        Returns:
            torch.Tensor: A tensor of shape (num_samples, transformed_features) containing the transformed features for each sample.
        """
        raise NotImplementedError()


