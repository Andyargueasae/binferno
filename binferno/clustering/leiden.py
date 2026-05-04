import torch
import numpy as np

from .base import BaseClustering

class LeidenClustering(BaseClustering):
    def __call__(
        self,
        training_results:torch.Tensor,
    ) -> np.ndarray:
        raise NotImplementedError()