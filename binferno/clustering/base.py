from abc import ABC
from pathlib import Path
import numpy as np

import torch

class BaseClustering(ABC):
    def __call__(
        training_results:torch.Tensor,
    ) -> np.ndarray:
        raise NotImplementedError()


