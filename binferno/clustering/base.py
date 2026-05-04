from pydantic import BaseModel
import numpy as np

import torch

class BaseClustering(BaseModel):
    def __call__(
        training_results:torch.Tensor,
    ) -> np.ndarray:
        raise NotImplementedError()


