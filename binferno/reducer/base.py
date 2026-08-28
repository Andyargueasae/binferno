from pathlib import Path

from pydantic import BaseModel, Field
import loguru
import numpy as np


class BaseReducer(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, data, logger: loguru._logger.Logger = None, threads: int = 1):
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    def l2_normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-12)
