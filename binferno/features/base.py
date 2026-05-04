from pydantic import BaseModel

import torch

class BaseFeatures(BaseModel):
    def __call__(
        self,
        seq:str,
        coverage:int,
    ) -> torch.Tensor:
        raise NotImplementedError()


