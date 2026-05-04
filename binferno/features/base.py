from abc import ABC
from pathlib import Path

import torch

class BaseFeatures(ABC):
    def __call__(
        fasta:Path,
        bam:Path,
    ) -> torch.Tensor:
        raise NotImplementedError()


