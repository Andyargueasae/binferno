import torch

from .base import BaseFeatures
from .corgi import CorgiFeatures
from .coverage import CoverageFeatures
from .kmer import KmerFeatures


class BinfernoFeatures(BaseFeatures):
    corgi: CorgiFeatures
    kmer: KmerFeatures
    coverage: CoverageFeatures

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        results = []
        if self.corgi.use:
            results.append(self.corgi(*args, **kwargs))
        if self.kmer.use:
            results.append(self.kmer(*args, **kwargs))
        if self.coverage.use:
            results.append(self.coverage(*args, **kwargs))

        return torch.cat(results, dim=1)