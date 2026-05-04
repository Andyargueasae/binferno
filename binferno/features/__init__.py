import torch
from pydantic import Field

from .base import BaseFeatures
from .corgi import CorgiFeatures
from .coverage import CoverageFeatures
from .kmer import KmerFeatures


class BinfernoFeatures(BaseFeatures):
    use_corgi:bool = Field(default=True, description="Whether to use Corgi to generate features for training.")
    corgi: CorgiFeatures
    use_kmer:bool = Field(default=True, description="Whether to use k-mer features for training.")
    kmer: KmerFeatures
    use_coverage:bool = Field(default=True, description="Whether to use coverage features for training.")
    coverage: CoverageFeatures

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        results = []
        if self.use_corgi:
            results.append(self.corgi(*args, **kwargs))
        if self.use_kmer:
            results.append(self.kmer(*args, **kwargs))
        if self.use_coverage:
            results.append(self.coverage(*args, **kwargs))

        return torch.cat(results, dim=1)