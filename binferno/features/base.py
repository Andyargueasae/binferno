from pydantic import BaseModel
import pandas as pd
from pathlib import Path


class BaseFeatureBuilder(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, contigs: Path, bam: Path, output_dir: Path = None, **kwargs):
        raise NotImplementedError(
            "The __call__ method must be implemented by the subclass."
        )
