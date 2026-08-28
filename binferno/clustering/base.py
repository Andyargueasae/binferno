from pathlib import Path
from typing import Dict
from enum import Enum

from pydantic import BaseModel, Field
import loguru
import pandas as pd
from dataclasses import dataclass


# Pydantic class replace.
@dataclass
class ClusterResult:
    n_genome: int
    clusters: Dict[str, pd.DataFrame]


class ClusterMethod(Enum):
    leiden = "leiden"
    dbscan = "dbscan"


class BaseClustering(BaseModel):
    cluster_method: ClusterMethod = Field(
        default=ClusterMethod.leiden,
        description="The clustering method to use.",
    )

    plastid: bool = Field(
        default=False,
        description="Whether to use plastid-specific seed generation and minimum bin size.",
    )
    longread: bool = Field(
        False,
        description="Whether the input data is from long-read sequencing. This will impact the downstream binning choices of parameters and thresholds.",
    )

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int = 1,
        plastid: bool = False,
    ) -> ClusterResult:
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def l2_normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-12)
