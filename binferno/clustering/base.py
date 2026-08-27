from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field
import loguru
import pandas as pd
from dataclasses import dataclass

# make this a data object to store clustering results, including cluster assignments and other relevant information.


# Pydantic class replace.
@dataclass
class ClusterResult:
    n_genome: int
    clusters: Dict[
        str, pd.DataFrame
    ]  # a generic dictionary to store all the results for a key and the corresponding dataframes.


class BaseClustering(BaseModel):
    cluster_method: ClusterMethod = Field(
        default=ClusterMethod.leiden,
        description="The clustering method to use.",
    )
    leiden: LeidenClustering
    dbscan: DbscanClustering

    plastid: bool = Field(
        default=False,
        description="Whether to use plastid-specific seed generation and minimum bin size.",
    )
    dimension_reduction: bool = Field(False, description="Whether to apply dimension reduction to embeddings before clustering.")
    pca: bool = Field(False, description="Whether to apply PCA for dimension reduction.")
    umap: bool = Field(False, description="Whether to apply UMAP for dimension reduction.")
    default_dim: int = Field(32, description="Default dimension for dimension reduction.")
    longread: bool = Field(False, description="Whether the input data is from long-read sequencing. This will impact the downstream binning choices of parameters and thresholds.")

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